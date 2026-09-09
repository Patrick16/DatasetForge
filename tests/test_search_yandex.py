from __future__ import annotations

import httpx
import pytest

from app.search import yandex as yandex_module
from app.search.yandex import YandexProvider

_real_async_client = httpx.AsyncClient


def patch_async_client(monkeypatch, handler) -> None:
    def fake_async_client(**kwargs):
        return _real_async_client(transport=httpx.MockTransport(handler), cookies=kwargs.get("cookies"))

    monkeypatch.setattr(yandex_module.httpx, "AsyncClient", fake_async_client)


def page_html(urls: list[str]) -> str:
    items = "".join(f'img_href&quot;:&quot;{u}&quot;,' for u in urls)
    return f"<html><body>{items}</body></html>"


class TestYandexProvider:
    async def test_extracts_image_urls_from_a_single_page(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, text=page_html(["https://example.com/a.jpg", "https://example.com/b.jpg"]))

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 10)

        assert {r.url for r in results} == {"https://example.com/a.jpg", "https://example.com/b.jpg"}

    async def test_html_entities_in_urls_are_unescaped(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, text=page_html(["https://example.com/a.jpg?x=1&amp;y=2"]))

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 10)
        assert results[0].url == "https://example.com/a.jpg?x=1&y=2"

    async def test_stops_once_n_is_reached_without_fetching_more_pages(self, monkeypatch):
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, text=page_html([f"https://example.com/{call_count}-{i}.jpg" for i in range(25)]))

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 10)

        assert call_count == 1  # one page already exceeds n=10, no need for a second fetch
        assert len(results) == 10  # ... but the final result list is still capped at n

    async def test_paginates_when_first_page_has_too_few_results(self, monkeypatch):
        pages = [
            page_html(["https://example.com/1.jpg", "https://example.com/2.jpg"]),
            page_html(["https://example.com/3.jpg", "https://example.com/4.jpg"]),
        ]
        call_count = 0

        async def handler(request):
            nonlocal call_count
            resp = httpx.Response(200, text=pages[call_count])
            call_count += 1
            return resp

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 4)

        assert call_count == 2
        assert len(results) == 4

    async def test_stops_when_a_page_returns_no_new_urls(self, monkeypatch):
        """Guards against looping forever if pagination stops producing
        anything new (e.g. markup changed, or fewer results exist than n)."""
        call_count = 0

        async def handler(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, text=page_html(["https://example.com/same.jpg"]))

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 50)

        # page 1 yields the one (new) URL; page 2 re-yields the same URL with
        # nothing new, which is what actually triggers the stop -- so it does
        # get fetched once before the loop gives up.
        assert call_count == 2
        assert len(results) == 1

    async def test_family_cookie_off_is_sent_for_safesearch_off(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["cookie"] = request.headers.get("cookie")
            return httpx.Response(200, text=page_html([]))

        patch_async_client(monkeypatch, handler)
        await YandexProvider().search("cats", 5, safesearch="off")
        assert "family=0" in (captured["cookie"] or "")

    async def test_no_family_cookie_sent_for_moderate(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["cookie"] = request.headers.get("cookie")
            return httpx.Response(200, text=page_html([]))

        patch_async_client(monkeypatch, handler)
        await YandexProvider().search("cats", 5, safesearch="moderate")
        assert not captured["cookie"]

    async def test_request_failure_returns_empty_list_instead_of_raising(self, monkeypatch):
        async def handler(request):
            raise httpx.ConnectError("blocked")

        patch_async_client(monkeypatch, handler)
        results = await YandexProvider().search("cats", 5)
        assert results == []
