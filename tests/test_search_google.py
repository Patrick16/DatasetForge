from __future__ import annotations

import httpx
import pytest

from app.search import google as google_module
from app.search.google import GoogleProvider

_real_async_client = httpx.AsyncClient


def patch_async_client(monkeypatch, handler) -> None:
    def fake_async_client(**kwargs):
        return _real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(google_module.httpx, "AsyncClient", fake_async_client)


def page_html(urls: list[str]) -> str:
    quoted = "".join(f'"{u}",' for u in urls)
    return f"<html><script>var data=[{quoted}];</script></html>"


class TestGoogleProvider:
    async def test_extracts_image_urls(self, monkeypatch):
        async def handler(request):
            return httpx.Response(
                200, text=page_html(["https://example.com/a.jpg", "https://example.com/b.png"])
            )

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 10)
        assert {r.url for r in results} == {"https://example.com/a.jpg", "https://example.com/b.png"}

    async def test_google_domain_urls_are_excluded(self, monkeypatch):
        """Google's own UI chrome/icon URLs shouldn't be mistaken for results."""

        async def handler(request):
            return httpx.Response(
                200,
                text=page_html(
                    ["https://www.google.com/logo.png", "https://example.com/real.jpg"]
                ),
            )

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 10)
        assert {r.url for r in results} == {"https://example.com/real.jpg"}

    async def test_stops_at_n_results(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, text=page_html([f"https://example.com/{i}.jpg" for i in range(20)]))

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 5)
        assert len(results) == 5

    async def test_rate_limit_response_returns_empty_list_not_an_error(self, monkeypatch):
        async def handler(request):
            return httpx.Response(429, text="unusual traffic detected")

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 5)
        assert results == []

    async def test_captcha_page_with_200_status_is_also_treated_as_blocked(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, text="Our systems have detected unusual traffic from your network.")

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 5)
        assert results == []

    async def test_other_non_200_status_returns_empty_list(self, monkeypatch):
        async def handler(request):
            return httpx.Response(503)

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 5)
        assert results == []

    async def test_request_failure_returns_empty_list_instead_of_raising(self, monkeypatch):
        async def handler(request):
            raise httpx.ConnectError("blocked")

        patch_async_client(monkeypatch, handler)
        results = await GoogleProvider().search("cats", 5)
        assert results == []

    async def test_safesearch_off_maps_to_safe_off_param(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["safe"] = request.url.params.get("safe")
            return httpx.Response(200, text=page_html([]))

        patch_async_client(monkeypatch, handler)
        await GoogleProvider().search("cats", 5, safesearch="off")
        assert captured["safe"] == "off"
