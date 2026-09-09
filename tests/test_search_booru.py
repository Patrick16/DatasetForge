from __future__ import annotations

import httpx
import pytest

from app.search import booru as booru_module
from app.search.booru import BooruProvider

_real_async_client = httpx.AsyncClient


def patch_async_client(monkeypatch, handler) -> None:
    def fake_async_client(**kwargs):
        return _real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(booru_module.httpx, "AsyncClient", fake_async_client)


class TestE621:
    async def test_maps_result_fields(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["params"] = dict(request.url.params)
            return httpx.Response(
                200,
                json={
                    "posts": [
                        {
                            "id": 123,
                            "file": {"url": "https://static1.e621.net/a.jpg", "width": 800, "height": 600},
                            "tags": {"general": ["cat", "outside"]},
                        }
                    ]
                },
            )

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="e621").search("cat", 5)

        assert len(results) == 1
        r = results[0]
        assert r.url == "https://static1.e621.net/a.jpg"
        assert r.width == 800
        assert r.height == 600
        assert r.source_page == "https://e621.net/posts/123"
        assert "cat" in r.title

    async def test_posts_with_no_file_url_are_skipped(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, json={"posts": [{"id": 1, "file": {"url": None}}]})

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="e621").search("cat", 5)
        assert results == []

    async def test_safesearch_maps_to_rating_tag(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["tags"] = request.url.params.get("tags")
            return httpx.Response(200, json={"posts": []})

        patch_async_client(monkeypatch, handler)
        await BooruProvider(site="e621").search("cat", 5, safesearch="on")
        assert "rating:s" in captured["tags"]

        await BooruProvider(site="e621").search("cat", 5, safesearch="off")
        assert captured["tags"] == "cat"


class TestGelbooruStyle:
    async def test_maps_result_fields(self, monkeypatch):
        async def handler(request):
            return httpx.Response(
                200,
                json={"post": [{"id": 1, "file_url": "https://img.gelbooru.com/a.jpg", "width": 500, "height": 500, "tags": "cat"}]},
            )

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="gelbooru").search("cat", 5)

        assert len(results) == 1
        assert results[0].url == "https://img.gelbooru.com/a.jpg"
        assert results[0].title == "cat"

    async def test_api_key_and_user_id_are_sent_when_configured(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"post": []})

        patch_async_client(monkeypatch, handler)
        await BooruProvider(site="gelbooru", api_key="KEY", user_id="42").search("cat", 5)

        assert captured["params"]["api_key"] == "KEY"
        assert captured["params"]["user_id"] == "42"

    async def test_no_credentials_means_no_auth_params_sent(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"post": []})

        patch_async_client(monkeypatch, handler)
        await BooruProvider(site="gelbooru").search("cat", 5)

        assert "api_key" not in captured["params"]
        assert "user_id" not in captured["params"]

    async def test_bare_list_response_is_handled(self, monkeypatch):
        """Some gelbooru-style APIs return a bare JSON array for 0 results
        instead of {"post": []}."""

        async def handler(request):
            return httpx.Response(200, json=[])

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="rule34").search("cat", 5)
        assert results == []


class TestDanbooru:
    async def test_maps_result_fields(self, monkeypatch):
        async def handler(request):
            return httpx.Response(
                200,
                json=[{"id": 1, "file_url": "https://danbooru.donmai.us/a.jpg", "image_width": 900, "image_height": 1200}],
            )

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="danbooru").search("cat", 5)
        assert results[0].url == "https://danbooru.donmai.us/a.jpg"
        assert results[0].width == 900

    async def test_sends_basic_auth_when_login_and_key_are_set(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=[])

        patch_async_client(monkeypatch, handler)
        await BooruProvider(site="danbooru", login="me", api_key="secret").search("cat", 5)
        assert captured["auth"] is not None and captured["auth"].startswith("Basic ")

    async def test_no_credentials_means_no_auth_header(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=[])

        patch_async_client(monkeypatch, handler)
        await BooruProvider(site="danbooru").search("cat", 5)
        assert captured["auth"] is None


class TestBooruErrorHandling:
    async def test_http_error_returns_empty_list_instead_of_raising(self, monkeypatch):
        async def handler(request):
            return httpx.Response(403)

        patch_async_client(monkeypatch, handler)
        results = await BooruProvider(site="danbooru").search("cat", 5)
        assert results == []

    async def test_unknown_site_returns_empty_list(self):
        results = await BooruProvider(site="not-a-real-site").search("cat", 5)
        assert results == []
