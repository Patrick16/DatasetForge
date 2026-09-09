from __future__ import annotations

import pytest

from app.search import duckduckgo
from app.search.duckduckgo import DuckDuckGoProvider


class FakeDDGS:
    """Stand-in for ddgs.DDGS, used as a context manager in production code."""

    def __init__(self, results=None, raise_on_search: Exception | None = None):
        self._results = results or []
        self._raise = raise_on_search

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def images(self, query, max_results):
        if self._raise:
            raise self._raise
        return self._results


class TestDuckDuckGoProvider:
    async def test_maps_result_fields(self, monkeypatch):
        raw = [
            {
                "image": "http://example.com/a.jpg",
                "title": "A cat",
                "width": 640,
                "height": 480,
                "url": "http://example.com/page-a",
            }
        ]
        monkeypatch.setattr(duckduckgo, "DDGS", lambda: FakeDDGS(raw))

        results = await DuckDuckGoProvider().search("cats", 5)
        assert len(results) == 1
        r = results[0]
        assert r.url == "http://example.com/a.jpg"
        assert r.title == "A cat"
        assert r.width == 640
        assert r.height == 480
        assert r.source_page == "http://example.com/page-a"

    async def test_skips_results_without_an_image_url(self, monkeypatch):
        raw = [{"title": "no image field"}, {"image": "http://example.com/b.jpg"}]
        monkeypatch.setattr(duckduckgo, "DDGS", lambda: FakeDDGS(raw))

        results = await DuckDuckGoProvider().search("cats", 5)
        assert len(results) == 1
        assert results[0].url == "http://example.com/b.jpg"

    async def test_empty_results(self, monkeypatch):
        monkeypatch.setattr(duckduckgo, "DDGS", lambda: FakeDDGS([]))
        results = await DuckDuckGoProvider().search("cats", 5)
        assert results == []

    async def test_search_failure_returns_empty_list_instead_of_raising(self, monkeypatch):
        monkeypatch.setattr(
            duckduckgo, "DDGS", lambda: FakeDDGS(raise_on_search=RuntimeError("blocked"))
        )
        results = await DuckDuckGoProvider().search("cats", 5)
        assert results == []
