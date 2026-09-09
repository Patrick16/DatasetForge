from __future__ import annotations

import asyncio
import logging

from .base import ImageResult

logger = logging.getLogger(__name__)

try:
    # newer PyPI name (the "duckduckgo-search" project renamed its package)
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS  # type: ignore


class DuckDuckGoProvider:
    """Image search backed by DuckDuckGo's (unofficial) image search. No API key required."""

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.images(query, max_results=n, safesearch=safesearch))

        try:
            raw = await asyncio.to_thread(_search)
        except Exception:
            logger.exception("DuckDuckGo image search failed for query %r", query)
            return []

        results: list[ImageResult] = []
        for item in raw:
            url = item.get("image")
            if not url:
                continue
            results.append(
                ImageResult(
                    url=url,
                    title=item.get("title"),
                    width=item.get("width"),
                    height=item.get("height"),
                    source_page=item.get("url"),
                )
            )
        return results
