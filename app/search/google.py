from __future__ import annotations

import logging
import re

import httpx

from .base import ImageResult

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Google embeds original image URLs as plain quoted strings inside inline
# <script> JSON blobs on the results page; this pattern doesn't depend on
# Google's exact variable naming (which changes often, unlike Yandex's more
# stable img_href key) -- just on the URL appearing as a quoted string
# ending in a common image extension, and not pointing back at google.com
# itself (its own UI chrome/icons).
_IMG_URL_RE = re.compile(r'"(https?://(?![^"]*google\.com)[^"]+?\.(?:jpg|jpeg|png|webp|gif))"', re.IGNORECASE)

_SAFE_PARAM = {"off": "off", "moderate": "images", "on": "active"}


class GoogleProvider:
    """Image search by scraping Google Images.

    Confirmed live during development: Google returned an HTTP 429 bot-check
    on the *very first* request from this environment, with full
    browser-like headers and no prior request history -- Google is far more
    aggressive about blocking plain HTTP scraping than DuckDuckGo or Yandex.
    Treat this as "may work occasionally from a residential IP with light,
    spaced-out usage", not a reliable default. A block surfaces as zero
    results plus a log warning (and the job's own "only found N/M" warning),
    not a crash.
    """

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        safe = _SAFE_PARAM.get(safesearch, "images")
        headers = {
            "User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as client:
                resp = await client.get(
                    "https://www.google.com/search",
                    params={"q": query, "udm": "2", "safe": safe, "num": min(max(n, 1), 100)},
                )
        except Exception:
            logger.exception("Google image search request failed for query %r", query)
            return []

        if resp.status_code == 429 or "detected unusual traffic" in resp.text.lower():
            logger.warning(
                "Google blocked/rate-limited the request for query %r (HTTP %s) -- "
                "this is common for Google specifically; try Yandex or DuckDuckGo instead.",
                query, resp.status_code,
            )
            return []
        if resp.status_code != 200:
            logger.warning("Google image search returned HTTP %s for query %r", resp.status_code, query)
            return []

        seen: set[str] = set()
        results: list[ImageResult] = []
        for url in _IMG_URL_RE.findall(resp.text):
            if url in seen:
                continue
            seen.add(url)
            results.append(ImageResult(url=url))
            if len(results) >= n:
                break
        return results
