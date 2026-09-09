from __future__ import annotations

import html as html_module
import logging
import re

import httpx

from .base import ImageResult

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Each search-result item's markup embeds an HTML-entity-escaped JSON blob
# containing "img_href":"<direct image url>" -- confirmed live: 25 unique,
# valid direct image URLs extracted from a single page fetch. Width/height
# aren't captured here (they'd need parsing a separate compressed blob) --
# not a real loss, since download_image() re-measures every file with
# Pillow after downloading regardless of what the search step reported.
# `.+?` (not `[^&]+?`) because the URL itself commonly contains its own
# HTML-entity-escaped "&amp;" (e.g. query strings) -- excluding every "&"
# would truncate the match right at the first one instead of the real
# "&quot;" terminator; non-greedy `.+?` still stops at the nearest one.
_IMG_HREF_RE = re.compile(r"img_href&quot;:&quot;(https?://.+?)&quot;")

# Yandex's "family filter" (SafeSearch equivalent) is controlled by a
# `family` cookie, not a URL parameter -- confirmed live: family=0 vs
# family=2 on a borderline query returned overlapping but distinct result
# sets (20/25 shared, 5 different), so this genuinely changes what comes
# back rather than being a no-op. No cookie == Yandex's own default.
_FAMILY_COOKIE = {"off": "0", "moderate": None, "on": "2"}

MAX_PAGES = 6  # a single page already yields ~20-25 results


class YandexProvider:
    """Image search by scraping Yandex Images. Unofficial, like DuckDuckGo --
    can break if Yandex changes their page markup."""

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        cookies = {}
        family = _FAMILY_COOKIE.get(safesearch)
        if family is not None:
            cookies["family"] = family

        results: list[ImageResult] = []
        seen: set[str] = set()
        headers = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}

        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers, cookies=cookies) as client:
                page = 0
                while len(results) < n and page < MAX_PAGES:
                    resp = await client.get(
                        "https://yandex.com/images/search",
                        params={"text": query, "p": page},
                    )
                    resp.raise_for_status()
                    new = 0
                    for m in _IMG_HREF_RE.findall(resp.text):
                        url = html_module.unescape(m)
                        if url in seen:
                            continue
                        seen.add(url)
                        results.append(ImageResult(url=url))
                        new += 1
                    if new == 0:
                        break  # ran out of new results (or markup didn't match) -- stop, don't loop forever
                    page += 1
        except Exception:
            logger.exception("Yandex image search failed for query %r", query)
            return results  # whatever was collected before the failure, if anything

        return results[:n] if n else results
