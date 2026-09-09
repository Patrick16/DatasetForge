from __future__ import annotations

import logging

import httpx

from .base import ImageResult

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Booru sites don't hide a SafeSearch toggle behind an opaque server-side
# filter -- content is explicitly tagged with a rating (naming varies by
# site) that you query directly, so "off" here genuinely means "no rating
# restriction" rather than "hope the filter didn't trigger".
_GELBOORU_STYLE_RATING = {"on": "rating:general", "moderate": "-rating:explicit", "off": ""}
_DANBOORU_RATING = {"on": "rating:g", "moderate": "-rating:e", "off": ""}
_E621_RATING = {"on": "rating:s", "moderate": "-rating:e", "off": ""}

_GELBOORU_STYLE_SITES = {
    "gelbooru": "https://gelbooru.com/index.php",
    "rule34": "https://api.rule34.xxx/index.php",
}


class BooruProvider:
    """Image search against a booru-style board (the Danbooru-API family).

    Confirmed live during development (2026-09):
    - e621: works with zero configuration, no account needed.
    - gelbooru / rule34: now require an api_key + user_id (free account,
      Account -> API Access Credentials) -- both returned HTTP 401/"Missing
      authentication" without one, a change from their historical open access.
    - danbooru: blocked by a Cloudflare bot-check ("Just a moment...") even
      with a real browser User-Agent, regardless of credentials, from at
      least some networks. Implemented (with login+api_key Basic Auth, the
      documented method) but best-effort -- it may simply not work for you.
    """

    def __init__(
        self,
        site: str = "e621",
        api_key: str | None = None,
        user_id: str | None = None,
        login: str | None = None,
    ):
        self.site = site
        self.api_key = api_key
        self.user_id = user_id
        self.login = login

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        try:
            if self.site == "e621":
                return await self._search_e621(query, n, safesearch)
            if self.site in _GELBOORU_STYLE_SITES:
                return await self._search_gelbooru_style(query, n, safesearch)
            if self.site == "danbooru":
                return await self._search_danbooru(query, n, safesearch)
            logger.error("Unknown booru site: %r", self.site)
            return []
        except Exception:
            logger.exception("Booru search failed for query %r on %s", query, self.site)
            return []

    async def _search_e621(self, query: str, n: int, safesearch: str) -> list[ImageResult]:
        rating = _E621_RATING.get(safesearch, "")
        tags = f"{query} {rating}".strip()
        params = {"tags": tags, "limit": min(max(n, 1), 320)}
        # e621 asks API clients to identify themselves in the User-Agent.
        headers = {"User-Agent": "DatasetForge/1.0 (image dataset tool)"}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get("https://e621.net/posts.json", params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for post in data.get("posts", []):
            file_info = post.get("file") or {}
            url = file_info.get("url")
            if not url:
                continue  # deleted/restricted posts have a null file url
            general_tags = (post.get("tags") or {}).get("general", [])
            results.append(
                ImageResult(
                    url=url,
                    title=", ".join(general_tags[:8]) or None,
                    width=file_info.get("width"),
                    height=file_info.get("height"),
                    source_page=f"https://e621.net/posts/{post.get('id')}",
                )
            )
        return results

    async def _search_gelbooru_style(self, query: str, n: int, safesearch: str) -> list[ImageResult]:
        base_url = _GELBOORU_STYLE_SITES[self.site]
        rating = _GELBOORU_STYLE_RATING.get(safesearch, "")
        tags = f"{query} {rating}".strip()
        params: dict[str, str | int] = {
            "page": "dapi", "s": "post", "q": "index", "json": "1",
            "limit": min(max(n, 1), 100), "tags": tags,
        }
        if self.api_key and self.user_id:
            params["api_key"] = self.api_key
            params["user_id"] = self.user_id
        headers = {"User-Agent": _UA}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

        posts = data.get("post", []) if isinstance(data, dict) else (data or [])
        results = []
        for post in posts:
            url = post.get("file_url")
            if not url:
                continue
            results.append(
                ImageResult(
                    url=url,
                    title=post.get("tags"),
                    width=post.get("width"),
                    height=post.get("height"),
                    source_page=f"{base_url}?page=post&s=view&id={post.get('id')}",
                )
            )
        return results

    async def _search_danbooru(self, query: str, n: int, safesearch: str) -> list[ImageResult]:
        rating = _DANBOORU_RATING.get(safesearch, "")
        # Anonymous Danbooru API access is limited to 2 combined tags -- a
        # multi-word query plus the rating tag can exceed that even with
        # credentials on a free account; let the API's own error surface
        # rather than silently truncating what the user typed.
        tags = f"{query} {rating}".strip()
        params = {"tags": tags, "limit": min(max(n, 1), 200)}
        auth = (self.login, self.api_key) if self.login and self.api_key else None
        headers = {"User-Agent": _UA}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get("https://danbooru.donmai.us/posts.json", params=params, auth=auth)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for post in data:
            url = post.get("file_url")
            if not url:
                continue
            results.append(
                ImageResult(
                    url=url,
                    title=post.get("tag_string_general"),
                    width=post.get("image_width"),
                    height=post.get("image_height"),
                    source_page=f"https://danbooru.donmai.us/posts/{post.get('id')}",
                )
            )
        return results
