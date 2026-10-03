from __future__ import annotations

import asyncio
import logging
import re

import instaloader

from .base import ImageResult

logger = logging.getLogger(__name__)

_PROFILE_URL_RE = re.compile(r"instagram\.com/([^/?#]+)", re.IGNORECASE)

# Instagram path segments that look like a username in a URL but are actually
# a reserved top-level route (a post/reel link, the explore page, ...) --
# without this, pasting e.g. "instagram.com/p/ABC123/" would try to look up
# a profile literally named "p" and fail with a confusing provider-level
# error instead of just being treated as "no profile here".
_RESERVED_PATH_SEGMENTS = {"p", "reel", "reels", "tv", "stories", "explore", "accounts", "direct"}


def extract_username(profile: str) -> str:
    """Pull a bare username out of a profile URL, an @handle, or a plain username.

    Returns "" if the input doesn't look like a profile at all (e.g. a post/
    reel/explore link), so callers can treat it the same as an empty query.
    """
    text = profile.strip().lstrip("@")
    match = _PROFILE_URL_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip("/ ")
    if text.lower() in _RESERVED_PATH_SEGMENTS:
        return ""
    return text


class InstagramProvider:
    """Downloads images from a public Instagram profile's posts, given a
    profile URL, @handle, or bare username as the "query" (the queries box
    is reused as a list of profiles -- n_per_query becomes "images per
    profile", query_subfolders becomes "one subfolder per profile").

    Confirmed live (2026-10) that anonymous access is a dead end: Instagram's
    own web app calls a JSON API for profile data, which returned HTTP 429
    on the very first request from a fresh process -- and the profile page's
    raw HTML has no post data left in it to scrape either (it's a
    client-rendered shell; the old `_sharedData`/`edge_owner_to_timeline_media`
    embedded-JSON approach stopped working years ago). A logged-in
    `instaloader` session is therefore required in practice, not just an
    accuracy nice-to-have.

    `username` is the Instagram account whose *session* to load -- created
    once, outside this app, by running `instaloader --login=<username>` in a
    terminal (handles 2FA/checkpoint prompts interactively, which this app
    has no way to do). We only ever load that saved session file; this class
    never sees or sends a raw password. `session_file` is optional and only
    needed if that session was saved somewhere other than instaloader's own
    default path for the given username.

    `safesearch` has no Instagram equivalent and is ignored. Videos are
    skipped (this app only handles still images); a carousel post
    contributes every image inside it.
    """

    def __init__(self, username: str | None = None, session_file: str | None = None):
        if session_file and not username:
            raise RuntimeError(
                "An Instagram session file was given without a username -- a session "
                "file belongs to one account, so its username is needed too (left blank, "
                "the session file is ignored and requests run anonymously, which is "
                "essentially guaranteed to be rate-limited)."
            )

        # max_connection_attempts=1: instaloader's default retry behaviour
        # sleeps (observed: 600+ seconds) and retries on HTTP 429 rather than
        # raising -- fine for its CLI, but it would silently freeze a web job
        # worker for minutes with no way to cancel. Fail fast instead and let
        # the normal per-query "warning" event surface the real error.
        # iphone_support=False: instaloader's default (True) makes Post.url /
        # get_sidecar_nodes() each fire an extra HTTP call per post (to
        # api/v1/media/{id}/info/) to substitute a "higher quality" image URL
        # -- a URL swap this app has no use for (every file is re-measured
        # with Pillow after download regardless), that roughly doubles
        # request volume against an endpoint already confirmed to 429 on the
        # very first request.
        self._loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            max_connection_attempts=1,
            iphone_support=False,
            quiet=True,
        )
        if username:
            try:
                self._loader.load_session_from_file(username, session_file)
            except Exception as e:
                where = f" at {session_file}" if session_file else f" for '{username}'"
                raise RuntimeError(
                    f"No usable Instagram session found{where} (missing, or not readable). "
                    f"Run `instaloader --login={username}` once in a terminal to create one, "
                    "then try again."
                ) from e

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        username = extract_username(query)
        if not username:
            return []
        try:
            return await asyncio.to_thread(self._fetch, username, n)
        except Exception:
            logger.exception("Instagram profile fetch failed for %r", query)
            return []

    def _fetch(self, username: str, n: int) -> list[ImageResult]:
        profile = instaloader.Profile.from_username(self._loader.context, username)
        results: list[ImageResult] = []
        posts = profile.get_posts()
        while len(results) < n:
            try:
                post = next(posts)
            except StopIteration:
                break
            except Exception:
                # A failure fetching the *next page* of posts (e.g. a rate
                # limit hit partway through a profile) shouldn't throw away
                # images already collected from earlier pages -- return what
                # we have instead of letting search()'s blanket except
                # discard it all.
                logger.exception(
                    "Instagram pagination stopped early for %r after %d image(s)", username, len(results)
                )
                break
            try:
                self._collect_post_images(post, n, results)
            except Exception:
                # Same reasoning, one level down: a single malformed post
                # (e.g. a whitespace-only caption) shouldn't cost every other
                # post's already-collected images.
                logger.exception("Skipping an Instagram post for %r due to an error", username)

        return results

    def _collect_post_images(self, post, n: int, results: list[ImageResult]) -> None:
        page_url = f"https://www.instagram.com/p/{post.shortcode}/"
        title = None
        if post.caption:
            lines = post.caption.strip().splitlines()
            title = lines[0][:200] if lines else None

        if post.typename == "GraphSidecar":
            for node in post.get_sidecar_nodes():
                if len(results) >= n:
                    return
                if node.is_video:
                    continue
                results.append(ImageResult(url=node.display_url, title=title, source_page=page_url))
        elif not post.is_video:
            results.append(ImageResult(url=post.url, title=title, source_page=page_url))
