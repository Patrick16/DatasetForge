from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.search import instagram as instagram_module
from app.search.instagram import InstagramProvider, extract_username


class FakeSidecarNode:
    def __init__(self, display_url: str, is_video: bool = False):
        self.display_url = display_url
        self.is_video = is_video


class FakePost:
    def __init__(
        self,
        shortcode: str,
        typename: str = "GraphImage",
        url: str | None = None,
        is_video: bool = False,
        caption: str | None = None,
        sidecar: list[FakeSidecarNode] | None = None,
    ):
        self.shortcode = shortcode
        self.typename = typename
        self.url = url
        self.is_video = is_video
        self.caption = caption
        self._sidecar = sidecar or []

    def get_sidecar_nodes(self):
        return iter(self._sidecar)


class FakeProfile:
    def __init__(self, posts: list[FakePost], pagination_error: Exception | None = None):
        self._posts = posts
        self._pagination_error = pagination_error

    def get_posts(self):
        def gen():
            yield from self._posts
            if self._pagination_error:
                raise self._pagination_error

        return gen()


class FakeInstaloaderModule:
    """Stand-in for the `instaloader` package, used as a module-level import
    in production code."""

    def __init__(
        self,
        profile: FakeProfile | None = None,
        raise_on_fetch: Exception | None = None,
        raise_on_load_session: Exception | None = None,
    ):
        self._profile = profile
        self._raise = raise_on_fetch
        self._raise_on_load_session = raise_on_load_session
        self.last_username: str | None = None
        self.loaded_session: tuple[str, str | None] | None = None
        self.instaloader_kwargs: dict | None = None
        self.Profile = SimpleNamespace(from_username=self._from_username)

    def Instaloader(self, **kwargs):
        self.instaloader_kwargs = kwargs
        return SimpleNamespace(context=object(), load_session_from_file=self._load_session_from_file)

    def _load_session_from_file(self, username, filename=None):
        if self._raise_on_load_session:
            raise self._raise_on_load_session
        self.loaded_session = (username, filename)

    def _from_username(self, context, username):
        if self._raise:
            raise self._raise
        self.last_username = username
        return self._profile


def patch_instaloader(monkeypatch, **kwargs) -> FakeInstaloaderModule:
    fake = FakeInstaloaderModule(**kwargs)
    monkeypatch.setattr(instagram_module, "instaloader", fake)
    return fake


class TestExtractUsername:
    def test_plain_username(self):
        assert extract_username("someone") == "someone"

    def test_at_handle(self):
        assert extract_username("@someone") == "someone"

    def test_profile_url(self):
        assert extract_username("https://www.instagram.com/someone/") == "someone"

    def test_profile_url_with_query_string(self):
        assert extract_username("https://instagram.com/someone/?hl=en") == "someone"

    def test_strips_surrounding_whitespace(self):
        assert extract_username("  someone  ") == "someone"

    def test_rejects_a_post_link(self):
        assert extract_username("https://instagram.com/p/ABC123/") == ""

    def test_rejects_a_reel_link(self):
        assert extract_username("https://www.instagram.com/reel/ABC123/") == ""

    def test_rejects_the_explore_page(self):
        assert extract_username("https://instagram.com/explore/") == ""


class TestInstagramProvider:
    async def test_single_image_posts(self, monkeypatch):
        posts = [
            FakePost("abc", url="https://example.com/a.jpg", caption="Hello world\nsecond line"),
            FakePost("def", url="https://example.com/b.jpg"),
        ]
        fake = patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 10)

        assert fake.last_username == "someone"
        assert [r.url for r in results] == ["https://example.com/a.jpg", "https://example.com/b.jpg"]
        assert results[0].title == "Hello world"
        assert results[0].source_page == "https://www.instagram.com/p/abc/"
        assert results[1].title is None

    async def test_videos_are_skipped(self, monkeypatch):
        posts = [
            FakePost("vid", url="https://example.com/v.jpg", is_video=True),
            FakePost("img", url="https://example.com/a.jpg"),
        ]
        patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 10)
        assert [r.url for r in results] == ["https://example.com/a.jpg"]

    async def test_sidecar_carousel_contributes_every_image(self, monkeypatch):
        posts = [
            FakePost(
                "carousel",
                typename="GraphSidecar",
                sidecar=[
                    FakeSidecarNode("https://example.com/1.jpg"),
                    FakeSidecarNode("https://example.com/2.jpg", is_video=True),
                    FakeSidecarNode("https://example.com/3.jpg"),
                ],
            ),
        ]
        patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 10)
        assert [r.url for r in results] == ["https://example.com/1.jpg", "https://example.com/3.jpg"]

    async def test_stops_once_n_is_reached(self, monkeypatch):
        posts = [FakePost(str(i), url=f"https://example.com/{i}.jpg") for i in range(5)]
        patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 2)
        assert len(results) == 2

    async def test_accepts_a_profile_url_as_the_query(self, monkeypatch):
        fake = patch_instaloader(monkeypatch, profile=FakeProfile([]))
        await InstagramProvider().search("https://instagram.com/someone/", 10)
        assert fake.last_username == "someone"

    async def test_empty_query_returns_empty_list_without_fetching(self, monkeypatch):
        fake = patch_instaloader(monkeypatch, profile=FakeProfile([]))
        results = await InstagramProvider().search("   ", 10)
        assert results == []
        assert fake.last_username is None

    async def test_fetch_failure_returns_empty_list_instead_of_raising(self, monkeypatch):
        patch_instaloader(monkeypatch, raise_on_fetch=RuntimeError("login required"))
        results = await InstagramProvider().search("someone", 10)
        assert results == []

    async def test_whitespace_only_caption_does_not_crash(self, monkeypatch):
        """Regression test: a caption that's non-empty but all whitespace
        used to raise IndexError (''.splitlines() -> [] -> [0]), which
        discarded every result for the whole profile."""
        posts = [
            FakePost("a", url="https://example.com/a.jpg", caption="   \n  "),
            FakePost("b", url="https://example.com/b.jpg"),
        ]
        patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 10)
        assert [r.url for r in results] == ["https://example.com/a.jpg", "https://example.com/b.jpg"]
        assert results[0].title is None

    async def test_a_single_malformed_post_does_not_discard_the_rest(self, monkeypatch):
        """Regression test: an error processing one post (here, a caption
        that isn't even a string) used to propagate out of _fetch and lose
        every image already collected from other posts in the same profile."""
        posts = [
            FakePost("good1", url="https://example.com/1.jpg"),
            FakePost("bad", url="https://example.com/bad.jpg", caption=12345),  # truthy, no .strip()
            FakePost("good2", url="https://example.com/2.jpg"),
        ]
        patch_instaloader(monkeypatch, profile=FakeProfile(posts))

        results = await InstagramProvider().search("someone", 10)
        assert [r.url for r in results] == ["https://example.com/1.jpg", "https://example.com/2.jpg"]

    async def test_a_pagination_failure_returns_images_already_collected(self, monkeypatch):
        """Regression test: a failure fetching a later page of posts (e.g. a
        rate limit hit mid-profile) used to discard every image already
        collected from earlier pages."""
        posts = [FakePost("a", url="https://example.com/a.jpg"), FakePost("b", url="https://example.com/b.jpg")]
        patch_instaloader(
            monkeypatch, profile=FakeProfile(posts, pagination_error=RuntimeError("429 too many requests"))
        )

        results = await InstagramProvider().search("someone", 10)
        assert [r.url for r in results] == ["https://example.com/a.jpg", "https://example.com/b.jpg"]

    async def test_disables_iphone_support(self, monkeypatch):
        """iphone_support defaults to True in instaloader, which makes every
        post/sidecar-node fetch an extra HTTP call for a 'higher quality' URL
        this app never uses -- doubling request volume against an endpoint
        already confirmed to rate-limit on the first request."""
        fake = patch_instaloader(monkeypatch, profile=FakeProfile([]))
        InstagramProvider()
        assert fake.instaloader_kwargs["iphone_support"] is False


class TestInstagramProviderSession:
    def test_no_username_skips_session_loading(self, monkeypatch):
        fake = patch_instaloader(monkeypatch)
        InstagramProvider()
        assert fake.loaded_session is None

    def test_loads_session_for_the_given_username(self, monkeypatch):
        fake = patch_instaloader(monkeypatch)
        InstagramProvider(username="me")
        assert fake.loaded_session == ("me", None)

    def test_passes_an_explicit_session_file_through(self, monkeypatch):
        fake = patch_instaloader(monkeypatch)
        InstagramProvider(username="me", session_file="/custom/path/session-me")
        assert fake.loaded_session == ("me", "/custom/path/session-me")

    def test_missing_session_file_raises_a_clear_error(self, monkeypatch):
        patch_instaloader(monkeypatch, raise_on_load_session=FileNotFoundError())

        with pytest.raises(RuntimeError, match="instaloader --login=me"):
            InstagramProvider(username="me")

    def test_corrupted_session_file_also_raises_a_clear_error(self, monkeypatch):
        """Regression test: only FileNotFoundError used to be caught here, so
        an existing-but-unreadable/corrupted session file (a different
        exception type) surfaced as a raw low-level traceback instead of the
        actionable 'run instaloader --login=...' message."""
        patch_instaloader(monkeypatch, raise_on_load_session=ValueError("bad session data"))

        with pytest.raises(RuntimeError, match="instaloader --login=me"):
            InstagramProvider(username="me")

    def test_session_file_without_username_raises_a_clear_error(self, monkeypatch):
        """Regression test: a session file is tied to one account, so giving
        the file without the username used to be silently ignored -- the
        provider fell back to anonymous access with no indication why."""
        fake = patch_instaloader(monkeypatch)

        with pytest.raises(RuntimeError, match="without a username"):
            InstagramProvider(session_file="/custom/path/session-me")
        assert fake.loaded_session is None
