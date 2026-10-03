from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.models import SearchConfig
from app.search import build_search_provider
from app.search import instagram as instagram_module
from app.search.booru import BooruProvider
from app.search.duckduckgo import DuckDuckGoProvider
from app.search.google import GoogleProvider
from app.search.instagram import InstagramProvider
from app.search.yandex import YandexProvider


class TestBuildSearchProvider:
    def test_duckduckgo_is_the_default(self):
        assert isinstance(build_search_provider(SearchConfig()), DuckDuckGoProvider)

    def test_yandex(self):
        assert isinstance(build_search_provider(SearchConfig(provider="yandex")), YandexProvider)

    def test_google(self):
        assert isinstance(build_search_provider(SearchConfig(provider="google")), GoogleProvider)

    def test_instagram(self):
        assert isinstance(build_search_provider(SearchConfig(provider="instagram")), InstagramProvider)

    def test_instagram_passes_username_and_session_file_through(self, monkeypatch):
        captured = {}

        class FakeLoader:
            context = object()

            def load_session_from_file(self, username, filename=None):
                captured["username"] = username
                captured["filename"] = filename

        fake_instaloader = SimpleNamespace(
            Instaloader=lambda **kw: FakeLoader(),
            Profile=SimpleNamespace(from_username=lambda *a: None),
        )
        monkeypatch.setattr(instagram_module, "instaloader", fake_instaloader)

        provider = build_search_provider(
            SearchConfig(provider="instagram", instagram_username="me", instagram_session_file="/x/session-me")
        )
        assert isinstance(provider, InstagramProvider)
        assert captured == {"username": "me", "filename": "/x/session-me"}

    def test_booru_passes_site_and_credentials_through(self):
        provider = build_search_provider(
            SearchConfig(
                provider="booru",
                booru_site="gelbooru",
                booru_api_key="KEY",
                booru_user_id="42",
                booru_login="me",
            )
        )
        assert isinstance(provider, BooruProvider)
        assert provider.site == "gelbooru"
        assert provider.api_key == "KEY"
        assert provider.user_id == "42"
        assert provider.login == "me"

    def test_booru_defaults_to_e621(self):
        provider = build_search_provider(SearchConfig(provider="booru"))
        assert provider.site == "e621"
