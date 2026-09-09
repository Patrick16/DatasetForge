from __future__ import annotations

import pytest

from app.models import SearchConfig
from app.search import build_search_provider
from app.search.booru import BooruProvider
from app.search.duckduckgo import DuckDuckGoProvider
from app.search.google import GoogleProvider
from app.search.yandex import YandexProvider


class TestBuildSearchProvider:
    def test_duckduckgo_is_the_default(self):
        assert isinstance(build_search_provider(SearchConfig()), DuckDuckGoProvider)

    def test_yandex(self):
        assert isinstance(build_search_provider(SearchConfig(provider="yandex")), YandexProvider)

    def test_google(self):
        assert isinstance(build_search_provider(SearchConfig(provider="google")), GoogleProvider)

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
