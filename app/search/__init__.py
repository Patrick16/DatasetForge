from __future__ import annotations

from ..models import SearchConfig
from .base import ImageSearchProvider
from .booru import BooruProvider
from .duckduckgo import DuckDuckGoProvider
from .google import GoogleProvider
from .instagram import InstagramProvider
from .yandex import YandexProvider


def build_search_provider(config: SearchConfig) -> ImageSearchProvider:
    """Construct the right ImageSearchProvider for a job's SearchConfig."""
    if config.provider == "duckduckgo":
        return DuckDuckGoProvider()
    if config.provider == "yandex":
        return YandexProvider()
    if config.provider == "google":
        return GoogleProvider()
    if config.provider == "instagram":
        return InstagramProvider(
            username=config.instagram_username,
            session_file=config.instagram_session_file,
        )
    if config.provider == "booru":
        return BooruProvider(
            site=config.booru_site,
            api_key=config.booru_api_key,
            user_id=config.booru_user_id,
            login=config.booru_login,
        )
    raise ValueError(f"Unknown search provider: {config.provider!r}")
