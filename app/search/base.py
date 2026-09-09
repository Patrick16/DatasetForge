from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ImageResult:
    """A single image search hit."""

    url: str
    title: str | None = None
    width: int | None = None
    height: int | None = None
    source_page: str | None = None


class ImageSearchProvider(Protocol):
    """Interface every image search backend must implement."""

    async def search(self, query: str, n: int) -> list[ImageResult]: ...
