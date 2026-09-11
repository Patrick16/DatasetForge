from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from .search.base import ImageResult

# Pillow format name -> file extension we write to disk.
EXT_MAP = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "webp": "webp",
    "gif": "gif",
    "bmp": "bmp",
}

# File suffixes treated as images elsewhere (caption-folder scanning, dedup
# scanning) -- kept alongside EXT_MAP since both describe "what counts as an
# image this app handles", just for different directions (write vs. scan).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\n\r\t]')


def sanitize_folder_name(name: str) -> str:
    cleaned = _INVALID_CHARS.sub("_", name).strip().strip(".")
    return cleaned[:100] or "query"


@dataclass
class DownloadResult:
    ok: bool
    url: str
    path: Path | None = None
    error: str | None = None
    duplicate: bool = False
    # True when this was rejected by our own format/min-size filters (i.e.
    # working as configured) rather than failing outright (network error, bad
    # data, ...) -- callers use this to report "filtered" separately from
    # "errors" so it's clear results were deliberately excluded, not that
    # something broke or that content is silently being blocked.
    filtered: bool = False
    content: bytes | None = None
    content_type: str | None = None


async def download_image(
    client: httpx.AsyncClient,
    result: ImageResult,
    dest_dir: Path,
    index: int,
    name_prefix: str,
    allowed_formats: set[str],
    min_width: int,
    min_height: int,
    seen_hashes: set[str],
    timeout: float = 15.0,
) -> DownloadResult:
    try:
        resp = await client.get(result.url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get("content-type", "")

        img = Image.open(io.BytesIO(content))
        img.verify()
        # verify() leaves the image unusable for further ops -> reopen.
        img = Image.open(io.BytesIO(content))
        fmt = (img.format or "").lower()
        ext = EXT_MAP.get(fmt)
        if ext is None or ext not in allowed_formats:
            return DownloadResult(ok=False, url=result.url, filtered=True, error=f"format '{fmt}' not allowed")

        width, height = img.size
        if width < min_width or height < min_height:
            return DownloadResult(
                ok=False, url=result.url, filtered=True, error=f"too small ({width}x{height})"
            )

        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_hashes:
            return DownloadResult(ok=False, url=result.url, duplicate=True, error="duplicate content")
        seen_hashes.add(digest)

        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{name_prefix}img_{index:04d}.{ext}"
        path = dest_dir / filename
        path.write_bytes(content)

        return DownloadResult(
            ok=True,
            url=result.url,
            path=path,
            content=content,
            content_type=content_type or f"image/{ext}",
        )
    except Exception as e:
        return DownloadResult(ok=False, url=result.url, error=str(e))
