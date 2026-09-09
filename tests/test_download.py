from __future__ import annotations

import httpx
import pytest

from app.download import download_image, sanitize_folder_name
from app.search.base import ImageResult


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestSanitizeFolderName:
    def test_replaces_invalid_characters(self):
        assert sanitize_folder_name('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"

    def test_strips_surrounding_whitespace_and_dots(self):
        assert sanitize_folder_name("  my query..  ") == "my query"

    def test_truncates_to_100_chars(self):
        assert len(sanitize_folder_name("x" * 500)) == 100

    def test_empty_input_falls_back_to_query(self):
        assert sanitize_folder_name("") == "query"
        assert sanitize_folder_name("   ...   ") == "query"


class TestDownloadImage:
    async def test_successful_download_writes_file(self, tmp_path, jpeg_bytes):
        async def handler(request):
            return httpx.Response(200, content=jpeg_bytes, headers={"content-type": "image/jpeg"})

        result = ImageResult(url="http://example.com/a.jpg")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.ok is True
        assert res.path is not None
        assert res.path.exists()
        assert res.path.read_bytes() == jpeg_bytes
        assert res.path.name == "img_0001.jpg"
        assert res.content == jpeg_bytes

    async def test_name_prefix_is_applied(self, tmp_path, jpeg_bytes):
        async def handler(request):
            return httpx.Response(200, content=jpeg_bytes)

        result = ImageResult(url="http://example.com/a.jpg")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 3, "cats_", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.path.name == "cats_img_0003.jpg"

    async def test_disallowed_format_is_rejected(self, tmp_path, png_bytes):
        async def handler(request):
            return httpx.Response(200, content=png_bytes)

        result = ImageResult(url="http://example.com/a.png")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.ok is False
        assert res.duplicate is False
        assert "not allowed" in res.error
        assert list(tmp_path.iterdir()) == []

    async def test_undersized_image_is_rejected(self, tmp_path, small_jpeg_bytes):
        async def handler(request):
            return httpx.Response(200, content=small_jpeg_bytes)

        result = ImageResult(url="http://example.com/a.jpg")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.ok is False
        assert "too small" in res.error

    async def test_duplicate_content_is_rejected(self, tmp_path, jpeg_bytes):
        async def handler(request):
            return httpx.Response(200, content=jpeg_bytes)

        seen: set[str] = set()
        result = ImageResult(url="http://example.com/a.jpg")
        async with make_client(handler) as client:
            first = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, seen
            )
            second = await download_image(
                client, ImageResult(url="http://example.com/b.jpg"), tmp_path, 2, "",
                {"jpg", "jpeg"}, 200, 200, seen,
            )

        assert first.ok is True
        assert second.ok is False
        assert second.duplicate is True

    async def test_http_error_status_is_reported(self, tmp_path):
        async def handler(request):
            return httpx.Response(404)

        result = ImageResult(url="http://example.com/missing.jpg")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.ok is False
        assert res.duplicate is False
        assert res.error

    async def test_non_image_content_is_reported(self, tmp_path):
        async def handler(request):
            return httpx.Response(200, content=b"not an image")

        result = ImageResult(url="http://example.com/a.jpg")
        async with make_client(handler) as client:
            res = await download_image(
                client, result, tmp_path, 1, "", {"jpg", "jpeg"}, 200, 200, set()
            )

        assert res.ok is False
        assert list(tmp_path.iterdir()) == []
