from __future__ import annotations

import httpx
import pytest

from app import local_models


_real_async_client = httpx.AsyncClient


def patch_async_client(monkeypatch, handler) -> None:
    """local_models.list_server_models() constructs its own httpx.AsyncClient
    internally, so route it through a MockTransport instead of a real socket.

    Note: `local_models.httpx` is the very same module object as `httpx`
    imported here, so patching `local_models.httpx.AsyncClient` patches the
    name globally -- the replacement must call the saved original class, not
    `httpx.AsyncClient` again, or it recurses into itself.
    """

    def fake_async_client(**kwargs):
        return _real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(local_models.httpx, "AsyncClient", fake_async_client)


class TestListServerModels:
    async def test_dict_shaped_response(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, json={"data": [{"id": "b-model"}, {"id": "a-model"}]})

        patch_async_client(monkeypatch, handler)
        result = await local_models.list_server_models("http://fake/v1", None)
        assert result == ["a-model", "b-model"]

    async def test_bare_list_response_does_not_crash(self, monkeypatch):
        """Regression test: some OpenAI-compatible servers return a bare JSON
        list instead of {"data": [...]}. Calling .get() on that used to raise
        AttributeError."""

        async def handler(request):
            return httpx.Response(200, json=["b-model", "a-model"])

        patch_async_client(monkeypatch, handler)
        result = await local_models.list_server_models("http://fake/v1", None)
        assert result == ["a-model", "b-model"]

    async def test_dedupes_model_ids(self, monkeypatch):
        async def handler(request):
            return httpx.Response(200, json={"data": [{"id": "same"}, {"id": "same"}]})

        patch_async_client(monkeypatch, handler)
        result = await local_models.list_server_models("http://fake/v1", None)
        assert result == ["same"]

    async def test_sends_bearer_token_when_api_key_given(self, monkeypatch):
        seen = {}

        async def handler(request):
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"data": []})

        patch_async_client(monkeypatch, handler)
        await local_models.list_server_models("http://fake/v1", "my-key")
        assert seen["auth"] == "Bearer my-key"

    async def test_raises_on_http_error(self, monkeypatch):
        async def handler(request):
            return httpx.Response(500)

        patch_async_client(monkeypatch, handler)
        with pytest.raises(httpx.HTTPStatusError):
            await local_models.list_server_models("http://fake/v1", None)


class TestScanFolderForModels:
    def test_finds_gguf_files_recursively(self, tmp_path):
        (tmp_path / "publisherA" / "modelA").mkdir(parents=True)
        (tmp_path / "publisherA" / "modelA" / "weights.gguf").write_bytes(b"x")
        (tmp_path / "top-level.gguf").write_bytes(b"x")

        result = local_models.scan_folder_for_models(str(tmp_path))
        assert result == ["publisherA/modelA/weights.gguf", "top-level.gguf"]

    def test_skips_mmproj_files(self, tmp_path):
        (tmp_path / "mmproj-model.gguf").write_bytes(b"x")
        (tmp_path / "mmproj-MODEL.gguf").write_bytes(b"x")  # case-insensitive
        (tmp_path / "real.gguf").write_bytes(b"x")

        result = local_models.scan_folder_for_models(str(tmp_path))
        assert result == ["real.gguf"]

    def test_nonexistent_folder_returns_empty_list(self, tmp_path):
        result = local_models.scan_folder_for_models(str(tmp_path / "does-not-exist"))
        assert result == []

    def test_folder_with_no_gguf_files_returns_empty_list(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hi")
        assert local_models.scan_folder_for_models(str(tmp_path)) == []
