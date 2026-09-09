from __future__ import annotations

import asyncio

import httpx
import pytest

from app import model_control

_real_async_client = httpx.AsyncClient


def patch_async_client(monkeypatch, handler) -> None:
    def fake_async_client(**kwargs):
        return _real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(model_control.httpx, "AsyncClient", fake_async_client)


class FakeProcess:
    def __init__(self, output: bytes, returncode: int):
        self._output = output
        self.returncode = returncode
        self.killed = False

    async def communicate(self):
        return self._output, b""

    def kill(self):
        self.killed = True


class TestUnloadOllamaModel:
    async def test_posts_keep_alive_zero_to_native_generate_endpoint(self, monkeypatch):
        captured = {}

        async def handler(request):
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(200, json={"done_reason": "unload"})

        patch_async_client(monkeypatch, handler)
        message = await model_control.unload_ollama_model("http://localhost:11434/v1", "llama3")

        assert captured["url"] == "http://localhost:11434/api/generate"
        assert b'"keep_alive":0' in captured["body"] or b'"keep_alive": 0' in captured["body"]
        assert "llama3" in message

    async def test_strips_v1_suffix_regardless_of_trailing_slash(self, monkeypatch):
        seen_urls = []

        async def handler(request):
            seen_urls.append(str(request.url))
            return httpx.Response(200, json={})

        patch_async_client(monkeypatch, handler)
        await model_control.unload_ollama_model("http://localhost:11434/v1/", "llama3")
        assert seen_urls == ["http://localhost:11434/api/generate"]

    async def test_raises_on_http_error(self, monkeypatch):
        async def handler(request):
            return httpx.Response(500)

        patch_async_client(monkeypatch, handler)
        with pytest.raises(httpx.HTTPStatusError):
            await model_control.unload_ollama_model("http://localhost:11434/v1", "llama3")


class TestUnloadLmstudioModel:
    async def test_success_runs_lms_unload_with_model_identifier(self, monkeypatch):
        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            return FakeProcess(b"Unloaded.", 0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        message = await model_control.unload_lmstudio_model("qwen3.5-4b")

        assert captured["args"] == ("lms", "unload", "qwen3.5-4b")
        assert message == "Unloaded."

    async def test_nonzero_exit_raises_with_output_as_message(self, monkeypatch):
        async def fake_exec(*args, **kwargs):
            return FakeProcess(b"model not found", 1)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        with pytest.raises(RuntimeError, match="model not found"):
            await model_control.unload_lmstudio_model("nope")

    async def test_missing_lms_cli_raises_helpful_error(self, monkeypatch):
        async def fake_exec(*args, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        with pytest.raises(RuntimeError, match="not found on PATH"):
            await model_control.unload_lmstudio_model("qwen3.5-4b")

    async def test_empty_output_on_success_still_returns_a_message(self, monkeypatch):
        async def fake_exec(*args, **kwargs):
            return FakeProcess(b"", 0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        message = await model_control.unload_lmstudio_model("qwen3.5-4b")
        assert "qwen3.5-4b" in message


class TestUnloadModelDispatch:
    async def test_dispatches_to_ollama(self, monkeypatch):
        called = {}

        async def fake(base_url, model):
            called["args"] = (base_url, model)
            return "ollama done"

        monkeypatch.setattr(model_control, "unload_ollama_model", fake)
        result = await model_control.unload_model("ollama", "http://fake/v1", "llama3")
        assert result == "ollama done"
        assert called["args"] == ("http://fake/v1", "llama3")

    async def test_dispatches_to_lmstudio(self, monkeypatch):
        called = {}

        async def fake(model):
            called["model"] = model
            return "lmstudio done"

        monkeypatch.setattr(model_control, "unload_lmstudio_model", fake)
        result = await model_control.unload_model("lmstudio", "http://fake/v1", "moondream")
        assert result == "lmstudio done"
        assert called["model"] == "moondream"

    async def test_cloud_provider_is_a_no_op_message_not_an_error(self):
        result = await model_control.unload_model("cloud", "https://api.openai.com/v1", "gpt-4o-mini")
        assert "nothing local to unload" in result.lower()
