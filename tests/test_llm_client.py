from __future__ import annotations

import json

import httpx
import pytest

from app.llm_client import REASONING_PREFILL, LLMClient, build_trigger_instruction
from app.models import LLMConfig, TriggerWordConfig


def client_with_handler(handler, **config_kwargs) -> LLMClient:
    """An LLMClient whose lazily-created httpx.AsyncClient is pre-seeded with a
    MockTransport, so _http_client() reuses it instead of opening a real socket."""
    client = LLMClient(LLMConfig(base_url="http://fake/v1", model="test-model", **config_kwargs))
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


class TestChatMessagePlumbing:
    async def test_authorization_header_sent_when_api_key_set(self):
        seen_headers = {}

        async def handler(request):
            seen_headers.update(request.headers)
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler, api_key="secret-key")
        await client.expand_queries("cats", 1)
        assert seen_headers.get("authorization") == "Bearer secret-key"

    async def test_no_authorization_header_when_no_api_key(self):
        seen_headers = {}

        async def handler(request):
            seen_headers.update(request.headers)
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler)
        await client.expand_queries("cats", 1)
        assert "authorization" not in seen_headers

    async def test_aclose_releases_the_underlying_client(self):
        async def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler)
        await client.expand_queries("cats", 1)
        assert client._client is not None
        await client.aclose()
        assert client._client is None

    async def test_async_context_manager_closes_on_exit(self):
        async def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = LLMClient(LLMConfig(base_url="http://fake/v1"))
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            pass
        assert client._client is None


class TestDisableReasoning:
    """The prefill trick: appending an already-closed <think></think> as if the
    model itself had written it, so a llama.cpp-based server treats it as a
    continuation and the model skips straight to answering. See the
    REASONING_PREFILL comment in app/llm_client.py for the full story --
    measured live against a Qwen3.5 fine-tune that ignored /no_think and the
    standard enable_thinking=False / think=False request fields: an empty
    response after 100+s -> a correct answer in ~1s, reliably."""

    async def test_prefill_message_appended_when_disable_reasoning_true(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler, disable_reasoning=True)
        await client.expand_queries("cats", 1)

        messages = captured["body"]["messages"]
        assert messages[-1] == {"role": "assistant", "content": REASONING_PREFILL}

    async def test_no_prefill_message_when_disable_reasoning_false(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler, disable_reasoning=False)
        await client.expand_queries("cats", 1)

        messages = captured["body"]["messages"]
        assert all(m.get("role") != "assistant" for m in messages)

    async def test_prefill_applies_to_caption_image_too(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "a cat"}}]})

        client = client_with_handler(handler, disable_reasoning=True)
        await client.caption_image(b"bytes")

        messages = captured["body"]["messages"]
        assert messages[-1] == {"role": "assistant", "content": REASONING_PREFILL}


class TestExpandQueries:
    async def test_zero_variations_returns_immediately_without_a_request(self):
        called = False

        async def handler(request):
            nonlocal called
            called = True
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler)
        result = await client.expand_queries("cats", 0)
        assert result == []
        assert called is False

    async def test_parses_json_array_from_content(self):
        async def handler(request):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '["red cat", "blue cat"]'}}]}
            )

        client = client_with_handler(handler)
        result = await client.expand_queries("cats", 2)
        assert result == ["red cat", "blue cat"]

    async def test_falls_back_to_line_splitting_when_not_valid_json(self):
        content = "1. red cat\n2. blue cat\n3. green cat"

        async def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = client_with_handler(handler)
        result = await client.expand_queries("cats", 3)
        assert result == ["red cat", "blue cat", "green cat"]

    async def test_raises_helpful_error_when_only_hidden_reasoning_was_produced(self):
        async def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "", "reasoning_content": "thinking " * 50}}
                    ]
                },
            )

        client = client_with_handler(handler)
        with pytest.raises(ValueError, match="hidden reasoning"):
            await client.expand_queries("cats", 3)

    async def test_raises_error_when_response_is_totally_empty(self):
        async def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

        client = client_with_handler(handler)
        with pytest.raises(ValueError, match="empty or unparseable"):
            await client.expand_queries("cats", 3)

    async def test_sends_the_query_and_count_in_the_prompt(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

        client = client_with_handler(handler)
        await client.expand_queries("mountain lions", 5)
        prompt = captured["body"]["messages"][0]["content"]
        assert "mountain lions" in prompt
        assert "5" in prompt


class TestParseJsonList:
    def test_valid_json_array(self):
        items = LLMClient._parse_json_list('noise ["a", "b", "c"] trailing', fallback_count=3)
        assert items == ["a", "b", "c"]

    def test_json_array_capped_at_fallback_count(self):
        items = LLMClient._parse_json_list('["a", "b", "c"]', fallback_count=2)
        assert items == ["a", "b"]

    def test_unclosed_bracket_falls_back_to_lines(self):
        # No closing "]" at all -> never attempts json.loads, goes straight to
        # per-line splitting of the raw text (stray "[" included -- the line
        # splitter only strips list-marker characters, not brackets).
        items = LLMClient._parse_json_list("[not json\nsecond line", fallback_count=5)
        assert items == ["[not json", "second line"]

    def test_unparseable_json_between_brackets_falls_back_to_lines(self):
        # Brackets are present but the content between them isn't valid JSON
        # -> json.loads raises, caught, and it falls back to the raw lines.
        items = LLMClient._parse_json_list("[red cat, blue cat]", fallback_count=5)
        assert items == ["[red cat, blue cat]"]

    def test_no_brackets_falls_back_to_lines(self):
        items = LLMClient._parse_json_list("- one\n- two\n", fallback_count=5)
        assert items == ["one", "two"]

    def test_empty_content_returns_empty_list(self):
        assert LLMClient._parse_json_list("", fallback_count=5) == []


class TestCaptionImage:
    async def test_sends_image_as_base64_data_url(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": " a red cat "}}]})

        client = client_with_handler(handler)
        caption = await client.caption_image(b"\x89fakepng", mime="image/png")

        assert caption == "a red cat"
        content_parts = captured["body"]["messages"][0]["content"]
        image_part = next(p for p in content_parts if p["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_extra_instruction_is_appended_to_the_prompt(self):
        captured = {}

        async def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = client_with_handler(handler)
        await client.caption_image(b"bytes", extra_instruction="mention the word FOO")

        content_parts = captured["body"]["messages"][0]["content"]
        text_part = next(p for p in content_parts if p["type"] == "text")
        assert "mention the word FOO" in text_part["text"]


class TestBuildTriggerInstruction:
    def test_disabled_returns_none(self):
        assert build_trigger_instruction(TriggerWordConfig(enabled=False, word="zog")) is None

    def test_empty_word_returns_none(self):
        assert build_trigger_instruction(TriggerWordConfig(enabled=True, word="  ")) is None

    def test_subject_role_template(self):
        instr = build_trigger_instruction(
            TriggerWordConfig(enabled=True, word="zog", role="subject")
        )
        assert 'zog' in instr
        assert "main subject" in instr

    def test_style_role_template(self):
        instr = build_trigger_instruction(
            TriggerWordConfig(enabled=True, word="vaporwave", role="style")
        )
        assert "visual/artistic style" in instr

    def test_custom_role_with_placeholder(self):
        instr = build_trigger_instruction(
            TriggerWordConfig(
                enabled=True, word="zog", role="custom", custom_instruction="Always say {word} twice."
            )
        )
        assert instr == "Always say zog twice."

    def test_custom_role_without_placeholder_appends_note(self):
        instr = build_trigger_instruction(
            TriggerWordConfig(
                enabled=True, word="zog", role="custom", custom_instruction="Be concise."
            )
        )
        assert instr == 'Be concise. (use the exact word "zog").'

    def test_custom_role_with_no_instruction_at_all(self):
        instr = build_trigger_instruction(
            TriggerWordConfig(enabled=True, word="zog", role="custom", custom_instruction=None)
        )
        assert 'zog' in instr
