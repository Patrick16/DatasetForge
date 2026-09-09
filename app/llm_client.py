from __future__ import annotations

import base64
import json

import httpx

from . import model_registry
from .models import LLMConfig, TriggerWordConfig

# How a trigger word should be woven into a caption depends on what it stands
# for -- a fixed identifier for a subject reads very differently in a sentence
# than a style name or an action. {word} is substituted with the actual word.
TRIGGER_ROLE_INSTRUCTIONS: dict[str, str] = {
    "subject": (
        'The word "{word}" is a special identifier for the main subject of this image '
        "(a specific person, character, or object) -- use exactly the word \"{word}\" to refer "
        'to that subject instead of a generic description (e.g. "a photo of {word} smiling" '
        'rather than "a photo of a woman smiling"), while still describing their visible '
        "appearance, pose, clothing, and setting."
    ),
    "style": (
        'The word "{word}" names a specific visual/artistic style. Describe the image\'s '
        'content normally, then note it is rendered in the "{word}" style '
        '(e.g. "a mountain landscape at sunset, in the style of {word}").'
    ),
    "action": (
        'The word "{word}" names a specific action or pose shown in the image. Describe the '
        'subject and setting normally, and use the word "{word}" to name that action/pose '
        '(e.g. "a man {word} in a park").'
    ),
}


def build_trigger_instruction(trigger: TriggerWordConfig) -> str | None:
    """Turn a TriggerWordConfig into an extra instruction appended to the caption prompt."""
    if not trigger.enabled:
        return None
    word = trigger.word.strip()
    if not word:
        return None
    if trigger.role == "custom":
        custom = (trigger.custom_instruction or "").strip()
        if not custom:
            return f'Make sure to include the word "{word}" naturally in the caption.'
        return custom.format(word=word) if "{word}" in custom else f'{custom} (use the exact word "{word}").'
    template = TRIGGER_ROLE_INSTRUCTIONS.get(trigger.role, TRIGGER_ROLE_INSTRUCTIONS["subject"])
    return template.format(word=word)


# The "disable reasoning" prefill trick: appending an already-closed, empty
# <think></think> block as if the model itself had already written it makes
# llama.cpp-based servers (LM Studio, text-generation-webui, koboldcpp, ...)
# treat it as a continuation to build on rather than a fresh turn -- so the
# model starts writing its actual answer immediately, skipping reasoning
# entirely. Measured on a Qwen3.5 fine-tune that ignored /no_think and the
# standard chat_template_kwargs.enable_thinking=False / think=False request
# fields: ~116s (or outright empty, budget exhausted) -> ~1s, reliably.
# Not guaranteed on every backend -- Ollama and cloud APIs may reject a
# trailing assistant message or just treat it as literal conversation history
# instead of a prefill, so this is opt-in (LLMConfig.disable_reasoning), not
# automatic.
REASONING_PREFILL = "<think>\n\n</think>\n\n"


class LLMClient:
    """Thin client for any OpenAI-compatible /chat/completions endpoint.

    Works against Ollama, LM Studio, or a real cloud API (OpenAI, OpenRouter, ...)
    -- whatever base_url/model/api_key the user configured in the UI.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying connection pool. A client may issue many
        requests over a job's lifetime (one per query/image); reusing one
        httpx.AsyncClient instead of opening a fresh one per request avoids
        a new TCP/TLS handshake every time, so this must be called once the
        caller is done with the instance."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _chat_message(self, messages: list[dict], max_tokens: int = 300) -> dict:
        """Return the full assistant message dict (content + any extra fields like reasoning_content)."""
        # Record that this (provider, base_url, model) is actually in use, so
        # the "Unload models" button and the shutdown hook know to try
        # freeing it later -- regardless of whether this particular call
        # succeeds (a failed request can still have caused the server to load
        # the model before erroring out).
        model_registry.note_used(self.config.provider, self.config.base_url, self.config.model)
        if self.config.disable_reasoning:
            messages = [*messages, {"role": "assistant", "content": REASONING_PREFILL}]
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        client = self._http_client()
        resp = await client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]

    async def _chat(self, messages: list[dict], max_tokens: int = 300) -> str:
        message = await self._chat_message(messages, max_tokens=max_tokens)
        return message.get("content") or ""

    async def expand_queries(self, query: str, n: int) -> list[str]:
        if n <= 0:
            return []
        prompt = (
            f'Generate {n} diverse, concise image search queries closely related to: "{query}". '
            f"Return ONLY a JSON array of {n} short strings, no explanation, no markdown."
        )
        message = await self._chat_message([{"role": "user", "content": prompt}], max_tokens=500)
        content = message.get("content") or ""
        items = self._parse_json_list(content, fallback_count=n)
        if items:
            return items

        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            raise ValueError(
                "The model produced no usable answer -- it spent its entire response budget "
                f"on hidden reasoning ({len(reasoning)} chars) instead. This is common with "
                "'thinking' models on short structured-output tasks; try a plain (non-reasoning) "
                "instruct model, or raise the response timeout/token budget."
            )
        snippet = content.strip()[:200]
        raise ValueError(
            "The model returned an empty or unparseable response"
            + (f": {snippet!r}" if snippet else " (no content at all).")
        )

    @staticmethod
    def _parse_json_list(content: str, fallback_count: int) -> list[str]:
        text = content.strip()
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                arr = json.loads(text[start : end + 1])
                items = [str(x).strip() for x in arr if str(x).strip()]
                if items:
                    return items[:fallback_count]
            except Exception:
                pass
        # Fallback: treat each non-empty line as one query.
        lines = [l.strip("-*0123456789. \t") for l in text.splitlines()]
        lines = [l for l in lines if l]
        return lines[:fallback_count]

    async def caption_image(
        self, image_bytes: bytes, mime: str = "image/jpeg", extra_instruction: str | None = None
    ) -> str:
        text = "Describe this image in one concise sentence, suitable as a caption."
        if extra_instruction:
            text += " " + extra_instruction
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        content = await self._chat(messages, max_tokens=120)
        return content.strip()
