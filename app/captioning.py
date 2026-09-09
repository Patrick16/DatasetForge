from __future__ import annotations

from typing import Optional, Protocol

from . import wd14_tagger
from .llm_client import LLMClient, build_trigger_instruction
from .models import LLMConfig, TriggerWordConfig, WD14Config


class Captioner(Protocol):
    """Common interface both captioning backends implement, so jobs.py can
    call either one the same way without caring which it got."""

    async def caption(self, image_bytes: bytes, mime: str = "image/jpeg") -> str: ...
    async def aclose(self) -> None: ...


class VisionLLMCaptioner:
    """Wraps LLMClient.caption_image() -- a vision LLM writes a natural-
    language caption. Trigger word (if any) is baked in as an extra prompt
    instruction at construction time."""

    def __init__(self, llm_config: LLMConfig, trigger: Optional[TriggerWordConfig] = None):
        self._client = LLMClient(llm_config)
        self._instruction = build_trigger_instruction(trigger) if trigger is not None else None

    async def caption(self, image_bytes: bytes, mime: str = "image/jpeg") -> str:
        return await self._client.caption_image(image_bytes, mime, extra_instruction=self._instruction)

    async def aclose(self) -> None:
        await self._client.aclose()


class WD14Captioner:
    """Wraps wd14_tagger.tag_image() -- a local ONNX classifier outputs a
    comma-separated booru-style tag list instead of a sentence. A trigger
    word doesn't have a "role" to play in a flat tag list the way it does in
    a sentence, so it's just prepended as the first tag (the common
    LoRA/Dreambooth training convention)."""

    def __init__(self, config: WD14Config, trigger: Optional[TriggerWordConfig] = None):
        self._config = config
        self._trigger_tag = None
        if trigger is not None and trigger.enabled:
            word = trigger.word.strip()
            self._trigger_tag = word or None

    async def caption(self, image_bytes: bytes, mime: str = "image/jpeg") -> str:
        tags = await wd14_tagger.tag_image(
            image_bytes,
            model=self._config.model,
            general_threshold=self._config.general_threshold,
            character_threshold=self._config.character_threshold,
        )
        if self._trigger_tag:
            return f"{self._trigger_tag}, {tags}" if tags else self._trigger_tag
        return tags

    async def aclose(self) -> None:
        pass  # no connection to release -- inference is local and synchronous


def build_captioner(
    method: str,
    llm_config: LLMConfig,
    wd14_config: WD14Config,
    trigger: Optional[TriggerWordConfig] = None,
) -> Captioner:
    """Takes the individual fields rather than a whole config object, since
    the two call sites (a download job's nested `captioning.*` vs. the
    standalone caption-folder request's top-level `method`/`llm`/`wd14`)
    shape them differently."""
    if method == "wd14":
        return WD14Captioner(wd14_config, trigger)
    return VisionLLMCaptioner(llm_config, trigger)
