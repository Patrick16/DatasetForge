from __future__ import annotations

from app import captioning as captioning_module
from app.captioning import VisionLLMCaptioner, WD14Captioner, build_captioner
from app.models import LLMConfig, TriggerWordConfig, WD14Config


class FakeLLMClient:
    instances: list["FakeLLMClient"] = []

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.caption_calls: list[tuple[bytes, str, str | None]] = []
        FakeLLMClient.instances.append(self)

    async def caption_image(self, image_bytes, mime="image/jpeg", extra_instruction=None):
        self.caption_calls.append((image_bytes, mime, extra_instruction))
        return "a vision caption"

    async def aclose(self):
        self.closed = True


class TestBuildCaptioner:
    def test_vision_llm_is_the_default(self):
        captioner = build_captioner("vision_llm", LLMConfig(), WD14Config())
        assert isinstance(captioner, VisionLLMCaptioner)

    def test_wd14_method_returns_wd14_captioner(self):
        captioner = build_captioner("wd14", LLMConfig(), WD14Config())
        assert isinstance(captioner, WD14Captioner)


class TestVisionLLMCaptioner:
    async def test_delegates_to_llm_client_caption_image(self, monkeypatch):
        FakeLLMClient.instances.clear()
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        captioner = VisionLLMCaptioner(LLMConfig())
        result = await captioner.caption(b"bytes", "image/png")

        assert result == "a vision caption"
        client = FakeLLMClient.instances[0]
        assert client.caption_calls == [(b"bytes", "image/png", None)]

    async def test_trigger_instruction_is_computed_once_and_reused(self, monkeypatch):
        FakeLLMClient.instances.clear()
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        trigger = TriggerWordConfig(enabled=True, word="zog", role="subject")
        captioner = VisionLLMCaptioner(LLMConfig(), trigger)
        await captioner.caption(b"a")
        await captioner.caption(b"b")

        client = FakeLLMClient.instances[0]
        instructions = [c[2] for c in client.caption_calls]
        assert instructions[0] is not None and "zog" in instructions[0]
        assert instructions[0] == instructions[1]  # same instruction both times, not recomputed

    async def test_no_trigger_means_no_instruction(self, monkeypatch):
        FakeLLMClient.instances.clear()
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        captioner = VisionLLMCaptioner(LLMConfig(), TriggerWordConfig(enabled=False))
        await captioner.caption(b"a")

        assert FakeLLMClient.instances[0].caption_calls[0][2] is None

    async def test_aclose_delegates_to_the_underlying_client(self, monkeypatch):
        FakeLLMClient.instances.clear()
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        captioner = VisionLLMCaptioner(LLMConfig())
        await captioner.aclose()

        assert FakeLLMClient.instances[0].closed is True


class TestWD14Captioner:
    async def test_delegates_to_tag_image_with_configured_thresholds(self, monkeypatch):
        captured = {}

        async def fake_tag_image(image_bytes, model, general_threshold, character_threshold):
            captured["args"] = (image_bytes, model, general_threshold, character_threshold)
            return "solo, blue hair"

        monkeypatch.setattr(captioning_module.wd14_tagger, "tag_image", fake_tag_image)

        config = WD14Config(model="wd-swinv2-tagger-v3", general_threshold=0.4, character_threshold=0.8)
        captioner = WD14Captioner(config)
        result = await captioner.caption(b"bytes", "image/jpeg")

        assert result == "solo, blue hair"
        assert captured["args"] == (b"bytes", "wd-swinv2-tagger-v3", 0.4, 0.8)

    async def test_trigger_word_is_prepended_as_the_first_tag(self, monkeypatch):
        async def fake_tag_image(image_bytes, model, general_threshold, character_threshold):
            return "solo, blue hair"

        monkeypatch.setattr(captioning_module.wd14_tagger, "tag_image", fake_tag_image)

        trigger = TriggerWordConfig(enabled=True, word="zog", role="style")  # role is irrelevant for WD14
        captioner = WD14Captioner(WD14Config(), trigger)
        result = await captioner.caption(b"bytes")

        assert result == "zog, solo, blue hair"

    async def test_trigger_word_alone_when_no_tags_pass_threshold(self, monkeypatch):
        async def fake_tag_image(image_bytes, model, general_threshold, character_threshold):
            return ""

        monkeypatch.setattr(captioning_module.wd14_tagger, "tag_image", fake_tag_image)

        trigger = TriggerWordConfig(enabled=True, word="zog")
        captioner = WD14Captioner(WD14Config(), trigger)
        result = await captioner.caption(b"bytes")

        assert result == "zog"

    async def test_disabled_trigger_does_not_prepend_anything(self, monkeypatch):
        async def fake_tag_image(image_bytes, model, general_threshold, character_threshold):
            return "solo"

        monkeypatch.setattr(captioning_module.wd14_tagger, "tag_image", fake_tag_image)

        captioner = WD14Captioner(WD14Config(), TriggerWordConfig(enabled=False, word="zog"))
        result = await captioner.caption(b"bytes")

        assert result == "solo"

    async def test_aclose_is_a_no_op(self):
        captioner = WD14Captioner(WD14Config())
        await captioner.aclose()  # must not raise
