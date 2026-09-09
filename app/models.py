from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Config for any OpenAI-compatible chat completions endpoint.

    Covers Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1)
    and any cloud provider that speaks the OpenAI chat-completions API.
    """

    provider: Literal["ollama", "lmstudio", "cloud"] = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    model: str = "qwen2.5:1.5b-instruct"
    # Local CPU inference can be very slow (measured: 100+s for a single short
    # completion on a 7B model with no GPU offload) -- default generously.
    timeout_seconds: float = Field(default=180.0, gt=0)
    # Some local "thinking" models (Qwen3-family fine-tunes especially) burn
    # their whole response budget on hidden reasoning and never answer, even
    # when asked to via /no_think or chat_template_kwargs.enable_thinking --
    # see LLMClient._chat_message for the prefill trick this enables.
    disable_reasoning: bool = False


class LLMExpansionConfig(BaseModel):
    enabled: bool = False
    variations_per_query: int = Field(default=3, ge=0)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class CaptioningConfig(BaseModel):
    enabled: bool = False
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="moondream"))


class FilterConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png", "webp"])
    min_width: int = Field(default=200, ge=0)
    min_height: int = Field(default=200, ge=0)


class JobCreateRequest(BaseModel):
    queries: list[str]
    n_per_query: int = Field(default=10, ge=1)
    output_folder: str
    query_subfolders: bool = True
    llm_expansion: LLMExpansionConfig = Field(default_factory=LLMExpansionConfig)
    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    concurrency: int = Field(default=8, ge=1)


class TriggerWordConfig(BaseModel):
    """A trigger word to weave into every caption -- common for LoRA/Dreambooth
    training sets, where captions need to consistently reference a token that
    stands in for a specific subject, style, or action.

    What the word *means* changes how it should be used in the sentence, hence
    `role` -- and since that 3-way split won't cover everything, `custom_instruction`
    lets the user just describe it themselves (with `{word}` as a placeholder).
    """

    enabled: bool = False
    word: str = ""
    role: Literal["subject", "style", "action", "custom"] = "subject"
    custom_instruction: Optional[str] = None


class CaptionFolderRequest(BaseModel):
    """Caption every image already sitting in a folder, independent of any
    download job -- e.g. an existing dataset you want captions for."""

    folder: str
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="moondream"))
    recursive: bool = False
    overwrite: bool = False
    trigger: TriggerWordConfig = Field(default_factory=TriggerWordConfig)
