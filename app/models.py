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


class WD14Config(BaseModel):
    """Config for WD14-family tagger models (ONNX image classifiers that
    output booru-style tags with confidence scores, as an alternative to a
    vision LLM writing a natural-language caption).

    `model` accepts either one of the built-in presets (see
    app/wd14_tagger.py MODEL_PRESETS) or a raw Hugging Face repo id for any
    compatible WD14-family tagger. The first use of a given model downloads
    it (cached afterwards by huggingface_hub in its own cache dir) -- no
    local server/API key needed, unlike the vision-LLM path.
    """

    model: str = "wd-vit-tagger-v3"
    general_threshold: float = Field(default=0.35, ge=0, le=1)
    character_threshold: float = Field(default=0.85, ge=0, le=1)


class CaptioningConfig(BaseModel):
    enabled: bool = False
    method: Literal["vision_llm", "wd14"] = "vision_llm"
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="moondream"))
    wd14: WD14Config = Field(default_factory=WD14Config)


class FilterConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png", "webp"])
    min_width: int = Field(default=200, ge=0)
    min_height: int = Field(default=200, ge=0)
    # Matches the underlying `ddgs` library's own default ("moderate") -- the
    # search engine's SafeSearch filtering was previously always on with no
    # way to change it, which silently drops results for anyone building a
    # dataset that isn't strictly SFW. "on" is stricter filtering, "moderate"
    # is the DuckDuckGo default, "off" disables it.
    safesearch: Literal["on", "moderate", "off"] = "moderate"


class SearchConfig(BaseModel):
    """Which image search backend to use, and booru-specific extras.

    - duckduckgo: default, no config needed (proxies Bing's image index).
    - yandex: unofficial scraping, historically laxer filtering than DDG/Google.
    - google: unofficial scraping -- confirmed unreliable in testing (Google
      blocks plain HTTP scraping aggressively); expect frequent zero results.
    - instagram: downloads images from public profiles instead of searching
      by keyword -- each line in the queries box is a profile URL/@handle/
      username instead of a search term. Needs a logged-in `instaloader`
      session: confirmed live that both the anonymous API Instagram's own
      web app calls, and plain HTML scraping of the profile page (no
      embedded post data left in the markup -- it's a client-rendered shell
      now), are dead ends -- the API returned HTTP 429 on the very first
      request from a fresh process. `instagram_username` + optionally
      `instagram_session_file` point at a session created once via
      `instaloader --login=<username>` in a terminal; without a session,
      requests are anonymous and essentially guaranteed to be rate-limited.
      `filters.safesearch` has no Instagram equivalent and is ignored.
    - booru: Danbooru-API-family boards (e621/gelbooru/rule34/danbooru).
      Content is explicitly rating-tagged rather than hidden behind a
      SafeSearch toggle, so `filters.safesearch` maps onto a rating: tag
      instead of a provider-side filter flag.
    """

    provider: Literal["duckduckgo", "yandex", "google", "instagram", "booru"] = "duckduckgo"
    booru_site: Literal["e621", "gelbooru", "rule34", "danbooru"] = "e621"
    # gelbooru/rule34 use an api_key + user_id pair; danbooru uses login + api_key
    # (as HTTP Basic Auth). e621 needs none of these.
    booru_api_key: Optional[str] = None
    booru_user_id: Optional[str] = None
    booru_login: Optional[str] = None
    # The Instagram account whose session to load (the session file itself holds
    # the actual login cookies -- never a raw password, see InstagramProvider).
    # If instagram_session_file is left unset, instaloader's own default path
    # for this username is used (same path `instaloader --login=...` writes to).
    instagram_username: Optional[str] = None
    instagram_session_file: Optional[str] = None


class JobCreateRequest(BaseModel):
    queries: list[str]
    n_per_query: int = Field(default=10, ge=1)
    output_folder: str
    query_subfolders: bool = True
    llm_expansion: LLMExpansionConfig = Field(default_factory=LLMExpansionConfig)
    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
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
    method: Literal["vision_llm", "wd14"] = "vision_llm"
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="moondream"))
    wd14: WD14Config = Field(default_factory=WD14Config)
    recursive: bool = False
    overwrite: bool = False
    trigger: TriggerWordConfig = Field(default_factory=TriggerWordConfig)


class CaptionFilesRequest(BaseModel):
    """Caption a specific, user-picked set of files -- a multi-select in the
    gallery -- independent of any download job or whole-folder scan. Always
    (re)writes the caption for exactly the files given, regardless of
    whether one already existed: picking files and hitting "Caption
    selected" is itself the overwrite decision."""

    paths: list[str]
    method: Literal["vision_llm", "wd14"] = "vision_llm"
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="moondream"))
    wd14: WD14Config = Field(default_factory=WD14Config)
    trigger: TriggerWordConfig = Field(default_factory=TriggerWordConfig)


class DeleteFilesRequest(BaseModel):
    """Delete a specific, user-picked set of files -- a multi-select in the
    gallery. Removed files (and their sidecar .txt caption, if any) go to
    the OS Recycle Bin via send2trash, not a permanent delete."""

    paths: list[str]


class DedupFolderRequest(BaseModel):
    """Find images in a folder with byte-identical content (exact sha256
    match) and remove every copy but one -- e.g. after downloading the same
    query from multiple search providers. Removed files go to the OS Recycle
    Bin (via send2trash), not a permanent delete -- this is a bulk action on
    a user's dataset, so it stays recoverable."""

    folder: str
    recursive: bool = False
