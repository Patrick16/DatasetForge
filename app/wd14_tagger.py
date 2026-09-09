from __future__ import annotations

import asyncio
import csv
import io
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

# Curated presets -- HF repo ids for SmilingWolf's WD14 v3 tagger family.
# "vit" is the default: smallest download (~55MB) and fastest, while still
# giving coherent tags (confirmed live against a real image during
# development: 0.22s inference on CPU once loaded). The others trade
# download size/speed for accuracy; any other WD14-compatible HF repo id
# (one with a model.onnx + selected_tags.csv) also works, typed in directly.
MODEL_PRESETS: dict[str, str] = {
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}
DEFAULT_MODEL = "wd-vit-tagger-v3"

# A handful of emoticon-style tags conventionally keep their underscores
# (e.g. "^_^") when every other tag has "_" replaced with a space for
# readability -- this list matches the one kohya_ss/taggui use.
_KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}

# selected_tags.csv "category" column: 9=rating, 4=character, 0=general
# (everything else, e.g. copyright/artist tags on some models, is treated
# like "general" here rather than special-cased).
_RATING_CATEGORY = "9"
_CHARACTER_CATEGORY = "4"

# Loaded ONNX sessions + tag tables are cached in-process, keyed by resolved
# model repo id -- (re)loading is the slow part (a network fetch on first
# use, real init cost every time after), and one job can tag many images in
# a row. A failed load is cached too (as the exception itself), so a bad
# model/no internet fails once loudly instead of re-attempting a slow
# network call for every single image in the job.
_cache: dict[str, tuple | Exception] = {}


def resolve_model_repo(model: str) -> str:
    """Accepts either one of the short presets above or a raw HF repo id."""
    return MODEL_PRESETS.get(model, model)


def _load(model: str) -> tuple:
    repo_id = resolve_model_repo(model)
    cached = _cache.get(repo_id)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return cached

    try:
        model_path = hf_hub_download(repo_id, "model.onnx")
        csv_path = hf_hub_download(repo_id, "selected_tags.csv")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        _, height, width, _ = input_info.shape
        output_name = session.get_outputs()[0].name
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        loaded = (session, input_info.name, output_name, int(width), int(height), rows)
    except Exception as e:
        logger.exception("Failed to load WD14 model %r", repo_id)
        _cache[repo_id] = e
        raise
    _cache[repo_id] = loaded
    return loaded


def _preprocess(image_bytes: bytes, width: int, height: int) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")

    # Pad to square (centered, white fill) before resizing, so the aspect
    # ratio isn't distorted by a naive stretch to the model's fixed input size.
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    canvas = canvas.resize((width, height), Image.LANCZOS)

    arr = np.asarray(canvas, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR: these models were trained on BGR input
    return np.expand_dims(arr, axis=0)


def _format_tag(name: str) -> str:
    if name in _KAOMOJI:
        return name
    return name.replace("_", " ")


def _run_sync(image_bytes: bytes, model: str, general_threshold: float, character_threshold: float) -> str:
    session, input_name, output_name, width, height, rows = _load(model)
    arr = _preprocess(image_bytes, width, height)
    probs = session.run([output_name], {input_name: arr})[0][0]

    general: list[tuple[str, float]] = []
    character: list[tuple[str, float]] = []
    for row, p in zip(rows, probs):
        p = float(p)
        cat = row["category"]
        if cat == _RATING_CATEGORY:
            continue
        if cat == _CHARACTER_CATEGORY:
            if p >= character_threshold:
                character.append((row["name"], p))
        elif p >= general_threshold:
            general.append((row["name"], p))

    character.sort(key=lambda x: -x[1])
    general.sort(key=lambda x: -x[1])
    tags = [_format_tag(n) for n, _ in character] + [_format_tag(n) for n, _ in general]
    return ", ".join(tags)


async def tag_image(
    image_bytes: bytes,
    model: str = DEFAULT_MODEL,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
) -> str:
    """Tag an image with a WD14-family ONNX classifier, returning a
    comma-separated booru-style tag string (character tags first, then
    general tags, both ranked by confidence -- highest first).

    The first call for a given model downloads it from Hugging Face (cached
    by huggingface_hub afterwards) and loads the ONNX session, which can take
    a while; every call after that is fast (confirmed live: ~0.2s CPU
    inference for the default "vit" preset once loaded). Runs in a thread
    since both the model load and inference are blocking/CPU-bound.
    """
    return await asyncio.to_thread(_run_sync, image_bytes, model, general_threshold, character_threshold)
