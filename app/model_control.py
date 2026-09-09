from __future__ import annotations

import asyncio

import httpx


def _ollama_native_base(base_url: str) -> str:
    """Ollama's OpenAI-compatible base_url ends in /v1; unloading is only
    possible through Ollama's own native API, one level up from there."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b


async def unload_ollama_model(base_url: str, model: str) -> str:
    """Ollama unloads a model immediately when asked to generate with
    keep_alive=0 (confirmed live: response comes back with
    done_reason="unload" and the model drops out of `ollama ps`)."""
    url = _ollama_native_base(base_url) + "/api/generate"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json={"model": model, "keep_alive": 0})
        resp.raise_for_status()
    return f'Asked Ollama to unload "{model}".'


async def unload_lmstudio_model(model: str) -> str:
    """LM Studio's OpenAI-compatible API has no unload endpoint -- the `lms`
    CLI (installed alongside LM Studio, normally on PATH) is the only way."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lms", "unload", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "The 'lms' CLI (bundled with LM Studio) was not found on PATH -- "
            "unload the model manually in the LM Studio app, or add 'lms' to PATH."
        ) from e

    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("Timed out waiting for 'lms unload' to finish.")

    text = out.decode(errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(text or f"'lms unload {model}' exited with code {proc.returncode}")
    return text or f'Unloaded "{model}" via LM Studio.'


async def unload_model(provider: str, base_url: str, model: str) -> str:
    """Dispatch to the right unload mechanism for the given provider.

    Cloud models aren't loaded into local memory by this app, so there's
    nothing to unload there -- this returns a message rather than erroring,
    since it's a normal (not exceptional) outcome.
    """
    if provider == "ollama":
        return await unload_ollama_model(base_url, model)
    if provider == "lmstudio":
        return await unload_lmstudio_model(model)
    return f'"{model}" is a cloud model -- nothing local to unload.'
