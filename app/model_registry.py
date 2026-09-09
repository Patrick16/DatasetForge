from __future__ import annotations

"""Tracks which (provider, base_url, model) combinations this process has
actually sent a request to, so the "Unload models" button and the shutdown
hook know what to try to free -- without ever touching a model the user
loaded some other way, outside this app.
"""

_used: set[tuple[str, str, str]] = set()


def note_used(provider: str, base_url: str, model: str) -> None:
    if provider == "cloud" or not model.strip():
        return  # nothing local to track/unload
    _used.add((provider, base_url.rstrip("/"), model))


def all_used() -> list[tuple[str, str, str]]:
    return sorted(_used)


def forget(provider: str, base_url: str, model: str) -> None:
    _used.discard((provider, base_url.rstrip("/"), model))


def clear() -> None:
    _used.clear()
