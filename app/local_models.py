from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

# tkinter's Tk() is not thread-safe across concurrent instances; a single-user
# local tool never needs two folder dialogs open at once anyway.
_picker_lock = asyncio.Lock()


def _pick_folder_sync(initial_dir: str | None, title: str) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(initialdir=initial_dir or None, title=title)
    finally:
        root.destroy()
    return folder or None


async def pick_folder(initial_dir: str | None, title: str = "Select folder") -> str | None:
    """Open a native OS folder-picker dialog on the machine running this server.

    Only meaningful when the browser and this backend run on the same machine
    (the normal case for this local tool).
    """
    async with _picker_lock:
        return await asyncio.to_thread(_pick_folder_sync, initial_dir, title)


async def list_server_models(base_url: str, api_key: str | None) -> list[str]:
    """Ask a running OpenAI-compatible endpoint (Ollama, LM Studio, cloud, ...) what models it has."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    # Most OpenAI-compatible servers wrap the list as {"data": [...]}, but some
    # (older/nonstandard) return a bare list -- calling .get() on that would
    # raise AttributeError, so branch on the shape first.
    items = data if isinstance(data, list) else data.get("data", [])
    return sorted({str(m.get("id") if isinstance(m, dict) else m) for m in items if m})


def scan_folder_for_models(folder: str) -> list[str]:
    """Best-effort fallback: recursively find *.gguf files under a local models folder.

    Useful when the model server isn't running yet, or the user just wants to browse
    what's on disk. The returned names are file paths relative to `folder`, which may
    not exactly match the model "key" a server would expose via its API.
    """
    root = Path(folder)
    if not root.is_dir():
        return []
    names: set[str] = set()
    for p in root.rglob("*.gguf"):
        if p.name.lower().startswith("mmproj"):
            continue
        names.add(p.relative_to(root).as_posix())
    return sorted(names)
