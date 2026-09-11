from __future__ import annotations

from pathlib import Path

from .download import IMAGE_EXTENSIONS


def list_folder_images(root: Path, recursive: bool = True) -> list[dict]:
    """List every image already sitting in a folder, paired with its caption
    (the sibling `<name>.txt`, if one exists) -- the data the "active folder"
    gallery in the UI is built from.

    Recursive by default: unlike the caption/dedup actions (where "include
    subfolders" is a deliberate, user-controlled choice about what gets
    *written to* or *removed from*), browsing is read-only, and per-query
    subfolders (see `query_subfolders` on a download job) are a normal part
    of a folder's contents the gallery should just show.
    """
    pattern = root.rglob("*") if recursive else root.glob("*")
    items = []
    for p in pattern:
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = None
        txt_path = p.with_suffix(".txt")
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8").strip() or None
            except OSError:
                caption = None
        items.append({"name": p.name, "path": str(p), "caption": caption})
    items.sort(key=lambda d: d["name"])
    return items
