from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .download import IMAGE_EXTENSIONS


def list_folder_images(root: Path, recursive: bool = True) -> list[dict]:
    """List every image already sitting in a folder, paired with its caption
    (the sibling `<name>.txt`, if one exists) and light metadata (file size,
    pixel dimensions) -- the data the "active folder" gallery and its
    per-image modal are built from.

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

        width = height = None
        try:
            # A lazy open -- this only parses the header, not the full pixel
            # data, so it stays cheap even over a folder with a few thousand
            # images.
            with Image.open(p) as img:
                width, height = img.size
        except (OSError, UnidentifiedImageError):
            pass  # corrupt/unreadable file -- still list it, just without dimensions

        try:
            size = p.stat().st_size
        except OSError:
            size = None

        items.append({
            "name": p.name, "path": str(p), "caption": caption,
            "size": size, "width": width, "height": height,
        })
    items.sort(key=lambda d: d["name"])
    return items
