from __future__ import annotations

import hashlib
from pathlib import Path

from .download import IMAGE_EXTENSIONS

_CHUNK_SIZE = 1 << 20  # 1 MiB -- read large files in chunks instead of all at once


def hash_file(path: Path) -> str:
    """sha256 of a file's raw bytes -- exact-content match, same notion of
    "duplicate" used during downloads (download.py's seen_hashes)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_groups(root: Path, recursive: bool = False) -> list[list[Path]]:
    """Group images under `root` by exact content hash, returning only the
    groups that actually have more than one member (i.e. the real
    duplicates) -- each group sorted so the caller can treat index 0 as "the
    one to keep" deterministically (alphabetically first) and the rest as
    redundant copies to remove.

    Groups themselves are also sorted (by their keeper's name) so repeated
    runs against an unchanged folder report results in the same order.
    """
    pattern = root.rglob("*") if recursive else root.glob("*")
    by_hash: dict[str, list[Path]] = {}
    for p in pattern:
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        digest = hash_file(p)
        by_hash.setdefault(digest, []).append(p)

    groups = [sorted(paths, key=lambda p: p.name) for paths in by_hash.values() if len(paths) > 1]
    groups.sort(key=lambda g: g[0].name)
    return groups
