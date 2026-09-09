from __future__ import annotations

import io

import pytest
from PIL import Image

from app import model_registry


@pytest.fixture(autouse=True)
def _clear_model_registry():
    """model_registry is process-global mutable state (which local models this
    app has used); LLMClient._chat_message writes to it on every call, so
    tests must not leak entries into each other."""
    model_registry.clear()
    yield
    model_registry.clear()


def make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (300, 300)) -> bytes:
    """Build a real, valid image file in memory (Pillow needs real pixel data,
    not a stub, to run format/size checks in download.py)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 40, 200)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes("JPEG", (300, 300))


@pytest.fixture
def small_jpeg_bytes() -> bytes:
    return make_image_bytes("JPEG", (50, 50))


@pytest.fixture
def png_bytes() -> bytes:
    return make_image_bytes("PNG", (300, 300))
