from __future__ import annotations

import numpy as np
import pytest

from app import wd14_tagger


class FakeSession:
    """Stand-in for onnxruntime.InferenceSession -- returns a fixed
    probability vector regardless of the actual preprocessed input, so tests
    can focus on the threshold/category/sort/format logic around it."""

    def __init__(self, probs: list[float]):
        self._probs = probs

    def run(self, output_names, input_feed):
        return [np.array([self._probs], dtype=np.float32)]


def make_rows(*entries: tuple[str, str]) -> list[dict]:
    """entries: (name, category) pairs, in the same order as the fake probs."""
    return [{"tag_id": str(i), "name": name, "category": cat, "count": "0"} for i, (name, cat) in enumerate(entries)]


@pytest.fixture(autouse=True)
def clear_cache():
    wd14_tagger._cache.clear()
    yield
    wd14_tagger._cache.clear()


class TestResolveModelRepo:
    def test_known_preset_resolves_to_full_repo_id(self):
        assert wd14_tagger.resolve_model_repo("wd-vit-tagger-v3") == "SmilingWolf/wd-vit-tagger-v3"

    def test_unknown_string_passes_through_unchanged(self):
        assert wd14_tagger.resolve_model_repo("someone/custom-wd14-repo") == "someone/custom-wd14-repo"

    def test_default_model_is_a_known_preset(self):
        assert wd14_tagger.DEFAULT_MODEL in wd14_tagger.MODEL_PRESETS


class TestFormatTag:
    def test_underscore_replaced_with_space(self):
        assert wd14_tagger._format_tag("blue_hair") == "blue hair"

    def test_kaomoji_keeps_its_underscore(self):
        assert wd14_tagger._format_tag("^_^") == "^_^"
        assert wd14_tagger._format_tag("o_o") == "o_o"

    def test_tag_with_no_underscore_is_unchanged(self):
        assert wd14_tagger._format_tag("solo") == "solo"


class TestTagImage:
    async def test_general_tags_above_threshold_are_included(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("rating", "9"), ("solo", "0"), ("background", "0"))
        session = FakeSession([0.9, 0.5, 0.1])  # rating ignored, solo passes 0.35, background doesn't
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(jpeg_bytes, general_threshold=0.35)
        assert result == "solo"

    async def test_character_tags_use_their_own_higher_threshold(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("some_character", "4"), ("generic_tag", "0"))
        session = FakeSession([0.6, 0.4])  # character: 0.6 < default 0.85 -> excluded; general: 0.4 >= 0.35 -> included
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(jpeg_bytes)
        assert result == "generic tag"

        session2 = FakeSession([0.9, 0.4])  # now character clears 0.85 too
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session2, "in", "out", 448, 448, rows))
        result2 = await wd14_tagger.tag_image(jpeg_bytes)
        assert result2 == "some character, generic tag"  # character tags sorted before general

    async def test_rating_category_is_never_included_in_output(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("explicit", "9"))
        session = FakeSession([0.99])
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(jpeg_bytes)
        assert result == ""

    async def test_general_tags_sorted_by_confidence_descending(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("low_conf", "0"), ("high_conf", "0"), ("mid_conf", "0"))
        session = FakeSession([0.4, 0.9, 0.6])
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(jpeg_bytes)
        assert result == "high conf, mid conf, low conf"

    async def test_no_tags_above_threshold_returns_empty_string(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("obscure", "0"))
        session = FakeSession([0.01])
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(jpeg_bytes)
        assert result == ""

    async def test_custom_thresholds_are_respected(self, monkeypatch, jpeg_bytes):
        rows = make_rows(("tag_a", "0"))
        session = FakeSession([0.5])
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        assert await wd14_tagger.tag_image(jpeg_bytes, general_threshold=0.6) == ""
        assert await wd14_tagger.tag_image(jpeg_bytes, general_threshold=0.4) == "tag a"

    async def test_works_with_png_and_transparency(self, monkeypatch, png_bytes):
        rows = make_rows(("solo", "0"))
        session = FakeSession([0.9])
        monkeypatch.setattr(wd14_tagger, "_load", lambda model: (session, "in", "out", 448, 448, rows))

        result = await wd14_tagger.tag_image(png_bytes)
        assert result == "solo"


class TestLoadCaching:
    def test_successful_load_is_cached_across_calls(self, monkeypatch, tmp_path):
        calls = {"download": 0, "session": 0}

        def fake_hf_hub_download(repo_id, filename):
            calls["download"] += 1
            return str(tmp_path / filename)

        class FakeOrtSession:
            def __init__(self, *a, **kw):
                calls["session"] += 1

            def get_inputs(self):
                class Info:
                    name = "input"
                    shape = ["batch", 448, 448, 3]

                return [Info()]

            def get_outputs(self):
                class Info:
                    name = "output"

                return [Info()]

        (tmp_path / "selected_tags.csv").write_text("tag_id,name,category,count\n0,solo,0,1\n", encoding="utf-8")

        monkeypatch.setattr(wd14_tagger, "hf_hub_download", fake_hf_hub_download)
        monkeypatch.setattr(wd14_tagger.ort, "InferenceSession", FakeOrtSession)

        wd14_tagger._load("wd-vit-tagger-v3")
        wd14_tagger._load("wd-vit-tagger-v3")

        assert calls["download"] == 2  # model.onnx + selected_tags.csv, once
        assert calls["session"] == 1  # not recreated on the second _load()

    def test_failed_load_is_cached_and_reraised_without_retrying(self, monkeypatch):
        attempts = {"n": 0}

        def failing_download(repo_id, filename):
            attempts["n"] += 1
            raise RuntimeError("network unreachable")

        monkeypatch.setattr(wd14_tagger, "hf_hub_download", failing_download)

        with pytest.raises(RuntimeError, match="network unreachable"):
            wd14_tagger._load("wd-vit-tagger-v3")
        with pytest.raises(RuntimeError, match="network unreachable"):
            wd14_tagger._load("wd-vit-tagger-v3")

        assert attempts["n"] == 1  # second call hit the cached exception, not the network again

    def test_different_models_are_cached_separately(self, monkeypatch, tmp_path):
        seen_repos = []

        def fake_hf_hub_download(repo_id, filename):
            seen_repos.append(repo_id)
            return str(tmp_path / filename)

        class FakeOrtSession:
            def __init__(self, *a, **kw):
                pass

            def get_inputs(self):
                class Info:
                    name = "input"
                    shape = ["batch", 448, 448, 3]

                return [Info()]

            def get_outputs(self):
                class Info:
                    name = "output"

                return [Info()]

        (tmp_path / "selected_tags.csv").write_text("tag_id,name,category,count\n0,solo,0,1\n", encoding="utf-8")

        monkeypatch.setattr(wd14_tagger, "hf_hub_download", fake_hf_hub_download)
        monkeypatch.setattr(wd14_tagger.ort, "InferenceSession", FakeOrtSession)

        wd14_tagger._load("wd-vit-tagger-v3")
        wd14_tagger._load("wd-swinv2-tagger-v3")

        assert "SmilingWolf/wd-vit-tagger-v3" in seen_repos
        assert "SmilingWolf/wd-swinv2-tagger-v3" in seen_repos
