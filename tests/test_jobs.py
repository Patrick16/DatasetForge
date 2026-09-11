from __future__ import annotations

from pathlib import Path

import pytest

from app import captioning as captioning_module
from app import jobs as jobs_module
from app.download import DownloadResult
from app.jobs import JobManager
from app.models import (
    CaptionFilesRequest,
    CaptionFolderRequest,
    CaptioningConfig,
    DedupFolderRequest,
    DeleteFilesRequest,
    FilterConfig,
    JobCreateRequest,
    LLMConfig,
    LLMExpansionConfig,
)
from app.search.base import ImageResult


def make_request(**overrides) -> JobCreateRequest:
    kwargs = dict(
        queries=["cats"],
        n_per_query=3,
        output_folder="unused",
        query_subfolders=True,
        concurrency=1,  # deterministic worker scheduling in tests
        filters=FilterConfig(min_width=0, min_height=0),
    )
    kwargs.update(overrides)
    return JobCreateRequest(**kwargs)


class FakeSearchProvider:
    """Returns `n` distinct fake ImageResults per call and records every call."""

    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []

    async def search(self, query: str, n: int, safesearch: str = "moderate") -> list[ImageResult]:
        self.calls.append((query, n, safesearch))
        return [ImageResult(url=f"http://example.com/{query}/{i}.jpg") for i in range(n)]


class FakeLLMClient:
    """Stand-in for LLMClient used by jobs.py -- tracks aclose() and returns
    canned answers instead of making network calls."""

    instances: list["FakeLLMClient"] = []

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.expand_calls: list[tuple[str, int]] = []
        self.caption_calls: list[bytes] = []
        FakeLLMClient.instances.append(self)

    async def expand_queries(self, query, n):
        self.expand_calls.append((query, n))
        return [f"{query} variation {i}" for i in range(n)]

    async def caption_image(self, image_bytes, mime="image/jpeg", extra_instruction=None):
        self.caption_calls.append(image_bytes)
        return "a fake caption"

    async def aclose(self):
        self.closed = True


class FakeCaptioner:
    """Stand-in for a Captioner (app/captioning.py) passed directly to
    _process_query -- unlike FakeLLMClient, this matches the `.caption()`
    protocol jobs.py actually calls, without going through build_captioner()."""

    def __init__(self, caption_text: str = "a fake caption", raise_error: Exception | None = None):
        self.caption_text = caption_text
        self.raise_error = raise_error
        self.calls: list[bytes] = []

    async def caption(self, image_bytes, mime="image/jpeg"):
        self.calls.append(image_bytes)
        if self.raise_error:
            raise self.raise_error
        return self.caption_text

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def reset_fake_llm_instances():
    FakeLLMClient.instances.clear()
    yield
    FakeLLMClient.instances.clear()


@pytest.fixture
def manager() -> JobManager:
    return JobManager()


class TestProcessQueryDownloadOnly:
    async def test_stops_once_wanted_count_is_reached(self, manager, monkeypatch, tmp_path):
        """With concurrency=1, downloads happen strictly one at a time, so the
        worker must stop exactly at `wanted` successes instead of draining the
        whole over-fetched candidate list."""
        call_count = 0

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            nonlocal call_count
            call_count += 1
            return DownloadResult(ok=True, url=result.url, path=dest_dir / f"img_{index}.jpg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=3, concurrency=1)
        state = manager.create_job(req)
        provider = FakeSearchProvider()

        await manager._process_query(
            state, "cats", req, provider, None, None, {"jpg"}, tmp_path, None
        )

        assert call_count == 3
        assert state.stats["downloaded"] == 3
        assert state.stats["errors"] == 0

    async def test_emits_warning_when_it_runs_out_of_candidates(self, manager, monkeypatch, tmp_path):
        async def always_fails(client, result, dest_dir, index, name_prefix,
                                allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=False, url=result.url, error="boom")

        monkeypatch.setattr(jobs_module, "download_image", always_fails)

        req = make_request(n_per_query=3, concurrency=1)
        state = manager.create_job(req)
        provider = FakeSearchProvider()

        await manager._process_query(
            state, "cats", req, provider, None, None, {"jpg"}, tmp_path, None
        )

        assert state.stats["downloaded"] == 0
        assert state.stats["errors"] > 0
        warnings = [e for e in state.events if e.type == "warning"]
        assert any("Only found 0/3" in e.data["message"] for e in warnings)

    async def test_duplicates_errors_and_filtered_are_tallied_separately(self, manager, monkeypatch, tmp_path):
        responses = [
            DownloadResult(ok=True, url="a", path=tmp_path / "a.jpg"),
            DownloadResult(ok=False, url="b", duplicate=True, error="dup"),
            DownloadResult(ok=False, url="c", duplicate=False, error="network timeout"),
            DownloadResult(ok=False, url="d", filtered=True, error="format 'gif' not allowed"),
        ]

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return responses.pop(0)

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=10, concurrency=1)
        state = manager.create_job(req)
        provider = FakeSearchProvider()

        # Only feed exactly 4 candidates so the worker stops after them
        # instead of running out and emitting an extra "ran out" warning.
        async def limited_search(query, n, safesearch="moderate"):
            return [ImageResult(url=f"http://example.com/{i}.jpg") for i in range(4)]

        provider.search = limited_search

        await manager._process_query(
            state, "cats", req, provider, None, None, {"jpg"}, tmp_path, None
        )

        assert state.stats["downloaded"] == 1
        assert state.stats["duplicates"] == 1
        assert state.stats["errors"] == 1
        assert state.stats["filtered"] == 1

    async def test_query_subfolders_toggle_controls_output_layout(self, manager, monkeypatch, tmp_path):
        seen_dirs = []

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            seen_dirs.append((dest_dir, name_prefix))
            return DownloadResult(ok=True, url=result.url, path=dest_dir / "x.jpg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=1, query_subfolders=False)
        state = manager.create_job(req)
        provider = FakeSearchProvider()
        await manager._process_query(
            state, "red cats", req, provider, None, None, {"jpg"}, tmp_path, None
        )

        dest_dir, name_prefix = seen_dirs[0]
        assert dest_dir == tmp_path
        assert name_prefix == "red cats_"


class TestProcessQueryWithExpansionAndCaptioning:
    async def test_expander_variations_each_get_searched(self, manager, monkeypatch, tmp_path):
        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=True, url=result.url, path=dest_dir / "x.jpg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=1, llm_expansion=LLMExpansionConfig(enabled=True, variations_per_query=2))
        state = manager.create_job(req)
        provider = FakeSearchProvider()
        expander = FakeLLMClient(req.llm_expansion.llm)

        await manager._process_query(
            state, "cats", req, provider, expander, None, {"jpg"}, tmp_path, None
        )

        # original query + 2 variations = 3 search calls
        assert len(provider.calls) == 3
        expanded_events = [e for e in state.events if e.type == "expanded"]
        assert expanded_events[0].data["variations"] == ["cats variation 0", "cats variation 1"]

    async def test_expansion_failure_emits_warning_and_still_searches_original(
        self, manager, monkeypatch, tmp_path
    ):
        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=True, url=result.url, path=dest_dir / "x.jpg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        class BrokenExpander(FakeLLMClient):
            async def expand_queries(self, query, n):
                raise RuntimeError("model unreachable")

        req = make_request(n_per_query=1, llm_expansion=LLMExpansionConfig(enabled=True, variations_per_query=2))
        state = manager.create_job(req)
        provider = FakeSearchProvider()
        expander = BrokenExpander(req.llm_expansion.llm)

        await manager._process_query(
            state, "cats", req, provider, expander, None, {"jpg"}, tmp_path, None
        )

        assert len(provider.calls) == 1  # only the original query
        warnings = [e for e in state.events if e.type == "warning"]
        assert any("LLM expansion failed" in e.data["message"] for e in warnings)

    async def test_successful_download_triggers_captioning(self, manager, monkeypatch, tmp_path):
        group_dir = tmp_path / "cats"
        group_dir.mkdir()
        img_path = group_dir / "img_0001.jpg"
        img_path.write_bytes(b"fake-bytes")

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=True, url=result.url, path=img_path, content=b"fake-bytes",
                                   content_type="image/jpeg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=1, captioning=CaptioningConfig(enabled=True))
        state = manager.create_job(req)
        provider = FakeSearchProvider()
        captioner = FakeCaptioner()

        await manager._process_query(
            state, "cats", req, provider, None, captioner, {"jpg"}, tmp_path, None
        )

        assert state.stats["captioned"] == 1
        assert img_path.with_suffix(".txt").read_text(encoding="utf-8") == "a fake caption"
        captioned_events = [e for e in state.events if e.type == "captioned"]
        assert captioned_events[0].data["caption"] == "a fake caption"

    async def test_captioning_failure_emits_warning_but_download_still_counts(
        self, manager, monkeypatch, tmp_path
    ):
        group_dir = tmp_path / "cats"
        group_dir.mkdir()
        img_path = group_dir / "img_0001.jpg"
        img_path.write_bytes(b"fake-bytes")

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=True, url=result.url, path=img_path, content=b"x",
                                   content_type="image/jpeg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)

        req = make_request(n_per_query=1, captioning=CaptioningConfig(enabled=True))
        state = manager.create_job(req)
        provider = FakeSearchProvider()
        captioner = FakeCaptioner(raise_error=RuntimeError("vision model down"))

        await manager._process_query(
            state, "cats", req, provider, None, captioner, {"jpg"}, tmp_path, None
        )

        assert state.stats["downloaded"] == 1
        assert state.stats["captioned"] == 0
        warnings = [e for e in state.events if e.type == "warning"]
        assert any("Captioning failed" in e.data["message"] for e in warnings)


class TestRunJob:
    async def test_full_run_reaches_done_and_closes_llm_clients(self, manager, monkeypatch, tmp_path):
        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / f"img_{index}.jpg"
            path.write_bytes(b"x")
            return DownloadResult(ok=True, url=result.url, path=path, content=b"x", content_type="image/jpeg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)
        monkeypatch.setattr(jobs_module, "build_search_provider", lambda config: FakeSearchProvider())
        monkeypatch.setattr(jobs_module, "LLMClient", FakeLLMClient)  # the query-expander
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)  # inside build_captioner()

        req = make_request(
            queries=["cats", "dogs"],
            n_per_query=1,
            output_folder=str(tmp_path),
            llm_expansion=LLMExpansionConfig(enabled=True, variations_per_query=0),
            captioning=CaptioningConfig(enabled=True),
        )
        state = manager.create_job(req)

        await manager.run_job(state)

        assert state.status == "done"
        assert state.stats["downloaded"] == 2
        assert state.stats["captioned"] == 2
        assert all(inst.closed for inst in FakeLLMClient.instances)
        final_status = [e for e in state.events if e.type == "status"][-1]
        assert final_status.data["status"] == "done"
        assert final_status.data["stats"] == state.stats

    async def test_cancellation_stops_before_the_next_query(self, manager, monkeypatch, tmp_path):
        provider = FakeSearchProvider()

        async def fake_download_image(client, result, dest_dir, index, name_prefix,
                                       allowed_formats, min_width, min_height, seen_hashes):
            return DownloadResult(ok=True, url=result.url, path=tmp_path / "x.jpg")

        monkeypatch.setattr(jobs_module, "download_image", fake_download_image)
        monkeypatch.setattr(jobs_module, "build_search_provider", lambda config: provider)

        req = make_request(queries=["cats", "dogs"], n_per_query=1, output_folder=str(tmp_path))
        state = manager.create_job(req)
        state.cancel_requested = True

        await manager.run_job(state)

        assert state.status == "cancelled"
        assert provider.calls == []  # cancelled before the first query ran

    async def test_llm_clients_are_closed_even_when_the_job_errors(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(jobs_module, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        def broken_provider(config):
            raise RuntimeError("provider init failed")

        monkeypatch.setattr(jobs_module, "build_search_provider", broken_provider)

        req = make_request(
            output_folder=str(tmp_path),
            llm_expansion=LLMExpansionConfig(enabled=True),
            captioning=CaptioningConfig(enabled=True),
        )
        state = manager.create_job(req)

        await manager.run_job(state)

        assert state.status == "error"
        assert len(FakeLLMClient.instances) == 2
        assert all(inst.closed for inst in FakeLLMClient.instances)


class TestRunCaptionFolderJob:
    def _make_image(self, path: Path) -> None:
        path.write_bytes(b"fake-image-bytes")

    async def test_captions_every_image_in_a_flat_folder(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        self._make_image(tmp_path / "b.png")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFolderRequest(folder=str(tmp_path), llm=LLMConfig())
        state = manager.create_caption_folder_job(req)

        await manager.run_caption_folder_job(state)

        assert state.status == "done"
        assert state.stats["captioned"] == 2
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a fake caption"
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "a fake caption"
        assert FakeLLMClient.instances[0].closed is True

    async def test_skips_images_that_already_have_a_caption(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        (tmp_path / "a.txt").write_text("existing caption", encoding="utf-8")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFolderRequest(folder=str(tmp_path), llm=LLMConfig(), overwrite=False)
        state = manager.create_caption_folder_job(req)

        await manager.run_caption_folder_job(state)

        assert state.stats["captioned"] == 0
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "existing caption"
        skips = [e for e in state.events if e.type == "skip"]
        assert len(skips) == 1

    async def test_overwrite_forces_recaptioning(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        (tmp_path / "a.txt").write_text("stale caption", encoding="utf-8")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFolderRequest(folder=str(tmp_path), llm=LLMConfig(), overwrite=True)
        state = manager.create_caption_folder_job(req)

        await manager.run_caption_folder_job(state)

        assert state.stats["captioned"] == 1
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a fake caption"

    async def test_recursive_flag_includes_nested_images(self, manager, monkeypatch, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        self._make_image(nested / "a.jpg")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        non_recursive_req = CaptionFolderRequest(folder=str(tmp_path), llm=LLMConfig(), recursive=False)
        state = manager.create_caption_folder_job(non_recursive_req)
        await manager.run_caption_folder_job(state)
        assert state.stats["captioned"] == 0

        recursive_req = CaptionFolderRequest(folder=str(tmp_path), llm=LLMConfig(), recursive=True)
        state2 = manager.create_caption_folder_job(recursive_req)
        await manager.run_caption_folder_job(state2)
        assert state2.stats["captioned"] == 1

    async def test_missing_folder_reports_error_status(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)
        req = CaptionFolderRequest(folder=str(tmp_path / "nope"), llm=LLMConfig())
        state = manager.create_caption_folder_job(req)

        await manager.run_caption_folder_job(state)

        assert state.status == "error"
        assert FakeLLMClient.instances[0].closed is True

    async def test_trigger_instruction_is_forwarded_to_the_captioner(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")

        received = {}

        class RecordingCaptioner(FakeLLMClient):
            async def caption_image(self, image_bytes, mime="image/jpeg", extra_instruction=None):
                received["instruction"] = extra_instruction
                return await super().caption_image(image_bytes, mime, extra_instruction)

        monkeypatch.setattr(captioning_module, "LLMClient", RecordingCaptioner)

        from app.models import TriggerWordConfig

        req = CaptionFolderRequest(
            folder=str(tmp_path), llm=LLMConfig(),
            trigger=TriggerWordConfig(enabled=True, word="zog", role="subject"),
        )
        state = manager.create_caption_folder_job(req)
        await manager.run_caption_folder_job(state)

        assert received["instruction"] is not None
        assert "zog" in received["instruction"]


class TestRunDedupJob:
    def _fake_send2trash(self, monkeypatch, removed: list[str] | None = None, fail_for: set[str] | None = None):
        """Stand-in for send2trash that actually deletes the file (so tests
        can assert on real filesystem state afterwards) instead of hitting
        the real OS Recycle Bin -- optionally raising for specific paths to
        exercise the failure path."""
        removed = removed if removed is not None else []
        fail_for = fail_for or set()

        def fake(path):
            if path in fail_for:
                raise OSError(f"simulated failure removing {path}")
            removed.append(path)
            Path(path).unlink()

        monkeypatch.setattr(jobs_module, "send2trash", fake)
        return removed

    async def test_removes_all_but_one_of_a_duplicate_pair(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "b.jpg").write_bytes(b"same")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert state.status == "done"
        remaining = sorted(p.name for p in tmp_path.glob("*.jpg"))
        assert remaining == ["a.jpg"]  # alphabetically-first kept, "b.jpg" removed

    async def test_stats_downloaded_is_groups_and_duplicates_is_removed_count(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        for name in ("a1.jpg", "a2.jpg", "a3.jpg"):
            (tmp_path / name).write_bytes(b"group a")
        (tmp_path / "b1.jpg").write_bytes(b"group b")
        (tmp_path / "b2.jpg").write_bytes(b"group b")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert state.stats["downloaded"] == 2  # 2 groups
        assert state.stats["duplicates"] == 3  # 2 removed from group a + 1 from group b

    async def test_orphaned_caption_file_is_removed_alongside_its_image(self, manager, monkeypatch, tmp_path):
        removed = self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "b.jpg").write_bytes(b"same")
        (tmp_path / "b.txt").write_text("stale caption for a duplicate", encoding="utf-8")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert str(tmp_path / "b.txt") in removed
        assert not (tmp_path / "b.txt").exists()

    async def test_keepers_caption_file_is_left_alone(self, manager, monkeypatch, tmp_path):
        removed = self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "a.txt").write_text("this caption belongs to the keeper", encoding="utf-8")
        (tmp_path / "b.jpg").write_bytes(b"same")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert str(tmp_path / "a.txt") not in removed
        assert (tmp_path / "a.txt").exists()

    async def test_no_duplicates_finishes_done_with_a_warning_and_zero_stats(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"one")
        (tmp_path / "b.jpg").write_bytes(b"two")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert state.status == "done"
        assert state.stats["downloaded"] == 0
        assert state.stats["duplicates"] == 0
        warnings = [e for e in state.events if e.type == "warning"]
        assert any("No duplicate images found" in e.data["message"] for e in warnings)

    async def test_missing_folder_reports_error_status(self, manager, tmp_path):
        req = DedupFolderRequest(folder=str(tmp_path / "nope"))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)
        assert state.status == "error"

    async def test_cancellation_stops_before_removing_more_groups(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a1.jpg").write_bytes(b"group a")
        (tmp_path / "a2.jpg").write_bytes(b"group a")
        (tmp_path / "b1.jpg").write_bytes(b"group b")
        (tmp_path / "b2.jpg").write_bytes(b"group b")

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        state.cancel_requested = True
        await manager.run_dedup_job(state)

        assert state.status == "cancelled"
        assert state.stats["duplicates"] == 0  # nothing removed once already cancelled

    async def test_removal_failure_is_counted_as_an_error_and_does_not_stop_the_job(
        self, manager, monkeypatch, tmp_path
    ):
        (tmp_path / "a.jpg").write_bytes(b"group a")
        (tmp_path / "a2.jpg").write_bytes(b"group a")
        (tmp_path / "b1.jpg").write_bytes(b"group b")
        (tmp_path / "b2.jpg").write_bytes(b"group b")
        self._fake_send2trash(monkeypatch, fail_for={str(tmp_path / "a2.jpg")})

        req = DedupFolderRequest(folder=str(tmp_path))
        state = manager.create_dedup_job(req)
        await manager.run_dedup_job(state)

        assert state.status == "done"
        assert state.stats["errors"] == 1
        assert state.stats["duplicates"] == 1  # the other group's duplicate still got removed
        assert (tmp_path / "a2.jpg").exists()  # the failed removal is still there
        assert not (tmp_path / "b2.jpg").exists()

    async def test_recursive_flag_controls_whether_subfolders_are_scanned(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"same")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.jpg").write_bytes(b"same")

        non_recursive = DedupFolderRequest(folder=str(tmp_path), recursive=False)
        state = manager.create_dedup_job(non_recursive)
        await manager.run_dedup_job(state)
        assert state.stats["duplicates"] == 0

        recursive = DedupFolderRequest(folder=str(tmp_path), recursive=True)
        state2 = manager.create_dedup_job(recursive)
        await manager.run_dedup_job(state2)
        assert state2.stats["duplicates"] == 1


class TestRunCaptionFilesJob:
    def _make_image(self, path: Path) -> None:
        path.write_bytes(b"fake-image-bytes")

    async def test_captions_exactly_the_given_files(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        self._make_image(tmp_path / "b.jpg")
        self._make_image(tmp_path / "c.jpg")  # not selected -- must stay untouched
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFilesRequest(paths=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")], llm=LLMConfig())
        state = manager.create_caption_files_job(req)
        await manager.run_caption_files_job(state)

        assert state.status == "done"
        assert state.stats["captioned"] == 2
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a fake caption"
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "a fake caption"
        assert not (tmp_path / "c.txt").exists()

    async def test_overwrites_an_existing_caption_unconditionally(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        (tmp_path / "a.txt").write_text("stale caption", encoding="utf-8")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFilesRequest(paths=[str(tmp_path / "a.jpg")], llm=LLMConfig())
        state = manager.create_caption_files_job(req)
        await manager.run_caption_files_job(state)

        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a fake caption"

    async def test_missing_file_is_counted_as_an_error_and_does_not_stop_the_job(self, manager, monkeypatch, tmp_path):
        self._make_image(tmp_path / "a.jpg")
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)

        req = CaptionFilesRequest(
            paths=[str(tmp_path / "nope.jpg"), str(tmp_path / "a.jpg")], llm=LLMConfig(),
        )
        state = manager.create_caption_files_job(req)
        await manager.run_caption_files_job(state)

        assert state.status == "done"
        assert state.stats["errors"] == 1
        assert state.stats["captioned"] == 1

    async def test_empty_selection_finishes_done_with_a_warning(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)
        req = CaptionFilesRequest(paths=[], llm=LLMConfig())
        state = manager.create_caption_files_job(req)
        await manager.run_caption_files_job(state)

        assert state.status == "done"
        assert any(e.type == "warning" for e in state.events)

    async def test_captioner_is_closed_even_when_every_file_is_missing(self, manager, monkeypatch, tmp_path):
        monkeypatch.setattr(captioning_module, "LLMClient", FakeLLMClient)
        req = CaptionFilesRequest(paths=[str(tmp_path / "nope.jpg")], llm=LLMConfig())
        state = manager.create_caption_files_job(req)
        await manager.run_caption_files_job(state)

        assert state.stats["errors"] == 1
        assert FakeLLMClient.instances[-1].closed is True


class TestRunDeleteFilesJob:
    def _fake_send2trash(self, monkeypatch, fail_for: set[str] | None = None):
        fail_for = fail_for or set()

        def fake(path):
            if path in fail_for:
                raise OSError(f"simulated failure removing {path}")
            Path(path).unlink()

        monkeypatch.setattr(jobs_module, "send2trash", fake)

    async def test_removes_exactly_the_given_files(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")
        (tmp_path / "c.jpg").write_bytes(b"c")  # not selected -- must survive

        req = DeleteFilesRequest(paths=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")])
        state = manager.create_delete_files_job(req)
        await manager.run_delete_files_job(state)

        assert state.status == "done"
        assert state.stats["duplicates"] == 2  # reused as the "removed" counter
        remaining = sorted(p.name for p in tmp_path.glob("*.jpg"))
        assert remaining == ["c.jpg"]

    async def test_removes_the_orphaned_caption_alongside_its_image(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "a.txt").write_text("a caption", encoding="utf-8")

        req = DeleteFilesRequest(paths=[str(tmp_path / "a.jpg")])
        state = manager.create_delete_files_job(req)
        await manager.run_delete_files_job(state)

        assert not (tmp_path / "a.jpg").exists()
        assert not (tmp_path / "a.txt").exists()

    async def test_removal_failure_is_counted_as_an_error_and_does_not_stop_the_job(self, manager, monkeypatch, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")
        self._fake_send2trash(monkeypatch, fail_for={str(tmp_path / "a.jpg")})

        req = DeleteFilesRequest(paths=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")])
        state = manager.create_delete_files_job(req)
        await manager.run_delete_files_job(state)

        assert state.status == "done"
        assert state.stats["errors"] == 1
        assert state.stats["duplicates"] == 1
        assert (tmp_path / "a.jpg").exists()  # the failed removal is still there
        assert not (tmp_path / "b.jpg").exists()

    async def test_empty_selection_finishes_done_with_a_warning(self, manager, tmp_path):
        req = DeleteFilesRequest(paths=[])
        state = manager.create_delete_files_job(req)
        await manager.run_delete_files_job(state)

        assert state.status == "done"
        assert any(e.type == "warning" for e in state.events)

    async def test_cancellation_stops_before_removing_more_files(self, manager, monkeypatch, tmp_path):
        self._fake_send2trash(monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"a")
        (tmp_path / "b.jpg").write_bytes(b"b")

        req = DeleteFilesRequest(paths=[str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")])
        state = manager.create_delete_files_job(req)
        state.cancel_requested = True
        await manager.run_delete_files_job(state)

        assert state.status == "cancelled"
        assert (tmp_path / "a.jpg").exists()
        assert (tmp_path / "b.jpg").exists()
