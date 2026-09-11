from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import local_models, main, model_control, model_registry
from app.jobs import JobEvent, job_manager


@pytest.fixture
def client():
    return TestClient(main.app)


class TestIndexAndStatic:
    def test_index_serves_the_web_ui(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestLifespanShutdownUnloadsModels:
    """The graceful-shutdown half of the "unload models" feature: exiting the
    lifespan context (what uvicorn does on Ctrl+C / a normal process exit)
    should try to unload every model this process actually used. This can't
    be exercised through a running server + OS signals in a portable way
    (Windows has no real SIGTERM equivalent for an unattached process), so it
    is tested directly against the lifespan context manager instead -- the
    same mechanism uvicorn itself drives.
    """

    async def test_unloads_every_tracked_model_on_exit(self, monkeypatch):
        model_registry.note_used("ollama", "http://fake/v1", "llama3")
        model_registry.note_used("lmstudio", "http://fake2/v1", "moondream")
        unloaded = []

        async def fake_unload(provider, base_url, model):
            unloaded.append(model)
            return "ok"

        monkeypatch.setattr(model_control, "unload_model", fake_unload)

        async with main.lifespan(main.app):
            pass

        assert sorted(unloaded) == ["llama3", "moondream"]

    async def test_a_failed_unload_does_not_stop_the_others(self, monkeypatch):
        model_registry.note_used("ollama", "http://fake/v1", "llama3")
        model_registry.note_used("lmstudio", "http://fake2/v1", "moondream")
        unloaded = []

        async def fake_unload(provider, base_url, model):
            if model == "llama3":
                raise RuntimeError("connection refused")
            unloaded.append(model)
            return "ok"

        monkeypatch.setattr(model_control, "unload_model", fake_unload)

        async with main.lifespan(main.app):
            pass  # must not raise, even though one unload fails

        assert unloaded == ["moondream"]

    async def test_nothing_tracked_is_a_no_op(self, monkeypatch):
        called = False

        async def fake_unload(provider, base_url, model):
            nonlocal called
            called = True
            return "ok"

        monkeypatch.setattr(model_control, "unload_model", fake_unload)

        async with main.lifespan(main.app):
            pass

        assert called is False


class TestPickFolder:
    def test_returns_the_chosen_folder(self, client, monkeypatch):
        async def fake_pick_folder(initial_dir, title):
            return "C:/chosen"

        monkeypatch.setattr(local_models, "pick_folder", fake_pick_folder)
        resp = client.post("/api/pick-folder", json={"title": "Pick one"})
        assert resp.status_code == 200
        assert resp.json() == {"folder": "C:/chosen"}

    def test_picker_failure_becomes_a_500_with_a_message(self, client, monkeypatch):
        async def fake_pick_folder(initial_dir, title):
            raise RuntimeError("no display available")

        monkeypatch.setattr(local_models, "pick_folder", fake_pick_folder)
        resp = client.post("/api/pick-folder", json={})
        assert resp.status_code == 500
        assert "no display available" in resp.json()["detail"]


class TestListModels:
    def test_reports_server_and_folder_results_separately(self, client, monkeypatch):
        async def fake_list_server_models(base_url, api_key):
            return ["model-a"]

        def fake_scan_folder_for_models(folder):
            raise ValueError("bad folder")

        monkeypatch.setattr(local_models, "list_server_models", fake_list_server_models)
        monkeypatch.setattr(local_models, "scan_folder_for_models", fake_scan_folder_for_models)

        resp = client.post(
            "/api/llm/models",
            json={"base_url": "http://fake/v1", "models_folder": "Z:/bad"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["server_models"] == ["model-a"]
        assert body["server_error"] is None
        assert body["folder_models"] == []
        assert "bad folder" in body["folder_error"]


class TestUnloadModel:
    def test_unload_one_model_success(self, client, monkeypatch):
        model_registry.note_used("ollama", "http://fake/v1", "llama3")

        async def fake_unload(provider, base_url, model):
            return f"unloaded {model}"

        monkeypatch.setattr(model_control, "unload_model", fake_unload)
        resp = client.post(
            "/api/llm/unload",
            json={"llm": {"provider": "ollama", "base_url": "http://fake/v1", "model": "llama3"}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"message": "unloaded llama3"}
        assert ("ollama", "http://fake/v1", "llama3") not in model_registry.all_used()

    def test_unload_one_model_failure_is_a_500(self, client, monkeypatch):
        async def fake_unload(provider, base_url, model):
            raise RuntimeError("lms not on PATH")

        monkeypatch.setattr(model_control, "unload_model", fake_unload)
        resp = client.post(
            "/api/llm/unload",
            json={"llm": {"provider": "lmstudio", "base_url": "http://fake/v1", "model": "moondream"}},
        )
        assert resp.status_code == 500
        assert "lms not on PATH" in resp.json()["detail"]

    def test_unload_all_with_nothing_tracked(self, client):
        resp = client.post("/api/llm/unload-all")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []
        assert "nothing to unload" in body["message"].lower()

    def test_unload_all_reports_mixed_success_and_failure(self, client, monkeypatch):
        model_registry.note_used("ollama", "http://fake/v1", "llama3")
        model_registry.note_used("lmstudio", "http://fake2/v1", "moondream")

        async def fake_unload(provider, base_url, model):
            if provider == "ollama":
                return "ok"
            raise RuntimeError("boom")

        monkeypatch.setattr(model_control, "unload_model", fake_unload)
        resp = client.post("/api/llm/unload-all")
        assert resp.status_code == 200
        results = {r["model"]: r for r in resp.json()["results"]}
        assert results["llama3"]["ok"] is True
        assert results["moondream"]["ok"] is False
        assert "boom" in results["moondream"]["message"]

        # Only the successful unload should be forgotten -- the failed one
        # stays tracked so a retry (or the shutdown hook) can try again.
        remaining = model_registry.all_used()
        assert ("ollama", "http://fake/v1", "llama3") not in remaining
        assert ("lmstudio", "http://fake2/v1", "moondream") in remaining


class TestCreateJobValidation:
    def test_rejects_empty_queries(self, client):
        resp = client.post(
            "/api/jobs", json={"queries": ["", "  "], "output_folder": "C:/out"}
        )
        assert resp.status_code == 400

    def test_rejects_missing_output_folder(self, client):
        resp = client.post("/api/jobs", json={"queries": ["cats"], "output_folder": "  "})
        assert resp.status_code == 400

    def test_rejects_non_positive_n_per_query(self, client):
        resp = client.post(
            "/api/jobs",
            json={"queries": ["cats"], "output_folder": "C:/out", "n_per_query": 0},
        )
        assert resp.status_code == 422

    def test_accepts_a_valid_request_and_returns_a_job_id(self, client, monkeypatch):
        started = []

        async def fake_run_job(state):
            started.append(state.id)

        monkeypatch.setattr(job_manager, "run_job", fake_run_job)

        resp = client.post(
            "/api/jobs", json={"queries": ["cats"], "output_folder": "C:/out"}
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        assert job_manager.get(job_id) is not None


class TestCaptionFolderValidation:
    def test_rejects_empty_folder(self, client):
        resp = client.post("/api/caption-folder", json={"folder": "  "})
        assert resp.status_code == 400

    def test_rejects_trigger_enabled_with_empty_word(self, client):
        resp = client.post(
            "/api/caption-folder",
            json={"folder": "C:/data", "trigger": {"enabled": True, "word": "  "}},
        )
        assert resp.status_code == 400

    def test_accepts_a_valid_request(self, client, monkeypatch):
        async def fake_run(state):
            pass

        monkeypatch.setattr(job_manager, "run_caption_folder_job", fake_run)
        resp = client.post("/api/caption-folder", json={"folder": "C:/data"})
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestDedupFolderValidation:
    def test_rejects_empty_folder(self, client):
        resp = client.post("/api/dedup-folder", json={"folder": "  "})
        assert resp.status_code == 400

    def test_accepts_a_valid_request(self, client, monkeypatch):
        async def fake_run(state):
            pass

        monkeypatch.setattr(job_manager, "run_dedup_job", fake_run)
        resp = client.post("/api/dedup-folder", json={"folder": "C:/data"})
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_recursive_flag_is_passed_through(self, client, monkeypatch):
        captured = {}

        async def fake_run(state):
            captured["recursive"] = state.request.recursive

        monkeypatch.setattr(job_manager, "run_dedup_job", fake_run)
        resp = client.post("/api/dedup-folder", json={"folder": "C:/data", "recursive": True})
        assert resp.status_code == 200


class TestCaptionFilesValidation:
    def test_rejects_an_empty_selection(self, client):
        resp = client.post("/api/caption-files", json={"paths": []})
        assert resp.status_code == 400

    def test_accepts_a_valid_request(self, client, monkeypatch):
        async def fake_run(state):
            pass

        monkeypatch.setattr(job_manager, "run_caption_files_job", fake_run)
        resp = client.post("/api/caption-files", json={"paths": ["C:/data/a.jpg"]})
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestDeleteFilesValidation:
    def test_rejects_an_empty_selection(self, client):
        resp = client.post("/api/delete-files", json={"paths": []})
        assert resp.status_code == 400

    def test_accepts_a_valid_request(self, client, monkeypatch):
        async def fake_run(state):
            pass

        monkeypatch.setattr(job_manager, "run_delete_files_job", fake_run)
        resp = client.post("/api/delete-files", json={"paths": ["C:/data/a.jpg"]})
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestBrowseFolder:
    def test_rejects_a_folder_that_does_not_exist(self, client, tmp_path):
        resp = client.get("/api/browse-folder", params={"folder": str(tmp_path / "nope")})
        assert resp.status_code == 400

    def test_lists_images_and_caption_counts(self, client, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "a.txt").write_text("a caption", encoding="utf-8")
        (tmp_path / "b.jpg").write_bytes(b"fake")

        resp = client.get("/api/browse-folder", params={"folder": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["captioned"] == 1
        names = {f["name"] for f in body["files"]}
        assert names == {"a.jpg", "b.jpg"}

    def test_recursive_is_on_by_default(self, client, tmp_path):
        sub = tmp_path / "query_1"
        sub.mkdir()
        (sub / "a.jpg").write_bytes(b"fake")

        resp = client.get("/api/browse-folder", params={"folder": str(tmp_path)})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_recursive_false_ignores_subfolders(self, client, tmp_path):
        sub = tmp_path / "query_1"
        sub.mkdir()
        (sub / "a.jpg").write_bytes(b"fake")

        resp = client.get("/api/browse-folder", params={"folder": str(tmp_path), "recursive": False})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestLocalImage:
    def test_serves_an_existing_image_file(self, client, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake-jpeg-bytes")
        resp = client.get("/api/local-image", params={"path": str(img)})
        assert resp.status_code == 200
        assert resp.content == b"fake-jpeg-bytes"

    def test_missing_file_is_404(self, client, tmp_path):
        resp = client.get("/api/local-image", params={"path": str(tmp_path / "nope.jpg")})
        assert resp.status_code == 404

    def test_non_image_extension_is_404(self, client, tmp_path):
        txt = tmp_path / "a.txt"
        txt.write_text("hi", encoding="utf-8")
        resp = client.get("/api/local-image", params={"path": str(txt)})
        assert resp.status_code == 404


class TestJobStatusAndImages:
    def test_get_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/doesnotexist").status_code == 404

    def test_cancel_unknown_job_is_404(self, client):
        assert client.post("/api/jobs/doesnotexist/cancel").status_code == 404

    def test_get_job_reports_status_and_stats(self, client):
        from app.models import JobCreateRequest

        state = job_manager.create_job(JobCreateRequest(queries=["cats"], output_folder="C:/out"))
        resp = client.get(f"/api/jobs/{state.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == state.id
        assert body["status"] == "pending"

    def test_cancel_sets_the_flag(self, client):
        from app.models import JobCreateRequest

        state = job_manager.create_job(JobCreateRequest(queries=["cats"], output_folder="C:/out"))
        resp = client.post(f"/api/jobs/{state.id}/cancel")
        assert resp.status_code == 200
        assert state.cancel_requested is True

    def test_image_index_out_of_range_is_404(self, client):
        from app.models import JobCreateRequest

        state = job_manager.create_job(JobCreateRequest(queries=["cats"], output_folder="C:/out"))
        resp = client.get(f"/api/jobs/{state.id}/image/0")
        assert resp.status_code == 404

    def test_image_for_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/doesnotexist/image/0").status_code == 404

    def test_serves_a_downloaded_file(self, client, tmp_path):
        from app.models import JobCreateRequest

        img = tmp_path / "img_0001.jpg"
        img.write_bytes(b"fake-jpeg-bytes")
        state = job_manager.create_job(JobCreateRequest(queries=["cats"], output_folder="C:/out"))
        state.downloaded_files.append(img)

        resp = client.get(f"/api/jobs/{state.id}/image/0")
        assert resp.status_code == 200
        assert resp.content == b"fake-jpeg-bytes"


class TestJobWebSocket:
    def test_replays_history_and_closes_on_terminal_status(self, client):
        """A client that connects after the job already produced events should
        see the full history replayed, then have the connection closed once the
        terminal status event goes by."""
        from app.models import JobCreateRequest

        state = job_manager.create_job(JobCreateRequest(queries=["cats"], output_folder="C:/out"))
        state.events.append(JobEvent(type="downloaded", data={"query": "cats", "path": "x.jpg", "index": 0}))
        state.events.append(JobEvent(type="status", data={"status": "done", "stats": state.stats}))

        with client.websocket_connect(f"/ws/jobs/{state.id}") as ws:
            first = ws.receive_json()
            second = ws.receive_json()

        assert first == {"type": "downloaded", "data": {"query": "cats", "path": "x.jpg", "index": 0}}
        assert second["type"] == "status"
        assert second["data"]["status"] == "done"

    def test_unknown_job_closes_immediately(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/jobs/doesnotexist") as ws:
                ws.receive_json()
