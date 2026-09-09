from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import local_models, model_control, model_registry
from .jobs import job_manager
from .llm_client import LLMClient
from .models import CaptionFolderRequest, JobCreateRequest, LLMConfig

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Best-effort: free whatever local models this process actually used, so
    # closing the app doesn't leave them sitting in RAM/VRAM. Only fires on a
    # graceful shutdown (Ctrl+C, or a normal process exit) -- a force-kill
    # (task manager "End task", `kill -9`) never runs this, since the OS
    # doesn't give the process a chance to run any cleanup code at all.
    for provider, base_url, model in model_registry.all_used():
        try:
            message = await model_control.unload_model(provider, base_url, model)
            logger.info("Shutdown cleanup: %s", message)
        except Exception as e:
            logger.warning("Shutdown cleanup: failed to unload %r (%s): %s", model, provider, e)


app = FastAPI(title="DatasetForge", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


class PickFolderRequest(BaseModel):
    initial_dir: Optional[str] = None
    title: Optional[str] = "Select folder"


@app.post("/api/pick-folder")
async def pick_folder(req: PickFolderRequest):
    """Open a native folder-picker dialog on the server machine and return the chosen path.

    Only works when the browser and this server run on the same machine (the normal
    setup for this tool). Requires tkinter (bundled with standard Python on Windows).
    """
    try:
        folder = await local_models.pick_folder(req.initial_dir, req.title or "Select folder")
    except Exception as e:
        raise HTTPException(500, f"Folder picker unavailable on the server machine: {e}")
    return {"folder": folder}


class ListModelsRequest(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    models_folder: Optional[str] = None


@app.post("/api/llm/models")
async def list_models(req: ListModelsRequest):
    """List models available from a running OpenAI-compatible server, plus a best-effort
    scan of a local models folder (fallback for when the server isn't reachable yet)."""
    server_models: list[str] = []
    server_error: Optional[str] = None
    try:
        server_models = await local_models.list_server_models(req.base_url, req.api_key)
    except Exception as e:
        server_error = str(e)

    folder_models: list[str] = []
    folder_error: Optional[str] = None
    if req.models_folder:
        try:
            folder_models = local_models.scan_folder_for_models(req.models_folder)
        except Exception as e:
            folder_error = str(e)

    return {
        "server_models": server_models,
        "folder_models": folder_models,
        "server_error": server_error,
        "folder_error": folder_error,
    }


class ExpandQueriesRequest(BaseModel):
    queries: list[str]
    variations_per_query: int = 3
    llm: LLMConfig


@app.post("/api/llm/expand")
async def expand_queries(req: ExpandQueriesRequest):
    """Generate query variations up front (used by the UI to preview/update the queries
    field *before* a download job starts, rather than expanding silently mid-job)."""
    results = []
    async with LLMClient(req.llm) as client:
        for q in req.queries:
            q = q.strip()
            if not q:
                continue
            try:
                variations = await client.expand_queries(q, req.variations_per_query)
                results.append({"query": q, "variations": variations, "error": None})
            except Exception as e:
                results.append({"query": q, "variations": [], "error": str(e)})
    return {"results": results}


class UnloadModelRequest(BaseModel):
    llm: LLMConfig


@app.post("/api/llm/unload")
async def unload_model(req: UnloadModelRequest):
    """Unload one specific model from local server memory (Ollama/LM Studio)."""
    try:
        message = await model_control.unload_model(req.llm.provider, req.llm.base_url, req.llm.model)
    except Exception as e:
        raise HTTPException(500, str(e))
    model_registry.forget(req.llm.provider, req.llm.base_url, req.llm.model)
    return {"message": message}


@app.post("/api/llm/unload-all")
async def unload_all_models():
    """Unload every local model this app has actually used this session --
    scoped to what we ourselves loaded, never anything the user loaded some
    other way that just happens to be sitting on the same server."""
    used = model_registry.all_used()
    if not used:
        return {"results": [], "message": "Nothing to unload -- no local models have been used yet."}

    results = []
    for provider, base_url, model in used:
        try:
            message = await model_control.unload_model(provider, base_url, model)
            model_registry.forget(provider, base_url, model)
            results.append({"model": model, "provider": provider, "ok": True, "message": message})
        except Exception as e:
            results.append({"model": model, "provider": provider, "ok": False, "message": str(e)})
    return {"results": results}


@app.post("/api/jobs")
async def create_job(req: JobCreateRequest):
    if not req.queries or not any(q.strip() for q in req.queries):
        raise HTTPException(400, "At least one query is required")
    if not req.output_folder.strip():
        raise HTTPException(400, "output_folder is required")
    state = job_manager.create_job(req)
    asyncio.create_task(job_manager.run_job(state))
    return {"job_id": state.id}


@app.post("/api/caption-folder")
async def caption_folder(req: CaptionFolderRequest):
    """Caption every image already sitting in a folder -- independent of any download job."""
    if not req.folder.strip():
        raise HTTPException(400, "folder is required")
    if req.trigger.enabled and not req.trigger.word.strip():
        raise HTTPException(400, "trigger word is enabled but empty")
    state = job_manager.create_caption_folder_job(req)
    asyncio.create_task(job_manager.run_caption_folder_job(state))
    return {"job_id": state.id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    state = job_manager.get(job_id)
    if not state:
        raise HTTPException(404, "job not found")
    return {"id": state.id, "status": state.status, "stats": state.stats}


@app.get("/api/jobs/{job_id}/image/{index}")
async def get_job_image(job_id: str, index: int):
    """Serve one of this job's own downloaded files (for UI thumbnails).

    Scoped to files the job itself wrote -- not an arbitrary-path file server.
    """
    state = job_manager.get(job_id)
    if not state:
        raise HTTPException(404, "job not found")
    if index < 0 or index >= len(state.downloaded_files):
        raise HTTPException(404, "image not found")
    path = state.downloaded_files[index]
    if not path.exists():
        raise HTTPException(404, "file no longer exists on disk")
    return FileResponse(str(path))


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    state = job_manager.get(job_id)
    if not state:
        raise HTTPException(404, "job not found")
    state.cancel_requested = True
    return {"ok": True}


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    state = job_manager.get(job_id)
    if not state:
        await websocket.close(code=4404)
        return

    queue: asyncio.Queue = asyncio.Queue()
    for ev in state.events:  # replay history so late-connecting clients catch up
        queue.put_nowait(ev)
    state.subscribers.append(queue)

    try:
        while True:
            ev = await queue.get()
            await websocket.send_json({"type": ev.type, "data": ev.data})
            if ev.type == "status" and ev.data.get("status") in ("done", "error", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in state.subscribers:
            state.subscribers.remove(queue)
