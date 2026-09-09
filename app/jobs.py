from __future__ import annotations

import asyncio
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .download import download_image, sanitize_folder_name
from .llm_client import LLMClient, build_trigger_instruction
from .models import CaptionFolderRequest, JobCreateRequest
from .search import build_search_provider

logger = logging.getLogger(__name__)

# When a query is asked for N images, we search for more than N candidates up
# front so that failed/duplicate/undersized results don't silently shrink the
# final count -- we keep attempting candidates until N succeed or we run out.
OVERFETCH_MULTIPLIER = 4
OVERFETCH_MIN_EXTRA = 10
MAX_SEARCH_FETCH = 150

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class JobEvent:
    type: str
    data: dict


@dataclass
class JobState:
    id: str
    request: JobCreateRequest | CaptionFolderRequest
    kind: str = "download"  # "download" | "caption_folder"
    status: str = "pending"  # pending, running, done, error, cancelled
    events: list[JobEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    stats: dict = field(
        default_factory=lambda: {
            "downloaded": 0, "errors": 0, "duplicates": 0, "captioned": 0, "filtered": 0,
        }
    )
    cancel_requested: bool = False
    # Files this job has successfully downloaded, in order -- served back to the
    # UI for thumbnails via GET /api/jobs/{id}/image/{index}. Only ever appended
    # to from the job's own coroutine, so a plain list is safe here.
    downloaded_files: list[Path] = field(default_factory=list)


class JobManager:
    def __init__(self):
        self.jobs: dict[str, JobState] = {}

    def create_job(self, request: JobCreateRequest) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(id=job_id, kind="download", request=request)
        self.jobs[job_id] = state
        return state

    def create_caption_folder_job(self, request: CaptionFolderRequest) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(id=job_id, kind="caption_folder", request=request)
        self.jobs[job_id] = state
        return state

    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    async def emit(self, state: JobState, type_: str, data: dict) -> None:
        ev = JobEvent(type=type_, data=data)
        state.events.append(ev)
        for q in list(state.subscribers):
            await q.put(ev)

    async def run_job(self, state: JobState) -> None:
        req = state.request
        state.status = "running"
        await self.emit(state, "status", {"status": "running"})

        expander = LLMClient(req.llm_expansion.llm) if req.llm_expansion.enabled else None
        captioner = LLMClient(req.captioning.llm) if req.captioning.enabled else None
        try:
            search_provider = build_search_provider(req.search)
            allowed_formats = {f.lower().lstrip(".") for f in req.filters.formats}
            out_root = Path(req.output_folder)

            async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
                for query in req.queries:
                    if state.cancel_requested:
                        break
                    await self._process_query(
                        state, query, req, search_provider, expander, captioner,
                        allowed_formats, out_root, client,
                    )

            state.status = "cancelled" if state.cancel_requested else "done"
        except Exception as e:
            logger.exception("Job %s failed", state.id)
            state.status = "error"
            await self.emit(state, "warning", {"message": f"Job failed: {e}"})
        finally:
            if expander:
                await expander.aclose()
            if captioner:
                await captioner.aclose()
            await self.emit(state, "status", {"status": state.status, "stats": state.stats})

    async def _process_query(
        self, state, query, req, search_provider, expander, captioner,
        allowed_formats, out_root, client,
    ) -> None:
        search_terms = [query]
        if expander:
            try:
                variations = await expander.expand_queries(query, req.llm_expansion.variations_per_query)
                search_terms += variations
                await self.emit(state, "expanded", {"query": query, "variations": variations})
            except Exception as e:
                await self.emit(state, "warning", {"message": f"LLM expansion failed for '{query}': {e}"})

        group_dir = (out_root / sanitize_folder_name(query)) if req.query_subfolders else out_root
        name_prefix = "" if req.query_subfolders else f"{sanitize_folder_name(query)}_"
        seen_hashes: set[str] = set()
        index_box = [1]  # mutable counter shared across concurrent downloads

        for term in search_terms:
            if state.cancel_requested:
                return
            wanted = req.n_per_query
            fetch_n = min(max(wanted * OVERFETCH_MULTIPLIER, wanted + OVERFETCH_MIN_EXTRA), MAX_SEARCH_FETCH)
            try:
                results = await search_provider.search(term, fetch_n, safesearch=req.filters.safesearch)
            except Exception as e:
                await self.emit(state, "warning", {"message": f"Search failed for '{term}': {e}"})
                continue

            # Over-fetched candidates let us keep trying after failures/duplicates
            # so we still land close to `wanted` successful downloads. A worker
            # pool (rather than firing every candidate at once under a shared
            # semaphore) is what actually makes the early-stop work: with N
            # candidates launched together, the first `concurrency`-many are all
            # already in flight before any of them can increment the success
            # count, so a plain semaphore only bounds *parallelism*, not *total
            # downloads*. Capping the pool at min(concurrency, wanted) keeps any
            # overshoot past `wanted` small instead of up to `concurrency`.
            success_count = 0
            queue: asyncio.Queue = asyncio.Queue()
            for r in results:
                queue.put_nowait(r)

            async def worker():
                nonlocal success_count
                while True:
                    if state.cancel_requested or success_count >= wanted:
                        return
                    try:
                        result = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    idx = index_box[0]
                    index_box[0] += 1
                    res = await download_image(
                        client, result, group_dir, idx, name_prefix,
                        allowed_formats, req.filters.min_width, req.filters.min_height,
                        seen_hashes,
                    )
                    if res.ok:
                        success_count += 1
                        file_index = len(state.downloaded_files)
                        state.downloaded_files.append(res.path)
                        state.stats["downloaded"] += 1
                        await self.emit(
                            state, "downloaded",
                            {"query": term, "path": str(res.path), "url": res.url, "index": file_index},
                        )
                        if captioner:
                            try:
                                caption = await captioner.caption_image(res.content, res.content_type or "image/jpeg")
                                txt_path = res.path.with_suffix(".txt")
                                txt_path.write_text(caption, encoding="utf-8")
                                state.stats["captioned"] += 1
                                await self.emit(
                                    state, "captioned",
                                    {"path": str(txt_path), "caption": caption, "index": file_index},
                                )
                            except Exception as e:
                                await self.emit(state, "warning", {"message": f"Captioning failed: {e}"})
                    else:
                        if res.duplicate:
                            state.stats["duplicates"] += 1
                        elif res.filtered:
                            state.stats["filtered"] += 1
                        else:
                            state.stats["errors"] += 1
                        await self.emit(
                            state, "skip",
                            {
                                "query": term, "url": res.url, "error": res.error,
                                "duplicate": res.duplicate, "filtered": res.filtered,
                            },
                        )

            worker_count = max(1, min(req.concurrency, wanted, len(results)))
            await asyncio.gather(*(worker() for _ in range(worker_count)))

            if success_count < wanted and not state.cancel_requested:
                await self.emit(
                    state, "warning",
                    {"message": f"Only found {success_count}/{wanted} usable images for '{term}' "
                                 f"(ran out of search results after fetching {len(results)} candidates)."},
                )

    async def run_caption_folder_job(self, state: JobState) -> None:
        """Caption every image already sitting in a folder -- independent of any
        download job, e.g. an existing dataset the user wants captions for."""
        req: CaptionFolderRequest = state.request
        state.status = "running"
        await self.emit(state, "status", {"status": "running"})

        captioner = LLMClient(req.llm)
        try:
            root = Path(req.folder)
            if not root.is_dir():
                raise ValueError(f"Folder not found: {req.folder}")

            pattern = root.rglob("*") if req.recursive else root.glob("*")
            image_files = sorted(
                p for p in pattern if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not image_files:
                await self.emit(state, "warning", {"message": "No images found in this folder."})

            instruction = build_trigger_instruction(req.trigger)

            for path in image_files:
                if state.cancel_requested:
                    break

                txt_path = path.with_suffix(".txt")
                if txt_path.exists() and not req.overwrite:
                    await self.emit(
                        state, "skip",
                        {"query": path.name, "url": str(path),
                         "error": "caption already exists (overwrite is off)", "duplicate": False},
                    )
                    continue

                try:
                    content = path.read_bytes()
                    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                    caption = await captioner.caption_image(content, mime, extra_instruction=instruction)
                except Exception as e:
                    state.stats["errors"] += 1
                    await self.emit(state, "warning", {"message": f"Captioning failed for {path.name}: {e}"})
                    continue

                try:
                    txt_path.write_text(caption, encoding="utf-8")
                except Exception as e:
                    state.stats["errors"] += 1
                    await self.emit(state, "warning", {"message": f"Could not write {txt_path.name}: {e}"})
                    continue

                file_index = len(state.downloaded_files)
                state.downloaded_files.append(path)
                state.stats["downloaded"] += 1
                state.stats["captioned"] += 1
                await self.emit(
                    state, "downloaded",
                    {"query": path.name, "path": str(path), "url": "", "index": file_index},
                )
                await self.emit(
                    state, "captioned",
                    {"path": str(txt_path), "caption": caption, "index": file_index},
                )

            state.status = "cancelled" if state.cancel_requested else "done"
        except Exception as e:
            logger.exception("Caption-folder job %s failed", state.id)
            state.status = "error"
            await self.emit(state, "warning", {"message": f"Job failed: {e}"})
        finally:
            await captioner.aclose()
            await self.emit(state, "status", {"status": state.status, "stats": state.stats})


job_manager = JobManager()
