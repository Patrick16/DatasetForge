from __future__ import annotations

import asyncio
import logging
import mimetypes
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from send2trash import send2trash

from .captioning import build_captioner
from .dedup import find_duplicate_groups
from .download import IMAGE_EXTENSIONS, download_image, sanitize_folder_name
from .llm_client import LLMClient
from .models import (
    CaptionFilesRequest,
    CaptionFolderRequest,
    DedupFolderRequest,
    DeleteFilesRequest,
    JobCreateRequest,
)
from .search import build_search_provider

logger = logging.getLogger(__name__)

# When a query is asked for N images, we search for more than N candidates up
# front so that failed/duplicate/undersized results don't silently shrink the
# final count -- we keep attempting candidates until N succeed or we run out.
OVERFETCH_MULTIPLIER = 4
OVERFETCH_MIN_EXTRA = 10
MAX_SEARCH_FETCH = 150


@dataclass
class JobEvent:
    type: str
    data: dict


@dataclass
class JobState:
    id: str
    request: JobCreateRequest | CaptionFolderRequest | DedupFolderRequest | CaptionFilesRequest | DeleteFilesRequest
    kind: str = "download"  # "download" | "caption_folder" | "dedup" | "caption_files" | "delete_files"
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

    def create_dedup_job(self, request: DedupFolderRequest) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(id=job_id, kind="dedup", request=request)
        self.jobs[job_id] = state
        return state

    def create_caption_files_job(self, request: CaptionFilesRequest) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(id=job_id, kind="caption_files", request=request)
        self.jobs[job_id] = state
        return state

    def create_delete_files_job(self, request: DeleteFilesRequest) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        state = JobState(id=job_id, kind="delete_files", request=request)
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
        captioner = (
            build_captioner(req.captioning.method, req.captioning.llm, req.captioning.wd14)
            if req.captioning.enabled
            else None
        )
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
                                caption = await captioner.caption(res.content, res.content_type or "image/jpeg")
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

        captioner = build_captioner(req.method, req.llm, req.wd14, trigger=req.trigger)
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
                    caption = await captioner.caption(content, mime)
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

    async def run_dedup_job(self, state: JobState) -> None:
        """Find images with byte-identical content in a folder and move every
        copy but one to the Recycle Bin -- independent of any download job."""
        req: DedupFolderRequest = state.request
        state.status = "running"
        await self.emit(state, "status", {"status": "running"})

        try:
            root = Path(req.folder)
            if not root.is_dir():
                raise ValueError(f"Folder not found: {req.folder}")

            groups = await asyncio.to_thread(find_duplicate_groups, root, req.recursive)
            if not groups:
                await self.emit(state, "warning", {"message": "No duplicate images found."})

            for keeper, *dupes in groups:
                if state.cancel_requested:
                    break

                file_index = len(state.downloaded_files)
                state.downloaded_files.append(keeper)
                state.stats["downloaded"] += 1
                await self.emit(
                    state, "downloaded",
                    {
                        "query": f"{len(dupes)} duplicate{'s' if len(dupes) != 1 else ''} found",
                        "path": str(keeper), "url": "", "index": file_index,
                    },
                )

                for dupe in dupes:
                    if state.cancel_requested:
                        break
                    try:
                        await asyncio.to_thread(send2trash, str(dupe))
                        # A duplicate image's own caption file (if any) is now
                        # orphaned -- send it along rather than leave it behind
                        # pointing at nothing.
                        txt_path = dupe.with_suffix(".txt")
                        if txt_path.exists():
                            await asyncio.to_thread(send2trash, str(txt_path))
                        state.stats["duplicates"] += 1
                        await self.emit(
                            state, "skip",
                            {
                                "query": keeper.name, "url": str(dupe),
                                "error": f"duplicate of {keeper.name} -- moved to Recycle Bin",
                                "duplicate": True, "filtered": False,
                            },
                        )
                    except Exception as e:
                        state.stats["errors"] += 1
                        await self.emit(
                            state, "warning",
                            {"message": f"Could not remove duplicate {dupe.name}: {e}"},
                        )

            state.status = "cancelled" if state.cancel_requested else "done"
        except Exception as e:
            logger.exception("Dedup job %s failed", state.id)
            state.status = "error"
            await self.emit(state, "warning", {"message": f"Job failed: {e}"})
        finally:
            await self.emit(state, "status", {"status": state.status, "stats": state.stats})

    async def run_caption_files_job(self, state: JobState) -> None:
        """Caption a specific, user-picked set of files (a gallery
        multi-select) -- independent of any download job or whole-folder
        scan. Always (re)writes the caption; there's no separate overwrite
        flag here, since picking exactly these files *is* that decision."""
        req: CaptionFilesRequest = state.request
        state.status = "running"
        await self.emit(state, "status", {"status": "running"})

        captioner = build_captioner(req.method, req.llm, req.wd14, trigger=req.trigger)
        try:
            paths = [Path(p) for p in req.paths]
            if not paths:
                await self.emit(state, "warning", {"message": "No files selected."})

            for path in paths:
                if state.cancel_requested:
                    break
                if not path.is_file():
                    state.stats["errors"] += 1
                    await self.emit(state, "warning", {"message": f"File no longer exists: {path.name}"})
                    continue

                try:
                    content = path.read_bytes()
                    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                    caption = await captioner.caption(content, mime)
                except Exception as e:
                    state.stats["errors"] += 1
                    await self.emit(state, "warning", {"message": f"Captioning failed for {path.name}: {e}"})
                    continue

                txt_path = path.with_suffix(".txt")
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
            logger.exception("Caption-files job %s failed", state.id)
            state.status = "error"
            await self.emit(state, "warning", {"message": f"Job failed: {e}"})
        finally:
            await captioner.aclose()
            await self.emit(state, "status", {"status": state.status, "stats": state.stats})

    async def run_delete_files_job(self, state: JobState) -> None:
        """Delete a specific, user-picked set of files (a gallery
        multi-select) -- to the Recycle Bin via send2trash, not a permanent
        delete. Independent of the hash-based dedup job."""
        req: DeleteFilesRequest = state.request
        state.status = "running"
        await self.emit(state, "status", {"status": "running"})

        try:
            paths = [Path(p) for p in req.paths]
            if not paths:
                await self.emit(state, "warning", {"message": "No files selected."})

            for path in paths:
                if state.cancel_requested:
                    break
                try:
                    if path.is_file():
                        await asyncio.to_thread(send2trash, str(path))
                    # An orphaned caption is worse than no caption -- take it
                    # with the image, same as the dedup job does.
                    txt_path = path.with_suffix(".txt")
                    if txt_path.exists():
                        await asyncio.to_thread(send2trash, str(txt_path))
                    # Reuses the "duplicates" counter as a generic "removed"
                    # count -- the UI relabels it for this job kind, same as
                    # it already does for the dedup job.
                    state.stats["duplicates"] += 1
                    await self.emit(
                        state, "skip",
                        {"query": path.name, "url": str(path),
                         "error": "moved to Recycle Bin", "duplicate": True, "filtered": False},
                    )
                except Exception as e:
                    state.stats["errors"] += 1
                    await self.emit(state, "warning", {"message": f"Could not remove {path.name}: {e}"})

            state.status = "cancelled" if state.cancel_requested else "done"
        except Exception as e:
            logger.exception("Delete-files job %s failed", state.id)
            state.status = "error"
            await self.emit(state, "warning", {"message": f"Job failed: {e}"})
        finally:
            await self.emit(state, "status", {"status": state.status, "stats": state.stats})


job_manager = JobManager()
