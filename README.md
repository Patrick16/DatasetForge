# DatasetForge

**Build and caption image datasets for LoRA / fine-tuning — locally, from a list of text queries.**

DatasetForge is a self-hosted web tool: you enter text queries, it finds and downloads matching images, removes duplicates, and writes training-ready captions with a local or cloud LLM. Everything runs on your machine; API keys never leave your browser.

![Python](https://img.shields.io/badge/python-FastAPI-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Query → dataset in one step.** N images per query, configurable concurrency, format and minimum-size filters.
- **LLM query expansion.** An LLM generates extra variations of each query to widen coverage.
- **Two captioning methods:**
  - **Vision LLM** — natural-language descriptions;
  - **WD14 tagger** — local tag-style captions via ONNX Runtime (Hugging Face models).
- **Trigger words.** Adds a token to every caption with a role you choose (subject, style, action or custom) for LoRA / DreamBooth training.
- **Deduplication.** Content-hash de-duplication during download, plus a standalone "remove duplicate images" action (duplicates go to the Recycle Bin, not deleted).
- **Live progress.** Thumbnail gallery and job status streamed over WebSocket; separate counters for filtered files and real errors.
- **Model lifecycle.** Discovers models on local LLM servers and unloads them from memory on demand and on graceful shutdown.
- **Caption an existing folder** without downloading anything.

## Works with any LLM provider

| Provider | Default base URL |
| --- | --- |
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| Any OpenAI-compatible cloud API | set in the UI |

Includes an opt-in workaround for reasoning ("thinking") models that otherwise return no final answer.

## Architecture

```
Browser UI (HTML/CSS/JS) ──HTTP / WebSocket──► FastAPI server
                                                 ├─ jobs.py          job orchestration (download, caption)
                                                 ├─ download.py      download, validation, de-duplication
                                                 ├─ search/          one module per provider behind a shared interface
                                                 ├─ captioning.py    vision LLM ⇄ WD14 strategy selection
                                                 ├─ llm_client.py    OpenAI-compatible client
                                                 └─ model_control.py / model_registry.py   model unloading & tracking
```

Design notes:

- **Pluggable search providers.** Each source implements `search/base.py`; adding a new one does not touch the job pipeline.
- **Strategy-based captioning.** The captioning method is selected at runtime, so the pipeline is the same for local and cloud models.
- **No server-side secrets.** Settings and keys live only in the browser's `localStorage`.
- **Tests without the network.** `pytest` suite with mocked HTTP; no live services required.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000.

Run the tests:

```bash
pytest
```

## Search providers

- **DuckDuckGo** — default, no API key.
- **Yandex**, **Google** (experimental).
- **Additional providers (opt-in)** — tag-based image boards with a Danbooru-compatible API; useful together with the WD14 tagger. Disabled by default; see `app/search/` for details.

SafeSearch is passed through to every provider.

## Output layout

Images are saved either into per-query subfolders or into one flat folder with query-prefixed file names. Captions are written next to each image as `.txt` files.

## Tech stack

Python · FastAPI · uvicorn · Pydantic · WebSocket · ONNX Runtime · Pillow · pytest · vanilla HTML/CSS/JS

## Known limitations

MVP: single-user, local use. See the issues for the roadmap.

## License

MIT
