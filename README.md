# DatasetForge

[![Tests](https://github.com/Patrick16/DatasetForge/actions/workflows/tests.yml/badge.svg)](https://github.com/Patrick16/DatasetForge/actions/workflows/tests.yml)

A local web tool: given a list of text queries, it downloads images from the web
(search via DuckDuckGo by default, no API key needed -- Yandex, Google, and
booru boards are also available, see **Search providers** below) into a folder
you choose. Optionally:

- **Query LLM** — generates N additional search-query variations for each input
  query (Ollama / LM Studio / any OpenAI-compatible cloud API).
- **Vision LLM (captioning)** — for every downloaded image, saves a `<name>.txt`
  file with a generated description (same provider choice).

The whole thing runs as a single local FastAPI server; the web UI (no build step)
talks to it over HTTP + WebSocket.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the backend (`app/`) with fakes/mocked HTTP instead of real
network calls or a live model server -- no external services required.

## Features

- Multiple queries at once, N images per query, configurable concurrency.
- `query_subfolders` toggle: one subfolder per query, or a flat folder with
  query-prefixed filenames.
- Format/min-size filters, content-hash de-duplication. `gif` is included by
  default alongside jpg/png/webp (only `bmp` is off by default).
- **SafeSearch control** — a dropdown (Off / Moderate / On) next to the format
  filters. This isn't a filter *we* apply -- it's passed straight through to
  the search source itself (DuckDuckGo/Yandex/Google) or turned into a
  `rating:` tag (booru boards), so results it excludes never reach this app
  at all. Previously there was no way to change it and it was silently always
  on (`ddgs`'s own default, "moderate"), which made the tool unable to fetch
  results DuckDuckGo itself considers not-safe-for-work no matter what you
  searched for or how your local filters were set. Confirmed live for both
  DuckDuckGo and Yandex that "off" vs "on" actually returns a different
  result set for a borderline query.
- **"Filtered" vs "Errors" are counted separately.** Images your own
  format/min-size filters excluded show up under "Filtered" (expected,
  working as configured); network failures, bad data, etc. show up under
  "Errors". They used to be lumped into one generic "Errors" counter, which
  made it hard to tell whether something was actually broken or the filters
  were just doing their job.
- **Folder picker** — the "Browse…" buttons open a native OS folder dialog *on
  the machine running the server* (works because the server and browser are on
  the same machine for this local tool; it will not work if you access the UI
  from a different device on your network).
- **Model discovery** — the "Refresh" button next to a model field queries the
  configured server's `/v1/models` endpoint and lets you pick from a dropdown,
  instead of typing the model id by hand. If the server isn't reachable, you can
  also set a **models folder** and it will be scanned for `*.gguf` files as a
  fallback (best-effort — the file name may not always match the id the server
  expects).
- **Generate queries with LLM** — a dedicated button expands your queries
  *before* anything is downloaded and writes the result straight back into the
  queries box, so you always see (and can edit) exactly what's about to be
  searched. Starting a download never expands on its own — it always searches
  literally what's in the box, so nothing expands twice.
- **Guaranteed-ish image counts** — asking for N images per query fetches more
  search candidates than N up front and keeps trying them after
  failures/duplicates, instead of quietly returning fewer than N whenever some
  downloads fail. Some parallel over-shoot past N is possible (bounded by
  `min(concurrency, N)`) but running short is not.
- **Thumbnail gallery** — downloaded images appear at the bottom of the page as
  they complete, each tagged with its query and (once ready) its caption.
- Live progress (WebSocket): per-file success/error/duplicate log (collapsed
  by default — click "Raw log" to expand), running counters, cancel button.
- The LLM/vision settings blocks are collapsed behind a "Settings" toggle by
  default; only the enable checkbox is always visible.
- **Caption an existing folder** — inside the captioning card, point at any
  local folder and click "Caption this folder" to write a `<name>.txt` for
  every image already in it, completely independent of running a download.
  Reuses the same vision LLM settings. Options: include subfolders, and
  whether to overwrite `.txt` files that already exist (off by default, so
  re-running is cheap -- already-captioned images are skipped).
- **Trigger word** (for the folder-captioning action) — a token every caption
  should consistently include, e.g. for LoRA/Dreambooth training sets. What it
  *means* changes how it should read in a sentence, so you pick its role:
  a **subject** ("a photo of `word` smiling" instead of "a woman smiling"), a
  **style** ("...rendered in the `word` style"), an **action/pose** ("a man
  `word` in a park"), or **something else**, where you write the instruction
  yourself with `{word}` as the placeholder.
- **Disable reasoning/thinking mode** — a checkbox in each LLM block's Settings
  (Query LLM and Vision LLM). Some local "thinking" models spend their entire
  response budget on hidden reasoning and never emit an answer, no matter how
  you ask them not to (`/no_think`, `chat_template_kwargs.enable_thinking`,
  `think: false` -- none of it worked in testing against one such fine-tune).
  What *did* work: appending an already-closed, empty `<think></think>` to the
  conversation as if the model itself had written it. llama.cpp-based servers
  (LM Studio, text-generation-webui, koboldcpp) treat a trailing assistant
  message as a continuation to build on, so the model starts its real answer
  immediately -- measured going from 100+s (or an outright empty response) to
  ~1s, reliably. It's opt-in, not automatic, since this exact trick isn't
  guaranteed on Ollama or cloud APIs (a trailing assistant message may just be
  rejected or treated as literal history there instead of a prefill).
- **Unload models from memory** — a button at the top of the page frees every
  local model this app has actually used this session from RAM/VRAM: Ollama
  via `keep_alive: 0` on its native API, LM Studio via the `lms` CLI
  (`lms unload <model>`, so it must be on PATH -- it is by default alongside
  a normal LM Studio install). Scoped to models *this app* used -- a model you
  loaded some other way and never touched through this tool is left alone.
  Cloud models are reported as a no-op (there's nothing local to free). The
  same cleanup also runs automatically on a graceful shutdown (Ctrl+C in the
  terminal running `uvicorn`) -- but not on a force-kill (Task Manager "End
  task", `taskkill /F`, `kill -9`), since the OS never gives the process a
  chance to run cleanup code at all in that case. If LM Studio already
  auto-unloaded a model on its own (its own idle TTL) before you click the
  button, that's reported as "already not loaded", not an error -- `lms
  unload` itself exits 0 with a "Model Not Found" message in that case, which
  used to be passed straight through and read exactly like a failure.

## Search providers

Chosen via the "Search source" card. All four confirmed working live during
development (2026-09) except where noted:

| Provider | What it is | SafeSearch mechanism | Notes |
|---|---|---|---|
| **DuckDuckGo** (default) | No API key. Under the hood this actually proxies **Bing's** image index (confirmed by inspecting the `ddgs` library's own provider tagging) -- there is no independent DuckDuckGo image index. | `p` query param | The more reliable of the two ways to reach that same Bing data -- Bing's own direct scraping route in `ddgs` ignores the safesearch argument entirely. |
| **Yandex** | Unofficial scraping of `yandex.com/images`. Historically laxer filtering than Google/Bing. | `family` cookie (`0`=off, `2`=on, unset=default) | Confirmed live: toggling it changed 5/25 results on a borderline query. Width/height aren't extracted from the scrape -- harmless, since every file is re-measured with Pillow after downloading anyway. |
| **Google** (experimental) | Unofficial scraping. | `safe` query param | ⚠️ Confirmed unreliable in testing: Google returned an HTTP 429 bot-check on the *first* request, with full browser-like headers and no prior history. Expect frequent zero results; a block degrades to "0 found" plus a log warning, not a crash. Try Yandex or DuckDuckGo instead if this keeps failing for you. |
| **Booru board** | Danbooru-API-family boards, for tag-driven / anime-style datasets. | An explicit `rating:` tag, not a hidden toggle -- content is opt-in tagged (general/sensitive/questionable/explicit; naming varies by board) rather than filtered by a black-box heuristic. | See below -- access requirements vary a lot by board. |

### Booru board access (confirmed live, 2026-09)

- **e621** — works with zero configuration.
- **Gelbooru** / **Rule34.xxx** — now require an `api_key` + `user_id` (free
  account -> Account -> API Access Credentials); both returned HTTP
  401/"Missing authentication" without one. This is a change from their
  historical open access.
- **Danbooru** — supports `login` + `api_key` (HTTP Basic Auth, the
  documented method), but requests were blocked by a Cloudflare bot-check
  ("Just a moment...") in testing regardless, from at least some networks.
  Implemented, but best-effort -- it may just not work for you. Also:
  anonymous Danbooru API access is limited to 2 combined tags, which a
  multi-word query plus the rating tag can easily exceed.

To add another search source, implement `ImageSearchProvider.search()` under
`app/search/` and register it in `build_search_provider()` in
`app/search/__init__.py`.

## Optional LLM providers

Both blocks (query expansion and captioning) are configured independently right
in the UI:

| Provider | Default base URL | Requirements |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | `ollama serve` + `ollama pull <model>` |
| LM Studio | `http://localhost:1234/v1` | start the local server in LM Studio |
| Cloud API | set manually (e.g. `https://api.openai.com/v1`) | API key |

Captioning needs a vision-capable model, e.g. `moondream`, `llava`,
`qwen2.5-vl` on Ollama/LM Studio, or `gpt-4o-mini` in the cloud.

All keys/settings live only in the browser's `localStorage` — the server never
writes them to disk.

### Notes from real-world testing with LM Studio

- **Local CPU inference can be slow.** A 7B model with no GPU offload measured
  ~100+ seconds for a single short completion in testing; the same request took
  ~2 seconds once GPU offload was enabled (`lms load <model> --gpu max`, or set
  it in LM Studio's settings). If requests are timing out, check GPU offload
  first, then raise the "Response timeout" field.
- **Reasoning ("thinking") models can misbehave for this task.** Some
  reasoning-tuned local models spend their entire token budget on hidden
  `reasoning_content` and never emit the final short JSON answer, even with a
  generous `max_tokens`. If query expansion keeps coming back empty, try the
  **"Disable reasoning/thinking mode"** checkbox first (see Features above) --
  it fixed this reliably in testing. Failing that, switch to a plain
  (non-reasoning) instruct model.
- LM Studio only lists a model in `lms ls` / the Refresh dropdown if its files
  sit in their own `publisher/model-name/` folder (matching the `mmproj-*.gguf`
  file alongside the model, if it has vision support). Loose `.gguf` files
  directly under a publisher folder won't be indexed until moved into their own
  subfolder.

## Output layout

```
output_folder/
  query_1/              # if "create a subfolder for each query" is on
    img_0001.jpg
    img_0001.txt        # caption, if captioning is enabled
  query_2/
    ...
```
If subfolders are off, everything goes into one folder and filenames get a
query prefix.

## Project layout

```
app/
  main.py          # FastAPI: routes, progress WebSocket, folder picker, model listing, image serving
  models.py        # Pydantic schemas for job requests (download + caption-folder)
  jobs.py          # orchestration: download job (search -> worker pool -> caption)
                    # and standalone caption-folder job, sharing one JobState/WebSocket model
  download.py      # async download + validation + de-dup
  llm_client.py    # generic OpenAI-compatible API client + trigger-word prompt building
  local_models.py  # native folder picker + model discovery (server /v1/models + folder scan)
  model_registry.py # tracks which (provider, base_url, model) this process has used
  model_control.py # unloads a model from Ollama (native API) or LM Studio (`lms` CLI)
  search/
    __init__.py     # build_search_provider() factory (picks provider by SearchConfig)
    base.py         # ImageSearchProvider interface
    duckduckgo.py    # DuckDuckGo search (no API key)
    yandex.py        # Yandex Images scraping
    google.py        # Google Images scraping (experimental, unreliable)
    booru.py         # Danbooru/e621/Gelbooru/Rule34
static/
  index.html, style.css, app.js   # web UI, no build step
tests/                            # pytest suite for app/ (see "Tests" above)
```

## Known limitations (MVP)

- The folder picker and model discovery only work when the browser and the
  server run on the same machine (they call back into the local OS/network,
  not something a remote browser can do).
- One backend process keeps jobs in memory — restarting the server loses the
  progress of any job that hasn't finished.
- DuckDuckGo search is unofficial: it can break if they change something on
  their end (update the `ddgs` package if that happens).

## License

[MIT](LICENSE)

## Where to next

See [docs/landscape-and-roadmap-notes.md](docs/landscape-and-roadmap-notes.md)
for a rundown of similar tools, what's actually distinctive here, and a list
of candidate next steps to discuss before building any of them.
