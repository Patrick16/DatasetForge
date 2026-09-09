# Landscape notes & roadmap ideas

Working notes from a competitive-landscape pass, kept for the next planning
session rather than left buried in chat history. Nothing here is committed
to -- it's a starting point for "how should DatasetForge grow next."

## What DatasetForge actually is

One local web tool covering the whole pipeline from "a list of topics" to "a
captioned image dataset": type queries -> optionally have an LLM expand them
into variations -> download -> optionally have a vision LLM caption each
image, with an explicit trigger-word role (subject / style / action /
custom). Provider-agnostic (Ollama / LM Studio / any OpenAI-compatible cloud
API), zero install beyond `pip install -r requirements.txt`.

Most existing tools cover only one half of that pipeline (either the
scraping half or the captioning half) -- see below.

## Download-only tools (no AI)

| Tool | What it does | How it differs from DatasetForge |
|---|---|---|
| **gallery-dl** | CLI scraper with per-site extractors (Pixiv, Danbooru, Twitter, ...) | Site-specific extractors, not query-based search; no UI, no AI |
| **icrawler** (Python lib) | Google/Bing/Baidu image crawlers by keyword | Similar query-search mechanic, but no UI, no LLM expansion, no captioning |
| **img2dataset** (rom1504) | Massively parallel download from an already-built URL list (LAION-style) | Doesn't search -- expects a ready URL list; built for millions of images, not "10 images across 5 topics" |
| bing/google-images-download scripts | Simple CLI, N images per keyword | No query expansion, no captioning, brittle against search-engine markup changes |
| Browser extensions (Fatkun, etc.) | Manual bulk-download from the currently open page | Manual, page-scoped -- not "give me topics, get a dataset" |

## Captioning-only tools (no downloading)

The closer competition, especially for the trigger-word feature:

| Tool | What it does | How it differs from DatasetForge |
|---|---|---|
| **kohya_ss / sd-scripts** | The standard LoRA/Dreambooth training GUI; ships BLIP captioning + a WD14 tagger | Captioning is a side feature of a trainer, not a standalone flow; WD14 produces booru-style tags (`1girl, red_hair, ...`), not natural-language captions; no web scraping at all |
| **taggui** (jhc13) | Desktop GUI purpose-built for captioning LoRA datasets, supports local vision LLMs (LLaVA, CogVLM, ...) | Closest analog to the captioning half specifically -- but no image downloading, and no concept of a trigger word's "role" (subject/style/action) |
| **Dataset-Tag-Editor-Standalone** | Tag editor (WD14 + BLIP + batch find/replace) | Editing tags you already have, not collecting data or LLM natural-language captioning |
| **FiftyOne** (Voxel51) | Heavyweight ML dataset curation/visualization, has a "model zoo" with captioning models | General-purpose ML platform, not tuned for LoRA datasets; much higher setup cost |

## What's actually distinctive about DatasetForge

1. **One flow, start to finish** -- topics in, captioned dataset out. Normally
   this takes 2-3 different tools chained together by hand.
2. **Trigger word with an explicit role** (subject / style / action / custom)
   -- haven't found another tool that asks what the trigger word *means* and
   changes the captioning prompt accordingly. Usually it's either a flat
   prefix tag (WD14-style) or manual find/replace after the fact.
3. **LLM-agnostic by design** -- Ollama / LM Studio / cloud through one
   OpenAI-compatible interface, with model auto-discovery and an unload
   button. Most captioning tools hardcode a specific model (BLIP/WD14) or
   require manually wiring up an inference server.
4. **Zero install for the end user** -- just a local web page over FastAPI,
   nothing to train or assemble beforehand.

## Honest weak spots vs. the mature alternatives

- **WD14 tagger** (taggui/kohya_ss) usually gives more consistent,
  "trainable" tags for anime/booru-style work than a generic vision-LLM
  caption. That's a specialized, widely-used, fine-tuned tool -- DatasetForge
  doesn't try to compete with it head-on yet.
- **gallery-dl** is far more robust and covers vastly more sites. DatasetForge
  is pinned to one engine (DuckDuckGo via `ddgs`), not an affiliated API, and
  can break if they change something on their end.
- No dataset-wide batch tag editing (find/replace across already-written
  `.txt` files).
- No similarity-based de-dup, only exact content-hash de-dup.
- No integration with the actual training step (kohya_ss trains; DatasetForge
  stops at "captioned folder ready for a trainer").

## Candidate next steps (not decided -- discuss before building any of these)

Roughly in the order the weak spots above suggest, not a priority order:

- **WD14/booru tagger mode** as an alternative captioning backend alongside
  the vision-LLM path, for anime/booru-style datasets where tag lists train
  better than natural-language captions.
- **Batch find/replace across existing captions** -- edit tags/captions for a
  folder that's already been captioned, without re-running the vision model.
- **A second search backend** (e.g. Google Custom Search API, Bing, or an
  aggregator like SerpAPI) so the tool isn't a single point of failure tied
  to one unofficial engine -- `ImageSearchProvider` in `app/search/` was
  already designed to make this a drop-in addition.
- **Similarity-based de-dup** (perceptual hash) in addition to exact
  content-hash de-dup, to catch near-duplicate images from different sources.
- **Dataset-level review UI** -- browse/filter/bulk-delete already-downloaded
  images and their captions from one screen, instead of only the per-job
  thumbnail gallery.

## Prior art references (for whoever picks this up)

- kohya_ss / sd-scripts: <https://github.com/bmaltais/kohya_ss>
- taggui: <https://github.com/jhc13/taggui>
- gallery-dl: <https://github.com/mikf/gallery-dl>
- img2dataset: <https://github.com/rom1504/img2dataset>
- Dataset-Tag-Editor-Standalone: <https://github.com/toshiaki1729/dataset-tag-editor-standalone>
- FiftyOne: <https://github.com/voxel51/fiftyone>
