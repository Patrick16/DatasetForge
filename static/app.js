const $ = (id) => document.getElementById(id);

const PROVIDER_DEFAULTS = {
  ollama: "http://localhost:11434/v1",
  lmstudio: "http://localhost:1234/v1",
  cloud: "https://api.openai.com/v1",
};

function wireProviderDefaults(providerSelect, baseUrlInput) {
  providerSelect.addEventListener("change", () => {
    baseUrlInput.value = PROVIDER_DEFAULTS[providerSelect.value] || "";
  });
}
wireProviderDefaults($("llm_provider"), $("llm_base_url"));
wireProviderDefaults($("cap_provider"), $("cap_base_url"));

function wireToggle(checkbox, ...blocks) {
  const sync = () => blocks.forEach((b) => b.classList.toggle("disabled", !checkbox.checked));
  checkbox.addEventListener("change", sync);
  sync();
}
wireToggle($("llm_enabled"), $("llm_block"));
wireToggle($("cap_enabled"), $("cap_vision_block"), $("cap_wd14_block"));
wireToggle($("trigger_enabled"), $("trigger_block"));

// ---- captioning method: vision LLM (a sentence) vs WD14 tagger (a booru-
// style tag list, local ONNX, no server needed) -- only one settings block
// is relevant at a time, and the trigger-word note below reads differently
// depending on which one is active (role-aware prompt vs. flat prepend). ----
const CAP_METHOD_NOTES = {
  vision_llm: "Needs a vision-capable model reachable via Ollama/LM Studio/a cloud API -- configure it below.",
  wd14: "Runs locally via ONNX -- no model server needed. Outputs a comma-separated tag list (e.g. \"solo, blue hair, outdoors\") instead of a sentence.",
};

function syncCapMethodUI() {
  const method = $("cap_method").value;
  $("cap_vision_block").hidden = method !== "vision_llm";
  $("cap_wd14_block").hidden = method !== "wd14";
  $("cap-method-note").textContent = CAP_METHOD_NOTES[method] || "";
}
$("cap_method").addEventListener("change", syncCapMethodUI);
syncCapMethodUI();

function syncGenerateBtn() {
  $("generate-btn").disabled = !$("llm_enabled").checked;
}
$("llm_enabled").addEventListener("change", syncGenerateBtn);

function syncTriggerCustomField() {
  $("trigger_custom_field").hidden = $("trigger_role").value !== "custom";
}
$("trigger_role").addEventListener("change", syncTriggerCustomField);
syncTriggerCustomField();

// ---- search source: booru sub-fields only matter (and only show) when
// "booru" is selected, and each provider gets a short note about what to
// expect (reliability, what SafeSearch actually controls there). ----
const SEARCH_PROVIDER_NOTES = {
  duckduckgo: "No configuration needed. Effectively proxies Bing's image index.",
  yandex: "Unofficial scraping (like DuckDuckGo) -- can break if Yandex changes their page markup. Historically laxer SafeSearch than Google/Bing.",
  google: "⚠️ Unofficial scraping -- confirmed unreliable in testing (Google blocked requests almost immediately, even with full browser headers). Expect frequent zero results; try Yandex or DuckDuckGo instead if this keeps failing.",
  booru: "Content is explicitly rating-tagged rather than hidden behind a SafeSearch toggle -- see board options below.",
};

function syncSearchProviderUI() {
  const provider = $("search_provider").value;
  $("booru_block").classList.toggle("disabled", provider !== "booru");
  $("search-provider-note").textContent = SEARCH_PROVIDER_NOTES[provider] || "";
}
$("search_provider").addEventListener("change", syncSearchProviderUI);
syncSearchProviderUI();

// ---- native folder picker (server-side dialog; browser & server must be on
// the same machine) -- used both by the generic "browse" buttons (models
// folders) and, separately, by the active-folder button below. ----
document.querySelectorAll(".browse-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const targetId = btn.dataset.target;
    const title = btn.dataset.title || "Select folder";
    const input = $(targetId);
    const original = btn.textContent;
    btn.textContent = "...";
    btn.disabled = true;
    try {
      const resp = await fetch("/api/pick-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ initial_dir: input.value.trim() || null, title }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        alert("Folder picker unavailable: " + (err.detail || resp.statusText));
        return;
      }
      const { folder } = await resp.json();
      if (folder) {
        input.value = folder;
        saveForm();
      }
    } catch (err) {
      alert("Could not reach the server for the folder picker: " + err);
    } finally {
      btn.textContent = original;
      btn.disabled = false;
    }
  });
});

// ---- model discovery: query the running server + optionally scan a local models folder ----
async function refreshModels(kind) {
  const baseUrlInput = $(`${kind}_base_url`);
  const apiKeyInput = $(`${kind}_api_key`);
  const folderInput = $(`${kind}_models_folder`);
  const datalist = $(`${kind}_model_list`);
  const hint = $(`${kind}_model_hint`);
  const btn = document.querySelector(`.refresh-btn[data-kind="${kind}"]`);

  btn.disabled = true;
  btn.textContent = "...";
  hint.textContent = "";
  hint.className = "hint";

  try {
    const resp = await fetch("/api/llm/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: baseUrlInput.value.trim(),
        api_key: apiKeyInput.value.trim() || null,
        models_folder: folderInput.value.trim() || null,
      }),
    });
    const data = await resp.json();
    datalist.innerHTML = "";
    const seen = new Set();
    for (const m of [...(data.server_models || []), ...(data.folder_models || [])]) {
      if (seen.has(m)) continue;
      seen.add(m);
      const opt = document.createElement("option");
      opt.value = m;
      datalist.appendChild(opt);
    }
    const parts = [];
    if (data.server_models?.length) parts.push(`${data.server_models.length} from server`);
    if (data.folder_models?.length) parts.push(`${data.folder_models.length} from folder`);
    const errors = [];
    if (data.server_error) errors.push(`server: ${data.server_error}`);
    if (data.folder_error) errors.push(`folder: ${data.folder_error}`);
    if (parts.length) {
      hint.textContent = `Found: ${parts.join(", ")}.`;
      hint.classList.add("ok");
      if (errors.length) hint.textContent += ` (${errors.join("; ")})`;
    } else if (errors.length) {
      hint.textContent = errors.join("; ");
      hint.classList.add("error");
    } else {
      hint.textContent = "No models found.";
      hint.classList.add("error");
    }
  } catch (err) {
    hint.textContent = "Request failed: " + err;
    hint.classList.add("error");
  } finally {
    btn.disabled = false;
    btn.textContent = "↻";
  }
}
document.querySelectorAll(".refresh-btn").forEach((btn) => {
  btn.addEventListener("click", () => refreshModels(btn.dataset.kind));
});

// ---- unload models from memory: frees whatever local models this app has
// actually used this session (tracked server-side) from Ollama/LM Studio.
// The server also tries this automatically on a graceful shutdown -- this
// button is for freeing memory without closing the app, e.g. between runs. ----
$("unload-models-btn").addEventListener("click", async () => {
  const btn = $("unload-models-btn");
  const hint = $("unload-models-hint");
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Unloading...";
  hint.textContent = "";
  hint.className = "hint";

  try {
    const resp = await fetch("/api/llm/unload-all", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);

    if (!data.results || data.results.length === 0) {
      hint.textContent = data.message || "Nothing to unload.";
      hint.classList.add("ok");
    } else {
      const ok = data.results.filter((r) => r.ok);
      const failed = data.results.filter((r) => !r.ok);
      const parts = [];
      if (ok.length) parts.push(`unloaded: ${ok.map((r) => r.model).join(", ")}`);
      if (failed.length) parts.push(`failed: ${failed.map((r) => `${r.model} (${r.message})`).join("; ")}`);
      hint.textContent = parts.join(" -- ");
      hint.classList.add(failed.length ? "error" : "ok");
    }
  } catch (err) {
    hint.textContent = "Request failed: " + err;
    hint.classList.add("error");
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});

// ---- persist form state in localStorage (settings only, no downloaded content) ----
const STORAGE_KEY = "datasetforge:form";
const FIELD_IDS = [
  "queries", "n_per_query", "concurrency", "query_subfolders",
  "search_provider", "booru_site", "booru_api_key", "booru_user_id", "booru_login",
  "min_width", "min_height", "safesearch",
  "llm_enabled", "llm_provider", "llm_variations", "llm_base_url", "llm_model", "llm_api_key", "llm_timeout", "llm_models_folder", "llm_disable_reasoning",
  "cap_enabled", "cap_method", "cap_provider", "cap_model", "cap_base_url", "cap_api_key", "cap_timeout", "cap_models_folder", "cap_disable_reasoning",
  "cap_wd14_model", "cap_wd14_general_threshold", "cap_wd14_character_threshold",
  "capfolder_recursive", "capfolder_overwrite",
  "dedup_recursive",
  "trigger_enabled", "trigger_word", "trigger_role", "trigger_custom",
];

function saveForm() {
  const data = {};
  for (const id of FIELD_IDS) {
    const el = $(id);
    if (!el) continue;
    data[id] = el.type === "checkbox" ? el.checked : el.value;
  }
  data.formats = Array.from(document.querySelectorAll("#formats input:checked")).map((el) => el.value);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function loadForm() {
  let data;
  try {
    data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch {
    return;
  }
  if (!data) return;
  for (const id of FIELD_IDS) {
    const el = $(id);
    if (!el || !(id in data)) continue;
    if (el.type === "checkbox") el.checked = data[id];
    else el.value = data[id];
  }
  if (Array.isArray(data.formats)) {
    document.querySelectorAll("#formats input").forEach((el) => {
      el.checked = data.formats.includes(el.value);
    });
  }
  $("llm_block").classList.toggle("disabled", !$("llm_enabled").checked);
  $("cap_vision_block").classList.toggle("disabled", !$("cap_enabled").checked);
  $("cap_wd14_block").classList.toggle("disabled", !$("cap_enabled").checked);
  $("trigger_block").classList.toggle("disabled", !$("trigger_enabled").checked);
}
loadForm();
syncGenerateBtn();
syncTriggerCustomField();
syncSearchProviderUI();
syncCapMethodUI();
document.body.addEventListener("change", (e) => {
  if (e.target.closest("#job-form, .rail-left, .rail-right")) saveForm();
});

// ---- right-rail accordions: click the header to expand/collapse. A checkbox
// sitting inside the header (query-expansion / captioning "enabled" toggles)
// stops its own clicks from also toggling the accordion. ----
document.querySelectorAll("[data-accordion]").forEach((acc) => {
  const toggle = acc.querySelector("[data-accordion-toggle]");
  const body = acc.querySelector(".accordion-body");
  toggle.addEventListener("click", () => {
    const open = acc.classList.toggle("open");
    body.hidden = !open;
  });
});
document.querySelectorAll("[data-stop-toggle]").forEach((el) => {
  el.addEventListener("click", (e) => e.stopPropagation());
});

// ---- footer status bar: collapse to just the stat line, hiding the raw log ----
$("statusbar-head").addEventListener("click", (e) => {
  if (e.target.closest("#cancel-btn")) return;
  $("statusbar").classList.toggle("expanded");
});

// ---- active folder: every action below (download, caption, dedupe) runs
// against this one folder, and it's also what the center gallery renders. ----
const ACTIVE_FOLDER_KEY = "datasetforge:active_folder";
let activeFolder = localStorage.getItem(ACTIVE_FOLDER_KEY) || "";

function setActiveFolder(folder) {
  activeFolder = folder;
  if (folder) localStorage.setItem(ACTIVE_FOLDER_KEY, folder);
  else localStorage.removeItem(ACTIVE_FOLDER_KEY);
  renderActiveFolderPath();
}

function renderActiveFolderPath() {
  const pathEl = $("active-folder-path");
  if (activeFolder) {
    pathEl.textContent = activeFolder;
    pathEl.title = activeFolder;
    pathEl.classList.remove("empty");
  } else {
    pathEl.textContent = "No folder selected yet";
    pathEl.title = "";
    pathEl.classList.add("empty");
  }
}

function folderBaseName(folder) {
  const parts = folder.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || folder;
}

// The gallery's last-loaded file list (path/name/caption/size/width/height),
// used both to render the grid and to look things up for the modal without
// a round-trip -- and which paths are currently multi-selected.
let galleryFiles = [];
const selectedPaths = new Set();

// Loads (or reloads) the gallery from whatever's on disk in `activeFolder`
// right now -- called on open, after each job's terminal status, and
// (debounced) while a job is progressing, so newly written files/captions
// show up without a manual refresh.
async function loadGallery() {
  const grid = $("thumb-grid");
  const empty = $("gallery-empty");
  const title = $("gallery-title");
  const meta = $("gallery-meta");
  const folderMeta = $("active-folder-meta");

  if (!activeFolder) {
    grid.innerHTML = "";
    empty.hidden = false;
    empty.textContent = "Open a folder on the left to see its images here.";
    title.textContent = "No folder open";
    meta.textContent = "";
    folderMeta.textContent = "";
    return;
  }

  title.textContent = folderBaseName(activeFolder);
  try {
    const resp = await fetch(`/api/browse-folder?folder=${encodeURIComponent(activeFolder)}`);
    const data = await resp.json();
    if (!resp.ok) {
      grid.innerHTML = "";
      empty.hidden = false;
      empty.textContent = `Could not read this folder: ${data.detail || resp.statusText}`;
      meta.textContent = "";
      folderMeta.textContent = "";
      return;
    }
    meta.textContent = `${data.total} image${data.total === 1 ? "" : "s"} · ${data.captioned} captioned`;
    folderMeta.innerHTML = data.total
      ? `<b>${data.total}</b> files · <b>${data.captioned}</b> captioned`
      : "Empty folder so far";

    // Selection survives a reload (a job progressing, a manual reopen) as
    // long as the same paths are still there -- drop anything that vanished
    // (e.g. just got deleted) instead of leaving a phantom "N selected".
    galleryFiles = data.files;
    const stillPresent = new Set(galleryFiles.map((f) => f.path));
    for (const p of Array.from(selectedPaths)) {
      if (!stillPresent.has(p)) selectedPaths.delete(p);
    }

    if (!data.files.length) {
      grid.innerHTML = "";
      empty.hidden = false;
      empty.textContent = "No images in this folder yet.";
      renderSelectionBar();
      return;
    }
    empty.hidden = true;
    grid.innerHTML = data.files.map((f) => `
      <figure class="thumb-card ${selectedPaths.has(f.path) ? "selected" : ""}" data-path="${escapeHtml(f.path)}">
        <div class="thumb-img">
          <input type="checkbox" class="thumb-select" title="Select" ${selectedPaths.has(f.path) ? "checked" : ""} />
          <img src="/api/local-image?path=${encodeURIComponent(f.path)}" loading="lazy" alt="${escapeHtml(f.name)}" />
        </div>
        <figcaption class="thumb-body">
          <div class="thumb-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
          <div class="thumb-caption ${f.caption ? "" : "empty"}">${f.caption ? escapeHtml(f.caption) : "No caption yet"}</div>
        </figcaption>
      </figure>
    `).join("");
    renderSelectionBar();
  } catch (err) {
    empty.hidden = false;
    empty.textContent = "Request failed: " + err;
  }
}

let galleryReloadTimer = null;
function scheduleGalleryReload(delay = 700) {
  if (!activeFolder) return;
  clearTimeout(galleryReloadTimer);
  galleryReloadTimer = setTimeout(loadGallery, delay);
}

$("open-folder-btn").addEventListener("click", async () => {
  const btn = $("open-folder-btn");
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "...";
  try {
    const resp = await fetch("/api/pick-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial_dir: activeFolder || null, title: "Select the active folder" }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert("Folder picker unavailable: " + (err.detail || resp.statusText));
      return;
    }
    const { folder } = await resp.json();
    if (folder) {
      setActiveFolder(folder);
      await loadGallery();
    }
  } catch (err) {
    alert("Could not reach the server for the folder picker: " + err);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
});

renderActiveFolderPath();
loadGallery();

// ---- gallery multi-select: a checkbox in the corner of each card, separate
// from clicking the card itself (which opens the full-size modal) -- picked
// files feed "Caption selected" / "Delete selected" in the selection bar. ----
function renderSelectionBar() {
  const bar = $("selection-bar");
  const n = selectedPaths.size;
  bar.hidden = n === 0;
  if (n > 0) $("selection-count").textContent = `${n} selected`;
}

function setCardSelected(path, selected) {
  if (selected) selectedPaths.add(path);
  else selectedPaths.delete(path);
  const card = $("thumb-grid").querySelector(`.thumb-card[data-path="${cssEscape(path)}"]`);
  if (card) {
    card.classList.toggle("selected", selected);
    const cb = card.querySelector(".thumb-select");
    if (cb) cb.checked = selected;
  }
  renderSelectionBar();
}

function clearSelection() {
  for (const path of Array.from(selectedPaths)) setCardSelected(path, false);
}

// CSS.escape isn't polyfilled everywhere this might run, but every path here
// came from our own backend (not user-typed), so a minimal escape is enough
// to keep it a valid attribute-value selector.
function cssEscape(s) {
  return String(s).replace(/["\\]/g, "\\$&");
}

$("thumb-grid").addEventListener("click", (e) => {
  const card = e.target.closest(".thumb-card");
  if (!card) return;
  const path = card.dataset.path;

  if (e.target.classList.contains("thumb-select")) {
    setCardSelected(path, e.target.checked);
    return;
  }
  const index = galleryFiles.findIndex((f) => f.path === path);
  if (index !== -1) openModal(index);
});

$("select-all-btn").addEventListener("click", () => {
  for (const f of galleryFiles) setCardSelected(f.path, true);
});
$("clear-selection-btn").addEventListener("click", () => clearSelection());

$("caption-selected-btn").addEventListener("click", async () => {
  const paths = Array.from(selectedPaths);
  if (!paths.length) return;
  await startJob("/api/caption-files", buildCaptionFilesRequest(paths), "caption_selected");
});

$("delete-selected-btn").addEventListener("click", async () => {
  const paths = Array.from(selectedPaths);
  if (!paths.length) return;
  const proceed = confirm(
    `Move ${paths.length} selected image${paths.length === 1 ? "" : "s"} (and any caption) to the Recycle Bin? ` +
    "This can be undone from the Recycle Bin, but not from here."
  );
  if (!proceed) return;
  await startJob("/api/delete-files", { paths }, "delete_selected");
});

// ---- image modal: click a card to see it full-size, with its metadata and
// caption -- and step through the rest of the gallery without closing it. ----
let modalIndex = -1;

function formatBytes(n) {
  if (n == null) return "unknown";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

function openModal(index) {
  modalIndex = index;
  renderModal();
  $("image-modal").hidden = false;
}

function closeModal() {
  $("image-modal").hidden = true;
  modalIndex = -1;
}

function renderModal() {
  const f = galleryFiles[modalIndex];
  if (!f) return;

  $("modal-image").src = `/api/local-image?path=${encodeURIComponent(f.path)}`;
  $("modal-image").alt = f.name;
  $("modal-filename").textContent = f.name;

  const dims = f.width && f.height ? `${f.width} × ${f.height}px` : "unknown";
  $("modal-meta").innerHTML = `
    <div><dt>Dimensions</dt><dd>${dims}</dd></div>
    <div><dt>File size</dt><dd>${formatBytes(f.size)}</dd></div>
    <div><dt>Path</dt><dd>${escapeHtml(f.path)}</dd></div>
  `;

  const capEl = $("modal-caption");
  capEl.textContent = f.caption || "No caption yet";
  capEl.classList.toggle("empty", !f.caption);

  $("modal-prev-btn").disabled = modalIndex <= 0;
  $("modal-next-btn").disabled = modalIndex >= galleryFiles.length - 1;
}

$("modal-close-btn").addEventListener("click", closeModal);
$("image-modal").addEventListener("click", (e) => {
  if (e.target.id === "image-modal") closeModal();
});
$("modal-prev-btn").addEventListener("click", () => {
  if (modalIndex > 0) {
    modalIndex--;
    renderModal();
  }
});
$("modal-next-btn").addEventListener("click", () => {
  if (modalIndex < galleryFiles.length - 1) {
    modalIndex++;
    renderModal();
  }
});
document.addEventListener("keydown", (e) => {
  if ($("image-modal").hidden) return;
  if (e.key === "Escape") closeModal();
  else if (e.key === "ArrowLeft") $("modal-prev-btn").click();
  else if (e.key === "ArrowRight") $("modal-next-btn").click();
});

// ---- query generation: expand queries with the LLM and write the result back
// into the queries textarea *before* any download happens, so the user always
// sees exactly what will be searched. Shared by the manual "Generate" button
// and the automatic pre-step when starting a download. ----
async function generateQueries() {
  if (!$("llm_enabled").checked) return true;
  const lines = $("queries").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!lines.length) return true;

  const btn = $("generate-btn");
  const hint = $("generate-hint");
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Generating...";
  hint.textContent = "";
  hint.className = "hint";

  try {
    const body = {
      queries: lines,
      variations_per_query: Number($("llm_variations").value) || 3,
      llm: {
        provider: $("llm_provider").value,
        base_url: $("llm_base_url").value.trim(),
        model: $("llm_model").value.trim(),
        api_key: $("llm_api_key").value.trim() || null,
        timeout_seconds: Number($("llm_timeout").value) || 180,
        disable_reasoning: $("llm_disable_reasoning").checked,
      },
    };
    const resp = await fetch("/api/llm/expand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);

    $("expanded-list").innerHTML = "";
    $("expanded-card").hidden = true;
    const allLines = [];
    let errorCount = 0;
    for (const r of data.results || []) {
      allLines.push(r.query);
      if (r.variations?.length) {
        allLines.push(...r.variations);
        addExpandedGroup(r.query, r.variations);
      }
      if (r.error) errorCount++;
    }
    $("queries").value = allLines.join("\n");
    saveForm();

    if (errorCount === 0) {
      hint.textContent = `Generated ${allLines.length - lines.length} new quer${allLines.length - lines.length === 1 ? "y" : "ies"} (${allLines.length} total).`;
      hint.classList.add("ok");
    } else {
      hint.textContent = `${errorCount} of ${lines.length} quer${lines.length === 1 ? "y" : "ies"} failed to expand -- continuing with the rest. Common cause: a "thinking" model burning its whole budget on hidden reasoning -- try a plain instruct model.`;
      hint.classList.add("error");
    }
    return true;
  } catch (err) {
    hint.textContent = "Query generation failed: " + err;
    hint.classList.add("error");
    return false;
  } finally {
    btn.textContent = originalLabel;
    syncGenerateBtn();
  }
}
$("generate-btn").addEventListener("click", () => generateQueries());

// ---- build request body ----
function buildRequest() {
  const queries = $("queries").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const formats = Array.from(document.querySelectorAll("#formats input:checked")).map((el) => el.value);

  return {
    queries,
    n_per_query: Number($("n_per_query").value),
    output_folder: activeFolder,
    query_subfolders: $("query_subfolders").checked,
    concurrency: Number($("concurrency").value),
    filters: {
      formats,
      min_width: Number($("min_width").value),
      min_height: Number($("min_height").value),
      safesearch: $("safesearch").value,
    },
    search: {
      provider: $("search_provider").value,
      booru_site: $("booru_site").value,
      booru_api_key: $("booru_api_key").value.trim() || null,
      booru_user_id: $("booru_user_id").value.trim() || null,
      booru_login: $("booru_login").value.trim() || null,
    },
    // Query expansion, if enabled, already ran client-side (generateQueries())
    // and its results are baked into `queries` above -- the backend must not
    // expand again here, or every generated line would itself get expanded.
    llm_expansion: {
      enabled: false,
      variations_per_query: Number($("llm_variations").value),
      llm: {
        provider: $("llm_provider").value,
        base_url: $("llm_base_url").value.trim(),
        model: $("llm_model").value.trim(),
        api_key: $("llm_api_key").value.trim() || null,
        timeout_seconds: Number($("llm_timeout").value) || 180,
        disable_reasoning: $("llm_disable_reasoning").checked,
      },
    },
    captioning: {
      enabled: $("cap_enabled").checked,
      method: $("cap_method").value,
      llm: {
        provider: $("cap_provider").value,
        base_url: $("cap_base_url").value.trim(),
        model: $("cap_model").value.trim(),
        api_key: $("cap_api_key").value.trim() || null,
        timeout_seconds: Number($("cap_timeout").value) || 180,
        disable_reasoning: $("cap_disable_reasoning").checked,
      },
      wd14: buildWD14Config(),
    },
  };
}

// Shared by the per-download captioning config and the standalone
// caption-folder request -- both use the same WD14 fields.
function buildWD14Config() {
  return {
    model: $("cap_wd14_model").value.trim() || "wd-vit-tagger-v3",
    general_threshold: Number($("cap_wd14_general_threshold").value) || 0.35,
    character_threshold: Number($("cap_wd14_character_threshold").value) || 0.85,
  };
}

// ---- job run + websocket progress (shared by the download job and the
// standalone "caption a folder"/"remove duplicates" actions -- only one can
// run at a time) ----
let ws = null;
let currentJobId = null;
let currentJobKind = "download"; // "download" | "caption" | "dedup" | "caption_selected" | "delete_selected"

function setActionButtonsDisabled(disabled) {
  $("start-btn").disabled = disabled;
  $("capfolder-btn").disabled = disabled;
  $("dedup-btn").disabled = disabled;
  $("open-folder-btn").disabled = disabled;
  $("caption-selected-btn").disabled = disabled;
  $("delete-selected-btn").disabled = disabled;
  $("cancel-btn").disabled = !disabled;
}

function applyJobKindLabels(kind) {
  $("stat-downloaded-row").hidden = false;
  if (kind === "caption" || kind === "caption_selected") {
    $("stat-downloaded-label").textContent = "Processed";
    $("stat-duplicates-row").hidden = true;
    $("stat-filtered-row").hidden = true;
  } else if (kind === "dedup") {
    $("stat-downloaded-label").textContent = "Duplicate groups";
    $("stat-duplicates-label").textContent = "Removed";
    $("stat-duplicates-row").hidden = false;
    $("stat-filtered-row").hidden = true;
  } else if (kind === "delete_selected") {
    // Nothing here maps to a "Downloaded"-shaped count -- every selected
    // file is simply removed or not, so that row would only ever read 0.
    $("stat-downloaded-row").hidden = true;
    $("stat-duplicates-label").textContent = "Removed";
    $("stat-duplicates-row").hidden = false;
    $("stat-filtered-row").hidden = true;
  } else {
    $("stat-downloaded-label").textContent = "Downloaded";
    $("stat-duplicates-label").textContent = "Duplicates";
    $("stat-duplicates-row").hidden = false;
    $("stat-filtered-row").hidden = false;
  }
}

function logLine(text, cls = "info") {
  const log = $("log");
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function setStatusBadge(status) {
  const badge = $("job-status");
  badge.textContent = status;
  badge.className = "status-badge " + status;
}

function updateStats(stats) {
  $("stat-downloaded").textContent = stats.downloaded ?? 0;
  $("stat-filtered").textContent = stats.filtered ?? 0;
  $("stat-errors").textContent = stats.errors ?? 0;
  $("stat-duplicates").textContent = stats.duplicates ?? 0;
  $("stat-captioned").textContent = stats.captioned ?? 0;
}

function addExpandedGroup(query, variations) {
  $("expanded-card").hidden = false;
  const list = $("expanded-list");
  const row = document.createElement("div");
  row.className = "expanded-group";
  const chips = variations.map((v) => `<span class="chip">${escapeHtml(v)}</span>`).join("");
  row.innerHTML = `<span class="orig">${escapeHtml(query)} →</span>${chips}`;
  list.appendChild(row);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function handleEvent(ev) {
  switch (ev.type) {
    case "status":
      setStatusBadge(ev.data.status);
      if (ev.data.stats) updateStats(ev.data.stats);
      if (["done", "error", "cancelled"].includes(ev.data.status)) {
        logLine(`Finished. Status: ${ev.data.status}`, "ok");
        setActionButtonsDisabled(false);
        clearTimeout(galleryReloadTimer);
        loadGallery();
        // A successful selection action consumed the selection it acted on
        // -- an error or a cancel leaves it as-is so the same pick can be retried.
        if (ev.data.status === "done" && (currentJobKind === "caption_selected" || currentJobKind === "delete_selected")) {
          clearSelection();
        }
      }
      break;
    case "expanded":
      addExpandedGroup(ev.data.query, ev.data.variations);
      logLine(`LLM: "${ev.data.query}" -> ${ev.data.variations.join(", ")}`, "info");
      break;
    case "downloaded":
      logLine(`OK  [${ev.data.query}] ${ev.data.path}`, "ok");
      updateStats(bumpStats({ downloaded: 1 }));
      scheduleGalleryReload();
      break;
    case "captioned":
      logLine(`CAP ${ev.data.path}: ${ev.data.caption}`, "info");
      updateStats(bumpStats({ captioned: 1 }));
      scheduleGalleryReload();
      break;
    case "skip": {
      // "filtered" = deliberately excluded by our own format/size filters
      // (working as configured); "duplicate" = same content seen before/
      // removed by the dedup job; anything else is a genuine error (network
      // failure, bad data, ...).
      const tag = ev.data.duplicate ? "DUP" : ev.data.filtered ? "FLT" : "ERR";
      const key = ev.data.duplicate ? "duplicates" : ev.data.filtered ? "filtered" : "errors";
      // Duplicates and filtered-out results are expected/working-as-intended
      // outcomes, not failures -- only a real error gets the alarming red.
      logLine(`${tag} [${ev.data.query}] ${ev.data.url} - ${ev.data.error}`, ev.data.duplicate || ev.data.filtered ? "info" : "err");
      updateStats(bumpStats({ [key]: 1 }));
      if (ev.data.duplicate) scheduleGalleryReload();
      break;
    }
    case "warning":
      logLine(`WARN ${ev.data.message}`, "err");
      break;
  }
}

// running counters, since backend "status" events only carry a snapshot at completion
const stats = { downloaded: 0, errors: 0, duplicates: 0, captioned: 0, filtered: 0 };
function bumpStats(delta) {
  for (const key of Object.keys(delta)) stats[key] += delta[key];
  return stats;
}

// Starts a job of either kind against `apiPath` with the given body, and wires
// up progress tracking (WebSocket) the same way for both. Returns nothing --
// errors are surfaced in the log/UI, not thrown.
async function startJob(apiPath, body, kind) {
  currentJobKind = kind;
  applyJobKindLabels(kind);
  setActionButtonsDisabled(true);

  stats.downloaded = stats.errors = stats.duplicates = stats.captioned = stats.filtered = 0;
  updateStats(stats);
  $("log").innerHTML = "";
  $("statusbar").classList.add("expanded");
  setStatusBadge("running");

  let resp;
  try {
    resp = await fetch(apiPath, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    logLine(`Could not reach the server: ${err}`, "err");
    setActionButtonsDisabled(false);
    return;
  }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    logLine(`Failed to start: ${err.detail || resp.statusText}`, "err");
    setActionButtonsDisabled(false);
    return;
  }
  const { job_id } = await resp.json();
  currentJobId = job_id;
  logLine(`Job started: ${job_id}`, "info");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${job_id}`);
  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onerror = () => logLine("WebSocket connection error", "err");
}

document.getElementById("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  saveForm();

  if ($("queries").value.split("\n").map((s) => s.trim()).filter(Boolean).length === 0) {
    alert("Add at least one query");
    return;
  }
  if (!activeFolder) {
    alert("Please open an active folder first (top of the left panel)");
    return;
  }

  // Start download always uses exactly what's in the queries box -- it never
  // expands on its own. If you want LLM-generated variations included, click
  // "Generate variations with LLM" first to update the box, review it, then
  // start. (Auto-expanding here too would double-expand an already-generated
  // list every time you click Start again.)
  if ($("llm_enabled").checked && $("expanded-card").hidden) {
    const proceed = confirm(
      'Query expansion (LLM) is enabled, but you haven\'t clicked "Generate variations with LLM" yet -- ' +
      "this will download using only the queries currently typed in the box, with no LLM variations. " +
      "Continue anyway?"
    );
    if (!proceed) return;
  }

  await startJob("/api/jobs", buildRequest(), "download");
});

// ---- standalone folder captioning (always targets the active folder) ----
function buildCaptionFolderRequest() {
  return {
    folder: activeFolder,
    method: $("cap_method").value,
    recursive: $("capfolder_recursive").checked,
    overwrite: $("capfolder_overwrite").checked,
    llm: {
      provider: $("cap_provider").value,
      base_url: $("cap_base_url").value.trim(),
      model: $("cap_model").value.trim(),
      api_key: $("cap_api_key").value.trim() || null,
      timeout_seconds: Number($("cap_timeout").value) || 180,
      disable_reasoning: $("cap_disable_reasoning").checked,
    },
    wd14: buildWD14Config(),
    trigger: {
      enabled: $("trigger_enabled").checked,
      word: $("trigger_word").value.trim(),
      role: $("trigger_role").value,
      custom_instruction: $("trigger_custom").value.trim() || null,
    },
  };
}

// ---- gallery multi-select captioning: same Method/Settings as above, but
// against exactly the picked files instead of a folder scan -- always
// (re)writes their caption, since picking them *is* the overwrite decision. ----
function buildCaptionFilesRequest(paths) {
  const req = buildCaptionFolderRequest();
  delete req.folder;
  delete req.recursive;
  delete req.overwrite;
  req.paths = paths;
  return req;
}

$("capfolder-btn").addEventListener("click", async () => {
  saveForm();
  const hint = $("capfolder-hint");
  hint.textContent = "";
  hint.className = "hint";

  if (!activeFolder) {
    alert("Please open an active folder first (top of the left panel)");
    return;
  }
  if ($("trigger_enabled").checked && !$("trigger_word").value.trim()) {
    alert('Trigger word is enabled but empty -- fill it in or turn "Use a trigger word" off');
    return;
  }

  await startJob("/api/caption-folder", buildCaptionFolderRequest(), "caption");
});

// ---- standalone duplicate removal (always targets the active folder) ----
function buildDedupFolderRequest() {
  return {
    folder: activeFolder,
    recursive: $("dedup_recursive").checked,
  };
}

$("dedup-btn").addEventListener("click", async () => {
  saveForm();
  const hint = $("dedup-hint");
  hint.textContent = "";
  hint.className = "hint";

  if (!activeFolder) {
    alert("Please open an active folder first (top of the left panel)");
    return;
  }
  // A bulk-delete action deserves an explicit confirmation even though it's
  // recoverable (Recycle Bin, not a permanent delete) -- the exact count
  // isn't known until the scan runs, so this confirms the action itself,
  // not a specific number.
  const proceed = confirm(
    `This will scan "${activeFolder}" for images with identical content and move every copy but one to the Recycle Bin ` +
    "(along with any orphaned .txt caption). This can be undone from the Recycle Bin, but not from here. Continue?"
  );
  if (!proceed) return;

  await startJob("/api/dedup-folder", buildDedupFolderRequest(), "dedup");
});

$("cancel-btn").addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!currentJobId) return;
  await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
  logLine("Cancellation requested...", "info");
  $("cancel-btn").disabled = true;
});
