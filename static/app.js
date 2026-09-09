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

function wireToggle(checkbox, block) {
  const sync = () => block.classList.toggle("disabled", !checkbox.checked);
  checkbox.addEventListener("change", sync);
  sync();
}
wireToggle($("llm_enabled"), $("llm_block"));
wireToggle($("cap_enabled"), $("cap_block"));
wireToggle($("trigger_enabled"), $("trigger_block"));

function syncGenerateBtn() {
  $("generate-btn").disabled = !$("llm_enabled").checked;
}
$("llm_enabled").addEventListener("change", syncGenerateBtn);

function syncTriggerCustomField() {
  $("trigger_custom_field").hidden = $("trigger_role").value !== "custom";
}
$("trigger_role").addEventListener("change", syncTriggerCustomField);
syncTriggerCustomField();

// ---- native folder picker (server-side dialog; browser & server must be on the same machine) ----
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
  btn.textContent = "Refreshing...";
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
      hint.textContent = `Found: ${parts.join(", ")}. Pick one from the dropdown below the field.`;
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
    btn.textContent = "↻ Refresh";
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
  "queries", "n_per_query", "concurrency", "output_folder", "query_subfolders",
  "min_width", "min_height", "safesearch",
  "llm_enabled", "llm_provider", "llm_variations", "llm_base_url", "llm_model", "llm_api_key", "llm_timeout", "llm_models_folder", "llm_disable_reasoning",
  "cap_enabled", "cap_provider", "cap_model", "cap_base_url", "cap_api_key", "cap_timeout", "cap_models_folder", "cap_disable_reasoning",
  "capfolder_path", "capfolder_recursive", "capfolder_overwrite",
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
  $("cap_block").classList.toggle("disabled", !$("cap_enabled").checked);
  $("trigger_block").classList.toggle("disabled", !$("trigger_enabled").checked);
}
loadForm();
syncGenerateBtn();
syncTriggerCustomField();
document.getElementById("job-form").addEventListener("change", saveForm);

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
    output_folder: $("output_folder").value.trim(),
    query_subfolders: $("query_subfolders").checked,
    concurrency: Number($("concurrency").value),
    filters: {
      formats,
      min_width: Number($("min_width").value),
      min_height: Number($("min_height").value),
      safesearch: $("safesearch").value,
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
      llm: {
        provider: $("cap_provider").value,
        base_url: $("cap_base_url").value.trim(),
        model: $("cap_model").value.trim(),
        api_key: $("cap_api_key").value.trim() || null,
        timeout_seconds: Number($("cap_timeout").value) || 180,
        disable_reasoning: $("cap_disable_reasoning").checked,
      },
    },
  };
}

// ---- job run + websocket progress (shared by the download job and the
// standalone "caption a folder" action -- only one can run at a time) ----
let ws = null;
let currentJobId = null;
let currentJobKind = "download"; // "download" | "caption"

function setActionButtonsDisabled(disabled) {
  $("start-btn").disabled = disabled;
  $("capfolder-btn").disabled = disabled;
  $("cancel-btn").disabled = !disabled;
}

function applyJobKindLabels(kind) {
  if (kind === "caption") {
    $("stat-downloaded-label").textContent = "Processed";
    $("stat-duplicates-row").hidden = true;
    $("stat-filtered-row").hidden = true;
    $("thumbs-title").textContent = "Captioned images";
  } else {
    $("stat-downloaded-label").textContent = "Downloaded";
    $("stat-duplicates-row").hidden = false;
    $("stat-filtered-row").hidden = false;
    $("thumbs-title").textContent = "Downloaded images";
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

// index (as assigned by the backend, matching /api/jobs/{id}/image/{index}) -> caption <div>
const thumbCaptionEls = {};

function addThumbnail(index, query) {
  $("thumbs-card").hidden = false;
  const grid = $("thumb-grid");
  const card = document.createElement("div");
  card.className = "thumb";

  const img = document.createElement("img");
  img.src = `/api/jobs/${currentJobId}/image/${index}`;
  img.loading = "lazy";
  img.alt = query || "";
  card.appendChild(img);

  const queryEl = document.createElement("div");
  queryEl.className = "query";
  queryEl.textContent = query || "";
  card.appendChild(queryEl);

  const capEl = document.createElement("div");
  capEl.className = "cap";
  card.appendChild(capEl);

  grid.appendChild(card);
  thumbCaptionEls[index] = capEl;
}

function handleEvent(ev) {
  switch (ev.type) {
    case "status":
      setStatusBadge(ev.data.status);
      if (ev.data.stats) updateStats(ev.data.stats);
      if (["done", "error", "cancelled"].includes(ev.data.status)) {
        logLine(`Finished. Status: ${ev.data.status}`, "ok");
        setActionButtonsDisabled(false);
      }
      break;
    case "expanded":
      addExpandedGroup(ev.data.query, ev.data.variations);
      logLine(`LLM: "${ev.data.query}" -> ${ev.data.variations.join(", ")}`, "info");
      break;
    case "downloaded":
      logLine(`OK  [${ev.data.query}] ${ev.data.path}`, "ok");
      updateStats(bumpStats({ downloaded: 1 }));
      if (ev.data.index !== undefined) addThumbnail(ev.data.index, ev.data.query);
      break;
    case "captioned":
      logLine(`CAP ${ev.data.path}: ${ev.data.caption}`, "info");
      updateStats(bumpStats({ captioned: 1 }));
      if (ev.data.index !== undefined && thumbCaptionEls[ev.data.index]) {
        thumbCaptionEls[ev.data.index].textContent = ev.data.caption;
      }
      break;
    case "skip": {
      // "filtered" = deliberately excluded by our own format/size filters
      // (working as configured); "duplicate" = same content seen before;
      // anything else is a genuine error (network failure, bad data, ...).
      const tag = ev.data.duplicate ? "DUP" : ev.data.filtered ? "FLT" : "ERR";
      const key = ev.data.duplicate ? "duplicates" : ev.data.filtered ? "filtered" : "errors";
      logLine(`${tag} [${ev.data.query}] ${ev.data.url} - ${ev.data.error}`, ev.data.filtered ? "info" : "err");
      updateStats(bumpStats({ [key]: 1 }));
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
  $("thumb-grid").innerHTML = "";
  $("thumbs-card").hidden = true;
  for (const k of Object.keys(thumbCaptionEls)) delete thumbCaptionEls[k];
  $("progress-card").hidden = false;
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
  if (!$("output_folder").value.trim()) {
    alert("Please set an output folder");
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

// ---- standalone folder captioning ----
function buildCaptionFolderRequest() {
  return {
    folder: $("capfolder_path").value.trim(),
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
    trigger: {
      enabled: $("trigger_enabled").checked,
      word: $("trigger_word").value.trim(),
      role: $("trigger_role").value,
      custom_instruction: $("trigger_custom").value.trim() || null,
    },
  };
}

$("capfolder-btn").addEventListener("click", async () => {
  saveForm();
  const hint = $("capfolder-hint");
  hint.textContent = "";
  hint.className = "hint";

  if (!$("capfolder_path").value.trim()) {
    alert("Please set a folder to caption");
    return;
  }
  if ($("trigger_enabled").checked && !$("trigger_word").value.trim()) {
    alert('Trigger word is enabled but empty -- fill it in or turn "Use a trigger word" off');
    return;
  }

  await startJob("/api/caption-folder", buildCaptionFolderRequest(), "caption");
});

document.getElementById("cancel-btn").addEventListener("click", async () => {
  if (!currentJobId) return;
  await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
  logLine("Cancellation requested...", "info");
  $("cancel-btn").disabled = true;
});
