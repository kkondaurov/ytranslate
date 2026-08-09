const state = {
  manifest: null,
  episode: null,
  variants: [],
  mode: "all",
  search: "",
  activeIndex: -1,
  lastFollowAt: 0,
};

const elements = {
  episodeSelect: document.querySelector("#episode-select"),
  audio: document.querySelector("#audio"),
  timeOutput: document.querySelector("#time-output"),
  jumpForm: document.querySelector("#jump-form"),
  jumpInput: document.querySelector("#jump-input"),
  searchInput: document.querySelector("#search-input"),
  followToggle: document.querySelector("#follow-toggle"),
  comparisonHead: document.querySelector("#comparison-head"),
  comparisonBody: document.querySelector("#comparison-body"),
  scoreStrip: document.querySelector("#score-strip"),
  emptyState: document.querySelector("#empty-state"),
  runState: document.querySelector("#run-state"),
};

const formatTime = (seconds) => {
  const value = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = Math.floor(value % 60);
  return [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
};

const parseTime = (value) => {
  const parts = String(value).trim().split(":").map(Number);
  if (!parts.length || parts.some((part) => !Number.isFinite(part) || part < 0)) return null;
  return parts.reduce((total, part) => total * 60 + part, 0);
};

const canonicalSpeaker = (label) => {
  const value = String(label || "").toLowerCase();
  if (/friedberg|freeberg|freiberg/.test(value)) return "friedberg";
  if (/sacks|sachs|zach/.test(value)) return "sacks";
  if (/chamath|chumath|jamath/.test(value)) return "chamath";
  if (/jason|j.?cal/.test(value)) return "jason";
  if (/brad|gerstner/.test(value)) return "brad";
  if (/gavin|baker/.test(value)) return "gavin";
  return "unknown";
};

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const variantUrl = (path) => `../output/${path}`;

async function loadManifest() {
  const response = await fetch("../output/manifest.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
  state.manifest = await response.json();
  elements.episodeSelect.innerHTML = state.manifest.episodes
    .map((episode) => `<option value="${escapeHtml(episode.key)}">${escapeHtml(episode.title)}</option>`)
    .join("");
  await loadEpisode(state.manifest.episodes[0].key);
}

async function loadEpisode(key) {
  const episode = state.manifest.episodes.find((item) => item.key === key);
  if (!episode) return;
  elements.runState.textContent = "Loading episode";
  const variants = await Promise.all(episode.variants.map(async (entry) => {
    const response = await fetch(variantUrl(entry.path), { cache: "no-store" });
    if (!response.ok) throw new Error(`${entry.label} request failed: ${response.status}`);
    return { entry, result: await response.json() };
  }));
  state.episode = episode;
  state.variants = variants;
  state.activeIndex = -1;
  elements.episodeSelect.value = key;
  elements.audio.src = variantUrl(episode.audio);
  renderScoreStrip();
  renderHead();
  renderRows();
  elements.runState.textContent = `${episode.variants.length} variants · ${variants[0].result.segments.length.toLocaleString()} aligned segments`;
}

function metricPills(entry) {
  const metrics = entry.metrics || {};
  const pills = [];
  const comparableAudit = metrics.known_host_audit || metrics.audit;
  if (comparableAudit) {
    const value = comparableAudit.duration_accuracy;
    const label = metrics.known_host_audit ? "Known-host audit" : "Audited";
    pills.push({ text: `${label} ${Math.round(value * 1000) / 10}%`, tone: value >= 0.9 ? "good" : "warn" });
  }
  if (metrics.windows) {
    const value = metrics.windows.passed / Math.max(1, metrics.windows.total);
    pills.push({ text: `Audited windows ${metrics.windows.passed}/${metrics.windows.total}`, tone: value === 1 ? "good" : "warn" });
  }
  if (metrics.boundaries) {
    pills.push({
      text: `Boundaries ${Math.round(metrics.boundaries.confident_accuracy * 1000) / 10}% @ ${Math.round(metrics.boundaries.coverage * 100)}%`,
      tone: metrics.boundaries.confident_accuracy >= 0.9 ? "good" : "warn",
    });
  }
  if (Number.isFinite(entry.disagreement_with_current)) {
    pills.push({ text: `Differs ${Math.round(entry.disagreement_with_current * 1000) / 10}%`, tone: "" });
  }
  return pills;
}

function renderScoreStrip() {
  elements.scoreStrip.innerHTML = state.episode.variants.map((entry) => `
    <article class="score-card">
      <h2>${escapeHtml(entry.label)}</h2>
      <div class="score-list">
        ${metricPills(entry).map((pill) => `<span class="score-pill ${pill.tone}">${escapeHtml(pill.text)}</span>`).join("") || '<span class="score-pill">Manual review</span>'}
      </div>
    </article>
  `).join("");
}

function renderHead() {
  elements.comparisonHead.innerHTML = `
    <div class="head-cell">Time</div>
    ${state.episode.variants.map((entry, index) => `
      <div class="head-cell">
        ${escapeHtml(entry.label)}
        <small>${index === 0 ? "Control" : index === 2 ? "Primary challenger" : "Reasoning sensitivity"}</small>
      </div>
    `).join("")}
  `;
}

function rowDisagrees(index) {
  const speakers = state.variants.map(({ result }) => canonicalSpeaker(result.segments[index]?.speaker_label));
  return new Set(speakers).size > 1;
}

function rowMatchesSearch(index) {
  if (!state.search) return true;
  const term = state.search.toLowerCase();
  return state.variants.some(({ result }) => {
    const segment = result.segments[index] || {};
    return `${segment.speaker_label || ""} ${segment.text || ""}`.toLowerCase().includes(term);
  });
}

function renderRows() {
  const baselineSegments = state.variants[0].result.segments;
  const rows = [];
  for (let index = 0; index < baselineSegments.length; index += 1) {
    const disagrees = rowDisagrees(index);
    if (state.mode === "disagreements" && !disagrees) continue;
    if (!rowMatchesSearch(index)) continue;
    const baseline = baselineSegments[index];
    const cells = state.variants.map(({ result }) => {
      const segment = result.segments[index] || {};
      const speaker = segment.speaker_label || segment.speaker_id || "Unknown";
      const speakerClass = canonicalSpeaker(speaker);
      const boundary = segment.boundary_before
        ? `<span class="boundary-mark">${escapeHtml(segment.boundary_before)}</span>`
        : "";
      return `
        <div class="transcript-cell" data-seek="${Number(baseline.start || 0)}">
          <span class="speaker-label speaker-${speakerClass}">${escapeHtml(speaker)}</span>
          ${boundary}
          <div class="transcript-text">${escapeHtml(segment.text || "")}</div>
        </div>
      `;
    }).join("");
    rows.push(`
      <div class="comparison-grid comparison-row ${disagrees ? "disagrees" : ""}" data-index="${index}" data-start="${Number(baseline.start || 0)}" data-end="${Number(baseline.end || baseline.start || 0)}">
        <div class="time-cell">${formatTime(Number(baseline.start || 0))}<br>${formatTime(Number(baseline.end || baseline.start || 0))}</div>
        ${cells}
      </div>
    `);
  }
  elements.comparisonBody.innerHTML = rows.join("");
  elements.emptyState.hidden = rows.length > 0;
  updateActiveRow(true, false);
}

function findSegmentIndex(time) {
  const segments = state.variants[0]?.result.segments || [];
  let low = 0;
  let high = segments.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const segment = segments[middle];
    if (time < Number(segment.start || 0)) high = middle - 1;
    else if (time > Number(segment.end || segment.start || 0)) low = middle + 1;
    else return middle;
  }
  return Math.max(0, Math.min(segments.length - 1, high));
}

function updateActiveRow(force = false, allowFollow = true) {
  if (!state.variants.length) return;
  const index = findSegmentIndex(elements.audio.currentTime || 0);
  if (!force && index === state.activeIndex) return;
  document.querySelector(".comparison-row.active")?.classList.remove("active");
  state.activeIndex = index;
  const row = document.querySelector(`.comparison-row[data-index="${index}"]`);
  if (!row) return;
  row.classList.add("active");
  const now = performance.now();
  if (allowFollow && elements.followToggle.checked && (force || now - state.lastFollowAt > 700)) {
    row.scrollIntoView({ behavior: force ? "auto" : "smooth", block: "center" });
    state.lastFollowAt = now;
  }
}

elements.episodeSelect.addEventListener("change", (event) => loadEpisode(event.target.value));
elements.searchInput.addEventListener("input", (event) => {
  state.search = event.target.value.trim();
  renderRows();
});
document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".mode-button").forEach((item) => item.classList.toggle("active", item === button));
    renderRows();
  });
});
elements.comparisonBody.addEventListener("click", (event) => {
  const cell = event.target.closest("[data-seek]");
  if (!cell) return;
  elements.audio.currentTime = Number(cell.dataset.seek || 0);
  updateActiveRow(true);
});
elements.audio.addEventListener("timeupdate", () => {
  elements.timeOutput.textContent = `${formatTime(elements.audio.currentTime)} / ${formatTime(elements.audio.duration)}`;
  updateActiveRow();
});
elements.audio.addEventListener("loadedmetadata", () => {
  elements.timeOutput.textContent = `${formatTime(elements.audio.currentTime)} / ${formatTime(elements.audio.duration)}`;
});
document.querySelector("#back-button").addEventListener("click", () => {
  elements.audio.currentTime = Math.max(0, elements.audio.currentTime - 10);
});
document.querySelector("#forward-button").addEventListener("click", () => {
  elements.audio.currentTime = Math.min(elements.audio.duration || Infinity, elements.audio.currentTime + 10);
});
elements.jumpForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const time = parseTime(elements.jumpInput.value);
  if (time === null) return;
  elements.audio.currentTime = Math.min(time, elements.audio.duration || time);
  updateActiveRow(true);
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select")) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (elements.audio.paused) elements.audio.play();
    else elements.audio.pause();
  }
  if (event.code === "ArrowLeft") elements.audio.currentTime = Math.max(0, elements.audio.currentTime - 5);
  if (event.code === "ArrowRight") elements.audio.currentTime += 5;
});

loadManifest().catch((error) => {
  console.error(error);
  elements.runState.textContent = error.message;
});
