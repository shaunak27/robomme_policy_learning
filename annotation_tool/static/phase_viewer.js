// =========================================================================
// Phase Label Viewer — Interactive pseudo-label inspector
// =========================================================================

const CELL_W = 56;
const TL_H   = 88;
const THUMB_SRC = 64;

const PHASE_COLORS = {
  precondition:          { bg: "#3498db", fg: "#fff" },
  contact_or_transition: { bg: "#f39c12", fg: "#fff" },
  postcondition_success: { bg: "#2ecc71", fg: "#fff" },
  failed_attempt:        { bg: "#e74c3c", fg: "#fff" },
  unknown:               { bg: "#555",    fg: "#aaa" },
};
const PHASE_LABELS_NICE = {
  precondition:          "Precondition",
  contact_or_transition: "Contact / Transition",
  postcondition_success: "Postcondition (Success)",
  failed_attempt:        "Failed Attempt",
  unknown:               "Unknown",
};

// Only 3 CLIP-scored phases (failed_attempt is temporal-only)
const CLIP_SCORE_COLORS = ["#3498db", "#f39c12", "#2ecc71"];
const CLIP_SCORE_KEYS   = ["precondition", "contact_or_transition", "postcondition_success"];

// Attempt score component colors
const ATTEMPT_COLORS = {
  attempt_score:  "#ffffff",
  clip_contact:   "#f39c12",
  visual_change:  "#8e44ad",
  state_change:   "#1abc9c",
  gripper_change: "#e67e22",
};

// ---- State ----
let allSegments = [];
let filteredSegments = [];
let currentIdx = 0;
let labelData = null;       // frame_labels.json
let scoreData = null;       // clip scores
let attemptData = null;     // attempt scores
let candidateData = null;   // candidate windows
let spriteImg = null;
let currentFrame = 0;
let animId = null;
let activeChart = "clip";   // "clip" or "attempt"

// ---- DOM ----
const $ = s => document.querySelector(s);
const envFilter      = $("#env-filter");
const segCounter     = $("#seg-counter");
const segListScroll  = $("#segment-list-scroll");
const segFilterInput = $("#segment-filter");
const video          = $("#video-player");
const videoBorder    = $("#video-border");
const frameCounter   = $("#frame-counter");
const btnPlayPause   = $("#btn-play-pause");
const btnReset       = $("#btn-reset");
const speedSlider    = $("#speed-slider-2");
const frontFrame     = $("#front-frame");
const wristFrame     = $("#wrist-frame");
const subtaskDisplay = $("#subtask-display");
const goalDisplay    = $("#task-goal-display");
const keyFramesRow   = $("#key-frames-row");
const labelDistBar   = $("#label-dist-bar");
const phaseBadge     = $("#phase-badge-current");
const phaseLegend    = $("#phase-legend");
const scoreCanvas    = $("#score-canvas");
const scoreCtx       = scoreCanvas.getContext("2d");
const tlScroll       = $("#timeline-scroll");
const tlCanvas       = $("#timeline-canvas");
const tlCtx          = tlCanvas.getContext("2d");

// =========================================================================
// Init
// =========================================================================
async function init() {
  allSegments = await fetchJSON("/api/segments");
  if (!allSegments.length) {
    segListScroll.innerHTML = '<div style="padding:20px;color:#666;">No segments found. Run the pseudo-labeling pipeline first.</div>';
    return;
  }

  // Populate env filter
  const envs = [...new Set(allSegments.map(s => s.env_id))].sort();
  envFilter.innerHTML = '<option value="">All Tasks</option>' +
    envs.map(e => `<option value="${e}">${e}</option>`).join("");
  envFilter.addEventListener("change", applyFilters);
  segFilterInput.addEventListener("input", applyFilters);

  $("#btn-prev").addEventListener("click", () => navigate(-1));
  $("#btn-next").addEventListener("click", () => navigate(1));
  btnPlayPause.addEventListener("click", togglePlay);
  btnReset.addEventListener("click", resetVideo);
  speedSlider.addEventListener("input", updateSpeed);
  // Sync both speed sliders
  $("#speed-slider").addEventListener("input", e => {
    speedSlider.value = e.target.value;
    updateSpeed();
  });

  tlCanvas.addEventListener("click", onTimelineClick);

  // Chart toggle
  scoreCanvas.addEventListener("click", () => {
    activeChart = activeChart === "clip" ? "attempt" : "clip";
    drawScoreChart();
    buildLegend();
  });

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") { e.preventDefault(); navigate(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); navigate(-1); }
    if (e.key === "r" || e.key === "R") resetVideo();
    if (e.key === "t" || e.key === "T") {
      activeChart = activeChart === "clip" ? "attempt" : "clip";
      drawScoreChart(); buildLegend();
    }
  });

  video.addEventListener("timeupdate", onVideoTime);
  startPlayheadLoop();
  buildLegend();

  applyFilters();
  await loadSegment(0);
}

// =========================================================================
// Filtering
// =========================================================================
function applyFilters() {
  const envVal = envFilter.value;
  const textVal = segFilterInput.value.toLowerCase().trim();
  filteredSegments = allSegments.filter(s => {
    if (envVal && s.env_id !== envVal) return false;
    if (textVal && !s.subtask_label.toLowerCase().includes(textVal) &&
        !s.env_id.toLowerCase().includes(textVal)) return false;
    return true;
  });
  currentIdx = 0;
  renderSegmentList();
  if (filteredSegments.length) loadSegment(0);
}

function renderSegmentList() {
  segListScroll.innerHTML = "";
  for (let i = 0; i < filteredSegments.length; i++) {
    const s = filteredSegments[i];
    const srcBadge = s.label_source === "mllm_verified"
      ? '<span style="color:#2ecc71;font-size:9px;margin-left:4px;">MLLM</span>'
      : '<span style="color:#888;font-size:9px;margin-left:4px;">CLIP</span>';
    const div = document.createElement("div");
    div.className = "seg-item" + (i === currentIdx ? " active" : "");
    div.dataset.idx = i;
    div.innerHTML = `
      <div class="seg-item-env">${s.env_id} &middot; ep${s.episode_idx}${srcBadge}</div>
      <div class="seg-item-label">${s.subtask_label}</div>
      <div class="seg-item-frames">frames ${s.start_frame}&ndash;${s.end_frame} (${s.num_frames})</div>
    `;
    div.addEventListener("click", () => loadSegment(i));
    segListScroll.appendChild(div);
  }
}

// =========================================================================
// Loading
// =========================================================================
async function loadSegment(idx) {
  if (idx < 0 || idx >= filteredSegments.length) return;
  currentIdx = idx;
  const seg = filteredSegments[idx];

  // Highlight in list
  segListScroll.querySelectorAll(".seg-item").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });
  const activeEl = segListScroll.querySelector(".seg-item.active");
  if (activeEl) activeEl.scrollIntoView({ block: "nearest" });

  segCounter.textContent = `${idx + 1} / ${filteredSegments.length}`;
  subtaskDisplay.textContent = `"${seg.subtask_label}"`;
  goalDisplay.textContent = seg.task_goal;

  // Key frames
  let kfHtml = "";
  if (seg.transition_start != null) {
    kfHtml += `<div class="kf-item"><span class="kf-dot" style="background:#e67e22;"></span>Transition start: frame ${seg.transition_start}</div>`;
  }
  if (seg.transition_frame != null) {
    kfHtml += `<div class="kf-item"><span class="kf-dot success"></span>Transition peak: frame ${seg.transition_frame}</div>`;
  }
  if (seg.completion_start != null) {
    kfHtml += `<div class="kf-item"><span class="kf-dot completion"></span>Completion: frame ${seg.completion_start}</div>`;
  }
  if (seg.num_failed_attempts > 0) {
    kfHtml += `<div class="kf-item"><span class="kf-dot fail"></span>Failed attempts: ${seg.num_failed_attempts}</div>`;
  }
  const srcLabel = seg.label_source === "mllm_verified" ? "MLLM-verified" : "CLIP heuristic";
  kfHtml += `<div class="kf-item" style="color:#888;font-size:11px;">Source: ${srcLabel}</div>`;
  keyFramesRow.innerHTML = kfHtml;

  // Label distribution bar
  const lc = seg.label_counts || {};
  const total = Object.values(lc).reduce((a, b) => a + b, 0) || 1;
  labelDistBar.innerHTML = Object.entries(lc).map(([label, count]) => {
    const pct = (count / total * 100).toFixed(1);
    const c = PHASE_COLORS[label] || PHASE_COLORS.unknown;
    return `<div class="dist-segment" style="width:${pct}%;background:${c.bg};" title="${PHASE_LABELS_NICE[label] || label}: ${count} frames (${pct}%)"></div>`;
  }).join("");

  // Load data in parallel
  const [labels, scores, attempt, candidates] = await Promise.all([
    fetchJSON(`/api/segment/${seg.seg_id}/labels`),
    fetchJSON(`/api/segment/${seg.seg_id}/scores`),
    fetchJSON(`/api/segment/${seg.seg_id}/attempt_scores`),
    fetchJSON(`/api/segment/${seg.seg_id}/candidates`),
  ]);
  labelData = labels;
  scoreData = scores;
  attemptData = attempt;
  candidateData = candidates;

  // Video
  video.pause();
  video.src = `/api/segment/${seg.seg_id}/video`;
  video.playbackRate = parseFloat(speedSlider.value);
  video.load();
  currentFrame = 0;
  btnPlayPause.textContent = "Play";

  // Thumbnail strip
  spriteImg = new Image();
  spriteImg.src = `/api/segment/${seg.seg_id}/thumbstrip`;
  await new Promise(r => { spriteImg.onload = r; spriteImg.onerror = r; });

  // Size timeline canvas
  const n = labelData.frames.length;
  tlCanvas.width = n * CELL_W;
  tlCanvas.height = TL_H;
  tlCanvas.style.width = n * CELL_W + "px";
  tlCanvas.style.height = TL_H + "px";

  drawTimeline();
  drawScoreChart();
  buildLegend();
  updateFrameInfo();
}

// =========================================================================
// Video
// =========================================================================
function togglePlay() {
  if (video.paused) { video.play(); btnPlayPause.textContent = "Pause"; }
  else { video.pause(); btnPlayPause.textContent = "Play"; }
}
function resetVideo() {
  video.pause(); video.currentTime = 0; currentFrame = 0;
  btnPlayPause.textContent = "Play";
  tlScroll.scrollLeft = 0;
  updateFrameInfo(); drawTimeline(); drawScoreChart();
}
function updateSpeed() {
  video.playbackRate = parseFloat(speedSlider.value);
  $("#speed-label").textContent = parseFloat(speedSlider.value).toFixed(1) + "x";
}
function onVideoTime() {
  const seg = filteredSegments[currentIdx];
  if (!seg || !labelData) return;
  const n = labelData.frames.length;
  const ratio = video.duration ? video.currentTime / video.duration : 0;
  currentFrame = Math.min(Math.floor(ratio * n), n - 1);
  updateFrameInfo(); drawTimeline(); drawScoreChart();
}
function startPlayheadLoop() {
  function tick() {
    if (!video.paused) onVideoTime();
    animId = requestAnimationFrame(tick);
  }
  tick();
}

function navigate(delta) {
  loadSegment(currentIdx + delta);
}

function timeFromFrame(f) {
  if (!labelData || !video.duration) return 0;
  return (f / labelData.frames.length) * video.duration;
}

// =========================================================================
// Frame info (camera views + badge)
// =========================================================================
function updateFrameInfo() {
  if (!labelData || !labelData.frames.length) return;
  const seg = filteredSegments[currentIdx];
  const f = Math.min(currentFrame, labelData.frames.length - 1);
  const frameInfo = labelData.frames[f];
  const globalIdx = frameInfo.frame;

  frameCounter.textContent = `Frame ${globalIdx} (${f + 1}/${labelData.frames.length})`;

  // Phase badge
  const label = frameInfo.label;
  const c = PHASE_COLORS[label] || PHASE_COLORS.unknown;
  phaseBadge.textContent = PHASE_LABELS_NICE[label] || label;
  phaseBadge.className = "phase-badge phase-" + label;

  // Camera views
  frontFrame.src = `/api/segment/${seg.seg_id}/frame/${globalIdx}`;
  wristFrame.src = `/api/segment/${seg.seg_id}/wrist_frame/${globalIdx}`;

  // Auto-scroll timeline
  const playheadX = f * CELL_W + CELL_W / 2;
  const viewW = tlScroll.clientWidth;
  const margin = viewW * 0.25;
  if (playheadX < tlScroll.scrollLeft + margin) {
    tlScroll.scrollLeft = playheadX - margin;
  } else if (playheadX > tlScroll.scrollLeft + viewW - margin) {
    tlScroll.scrollLeft = playheadX - viewW + margin;
  }
}

// =========================================================================
// Timeline
// =========================================================================
function drawTimeline() {
  if (!labelData || !labelData.frames.length) return;
  const frames = labelData.frames;
  const n = frames.length;
  const W = tlCanvas.width;
  const H = TL_H;
  const PHASE_BAR_H = 14;
  const THUMB_Y = PHASE_BAR_H + 1;
  const THUMB_SIZE = CELL_W - 4;

  tlCtx.clearRect(0, 0, W, H);

  // Phase color bar at top
  for (let i = 0; i < n; i++) {
    const label = frames[i].label;
    const c = PHASE_COLORS[label] || PHASE_COLORS.unknown;
    tlCtx.fillStyle = c.bg;
    tlCtx.fillRect(i * CELL_W, 0, CELL_W, PHASE_BAR_H);
  }

  // Candidate window shading on phase bar
  if (candidateData && candidateData.length) {
    for (const cand of candidateData) {
      const startLocal = cand.start_local;
      const endLocal = cand.end_local;
      for (let i = startLocal; i <= endLocal && i < n; i++) {
        tlCtx.fillStyle = "rgba(255,255,255,0.2)";
        tlCtx.fillRect(i * CELL_W, 0, CELL_W, PHASE_BAR_H);
      }
    }
  }

  // Thumbnails
  for (let i = 0; i < n; i++) {
    const x = i * CELL_W;
    const label = frames[i].label;
    const c = PHASE_COLORS[label] || PHASE_COLORS.unknown;
    tlCtx.fillStyle = c.bg + "18";
    tlCtx.fillRect(x, THUMB_Y, CELL_W, THUMB_SIZE);

    if (spriteImg && spriteImg.complete && spriteImg.naturalWidth > 0) {
      tlCtx.drawImage(spriteImg, i * THUMB_SRC, 0, THUMB_SRC, THUMB_SRC,
                       x + 2, THUMB_Y, THUMB_SIZE, THUMB_SIZE);
    }
    tlCtx.strokeStyle = c.bg + "60";
    tlCtx.lineWidth = 1;
    tlCtx.strokeRect(x + 1, THUMB_Y, CELL_W - 2, THUMB_SIZE);
  }

  // Transition start marker
  const seg = filteredSegments[currentIdx];
  if (seg && seg.transition_start != null) {
    const localIdx = frames.findIndex(f => f.frame === seg.transition_start);
    if (localIdx >= 0) {
      const sx = localIdx * CELL_W + CELL_W / 2;
      tlCtx.strokeStyle = "#e67e22";
      tlCtx.lineWidth = 2;
      tlCtx.setLineDash([3, 2]);
      tlCtx.beginPath();
      tlCtx.moveTo(sx, 0);
      tlCtx.lineTo(sx, H);
      tlCtx.stroke();
      tlCtx.setLineDash([]);
      tlCtx.fillStyle = "#e67e22";
      tlCtx.font = "bold 8px sans-serif";
      tlCtx.textAlign = "center";
      tlCtx.fillText("T-START", sx, H - 10);
    }
  }

  // Transition peak marker
  if (seg && seg.transition_frame != null) {
    const localIdx = frames.findIndex(f => f.frame === seg.transition_frame);
    if (localIdx >= 0) {
      const sx = localIdx * CELL_W + CELL_W / 2;
      tlCtx.strokeStyle = "#2ecc71";
      tlCtx.lineWidth = 2.5;
      tlCtx.setLineDash([4, 2]);
      tlCtx.beginPath();
      tlCtx.moveTo(sx, 0);
      tlCtx.lineTo(sx, H);
      tlCtx.stroke();
      tlCtx.setLineDash([]);
      tlCtx.fillStyle = "#2ecc71";
      tlCtx.font = "bold 9px sans-serif";
      tlCtx.textAlign = "center";
      tlCtx.fillText("T-PEAK", sx, H - 2);
    }
  }

  // Completion start marker
  if (seg && seg.completion_start != null) {
    const localIdx = frames.findIndex(f => f.frame === seg.completion_start);
    if (localIdx >= 0) {
      const cx = localIdx * CELL_W + CELL_W / 2;
      tlCtx.strokeStyle = "#27ae60";
      tlCtx.lineWidth = 1.5;
      tlCtx.setLineDash([2, 2]);
      tlCtx.beginPath();
      tlCtx.moveTo(cx, 0);
      tlCtx.lineTo(cx, H);
      tlCtx.stroke();
      tlCtx.setLineDash([]);
    }
  }

  // Failed attempt markers
  if (seg && seg.failed_attempt_frames) {
    for (const ff of seg.failed_attempt_frames) {
      const localIdx = frames.findIndex(f => f.frame === ff);
      if (localIdx >= 0) {
        const fx = localIdx * CELL_W + CELL_W / 2;
        tlCtx.strokeStyle = "#e74c3c";
        tlCtx.lineWidth = 2;
        tlCtx.setLineDash([3, 3]);
        tlCtx.beginPath();
        tlCtx.moveTo(fx, 0);
        tlCtx.lineTo(fx, H);
        tlCtx.stroke();
        tlCtx.setLineDash([]);
      }
    }
  }

  // Playhead
  const px = currentFrame * CELL_W + CELL_W / 2;
  tlCtx.strokeStyle = "rgba(255,255,255,0.9)";
  tlCtx.lineWidth = 2.5;
  tlCtx.beginPath();
  tlCtx.moveTo(px, 0);
  tlCtx.lineTo(px, H);
  tlCtx.stroke();
  tlCtx.fillStyle = "#fff";
  tlCtx.beginPath();
  tlCtx.moveTo(px - 5, 0);
  tlCtx.lineTo(px + 5, 0);
  tlCtx.lineTo(px, 7);
  tlCtx.closePath();
  tlCtx.fill();

  // Frame numbers
  tlCtx.fillStyle = "rgba(255,255,255,0.35)";
  tlCtx.font = "8px monospace";
  tlCtx.textAlign = "center";
  tlCtx.textBaseline = "bottom";
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 20))) {
    tlCtx.fillText(String(frames[i].frame), i * CELL_W + CELL_W / 2, H - 1);
  }
}

function onTimelineClick(e) {
  if (!labelData) return;
  const rect = tlCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const f = Math.max(0, Math.min(labelData.frames.length - 1, Math.floor(x / CELL_W)));
  currentFrame = f;
  video.currentTime = timeFromFrame(f);
  updateFrameInfo();
  drawTimeline();
  drawScoreChart();
}

// =========================================================================
// Score chart (drawn on canvas) — toggles between CLIP and Attempt views
// =========================================================================
function drawScoreChart() {
  if (activeChart === "attempt" && attemptData && attemptData.attempt_score) {
    drawAttemptChart();
  } else {
    drawClipChart();
  }
}

function drawClipChart() {
  if (!scoreData || !scoreData.frame_indices) return;

  const canvas = scoreCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width;
  const H = Math.max(140, rect.height - 30);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  scoreCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const n = scoreData.frame_indices.length;
  const PAD_L = 40, PAD_R = 10, PAD_T = 18, PAD_B = 24;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // Background
  scoreCtx.fillStyle = "#0d0d1a";
  scoreCtx.fillRect(0, 0, W, H);

  // Title
  scoreCtx.fillStyle = "#888";
  scoreCtx.font = "bold 10px sans-serif";
  scoreCtx.textAlign = "left";
  scoreCtx.fillText("CLIP Scores (3 phases) — click or press T to toggle", PAD_L, 12);

  // Find global min/max across 3 CLIP score arrays
  let gmin = Infinity, gmax = -Infinity;
  for (const key of CLIP_SCORE_KEYS) {
    const arr = scoreData[key];
    if (!arr) continue;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < gmin) gmin = arr[i];
      if (arr[i] > gmax) gmax = arr[i];
    }
  }
  const range = gmax - gmin || 1;

  // Grid lines
  drawGrid(W, H, PAD_L, PAD_R, PAD_T, PAD_B, plotW, plotH, gmin, gmax, range);

  // Draw 3 CLIP score lines
  for (let s = 0; s < CLIP_SCORE_KEYS.length; s++) {
    const arr = scoreData[CLIP_SCORE_KEYS[s]];
    if (!arr) continue;
    scoreCtx.strokeStyle = CLIP_SCORE_COLORS[s];
    scoreCtx.lineWidth = 1.8;
    scoreCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = PAD_L + (i / (n - 1)) * plotW;
      const y = PAD_T + plotH - ((arr[i] - gmin) / range) * plotH;
      if (i === 0) scoreCtx.moveTo(x, y);
      else scoreCtx.lineTo(x, y);
    }
    scoreCtx.stroke();
  }

  // Candidate window shading
  drawCandidateShading(n, PAD_L, PAD_T, plotW, plotH);

  // Event markers + playhead + x-axis
  drawEventMarkers(n, PAD_L, PAD_T, plotW, plotH);
  drawPlayhead(n, PAD_L, PAD_T, plotW, plotH);
  drawXAxis(n, PAD_L, PAD_T, plotW, plotH);
}

function drawAttemptChart() {
  if (!attemptData || !attemptData.attempt_score) return;

  const canvas = scoreCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width;
  const H = Math.max(140, rect.height - 30);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  scoreCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const n = attemptData.frame_indices.length;
  const PAD_L = 40, PAD_R = 10, PAD_T = 18, PAD_B = 24;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  scoreCtx.fillStyle = "#0d0d1a";
  scoreCtx.fillRect(0, 0, W, H);

  // Title
  scoreCtx.fillStyle = "#888";
  scoreCtx.font = "bold 10px sans-serif";
  scoreCtx.textAlign = "left";
  scoreCtx.fillText("Attempt Score (combined + components) — click or press T to toggle", PAD_L, 12);

  // All attempt components normalized to [0,1]
  const components = ["attempt_score", "clip_contact", "visual_change", "state_change", "gripper_change"];

  // Draw each component
  for (const key of components) {
    const arr = attemptData[key];
    if (!arr) continue;
    scoreCtx.strokeStyle = ATTEMPT_COLORS[key];
    scoreCtx.lineWidth = key === "attempt_score" ? 2.5 : 1.2;
    if (key !== "attempt_score") scoreCtx.setLineDash([4, 3]);
    scoreCtx.globalAlpha = key === "attempt_score" ? 1.0 : 0.6;
    scoreCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = PAD_L + (i / (n - 1)) * plotW;
      const y = PAD_T + plotH - arr[i] * plotH;
      if (i === 0) scoreCtx.moveTo(x, y);
      else scoreCtx.lineTo(x, y);
    }
    scoreCtx.stroke();
    scoreCtx.setLineDash([]);
    scoreCtx.globalAlpha = 1.0;
  }

  // Grid
  drawGrid(W, H, PAD_L, PAD_R, PAD_T, PAD_B, plotW, plotH, 0, 1, 1);

  // Candidate window shading
  drawCandidateShading(n, PAD_L, PAD_T, plotW, plotH);

  // Event markers + playhead + x-axis
  drawEventMarkers(n, PAD_L, PAD_T, plotW, plotH);
  drawPlayhead(n, PAD_L, PAD_T, plotW, plotH);
  drawXAxis(n, PAD_L, PAD_T, plotW, plotH);
}

// =========================================================================
// Chart helpers
// =========================================================================
function drawGrid(W, H, PAD_L, PAD_R, PAD_T, PAD_B, plotW, plotH, gmin, gmax, range) {
  scoreCtx.strokeStyle = "#222";
  scoreCtx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD_T + (plotH * i / 4);
    scoreCtx.beginPath();
    scoreCtx.moveTo(PAD_L, y);
    scoreCtx.lineTo(PAD_L + plotW, y);
    scoreCtx.stroke();
    scoreCtx.fillStyle = "#555";
    scoreCtx.font = "9px monospace";
    scoreCtx.textAlign = "right";
    scoreCtx.textBaseline = "middle";
    const val = gmax - (range * i / 4);
    scoreCtx.fillText(val.toFixed(2), PAD_L - 4, y);
  }
}

function drawCandidateShading(n, PAD_L, PAD_T, plotW, plotH) {
  if (!candidateData || !candidateData.length || !labelData) return;
  const seg = filteredSegments[currentIdx];

  for (const cand of candidateData) {
    const startLocal = cand.start_local;
    const endLocal = cand.end_local;
    const centerLocal = cand.center_frame;

    // Check if this is the success window
    const isSuccess = seg && seg.transition_frame != null &&
      labelData.frames[centerLocal] &&
      labelData.frames[centerLocal].frame === seg.transition_frame;

    const isFailed = seg && seg.failed_attempt_frames &&
      labelData.frames[centerLocal] &&
      seg.failed_attempt_frames.includes(labelData.frames[centerLocal].frame);

    const x1 = PAD_L + (startLocal / (n - 1)) * plotW;
    const x2 = PAD_L + (endLocal / (n - 1)) * plotW;

    if (isSuccess) {
      scoreCtx.fillStyle = "rgba(46, 204, 113, 0.12)";
    } else if (isFailed) {
      scoreCtx.fillStyle = "rgba(231, 76, 60, 0.12)";
    } else {
      scoreCtx.fillStyle = "rgba(127, 140, 141, 0.08)";
    }
    scoreCtx.fillRect(x1, PAD_T, x2 - x1, plotH);
  }
}

function drawEventMarkers(n, PAD_L, PAD_T, plotW, plotH) {
  const seg = filteredSegments[currentIdx];
  if (!seg || !labelData) return;

  // Transition start
  if (seg.transition_start != null) {
    const localIdx = labelData.frames.findIndex(f => f.frame === seg.transition_start);
    if (localIdx >= 0 && n > 1) {
      const sx = PAD_L + (localIdx / (n - 1)) * plotW;
      scoreCtx.strokeStyle = "#e67e22";
      scoreCtx.lineWidth = 1.5;
      scoreCtx.setLineDash([3, 2]);
      scoreCtx.beginPath();
      scoreCtx.moveTo(sx, PAD_T);
      scoreCtx.lineTo(sx, PAD_T + plotH);
      scoreCtx.stroke();
      scoreCtx.setLineDash([]);
    }
  }

  // Transition peak
  if (seg.transition_frame != null) {
    const localIdx = labelData.frames.findIndex(f => f.frame === seg.transition_frame);
    if (localIdx >= 0 && n > 1) {
      const sx = PAD_L + (localIdx / (n - 1)) * plotW;
      scoreCtx.strokeStyle = "#2ecc71";
      scoreCtx.lineWidth = 1.5;
      scoreCtx.setLineDash([3, 2]);
      scoreCtx.beginPath();
      scoreCtx.moveTo(sx, PAD_T);
      scoreCtx.lineTo(sx, PAD_T + plotH);
      scoreCtx.stroke();
      scoreCtx.setLineDash([]);
    }
  }

  // Completion start
  if (seg.completion_start != null) {
    const localIdx = labelData.frames.findIndex(f => f.frame === seg.completion_start);
    if (localIdx >= 0 && n > 1) {
      const cx = PAD_L + (localIdx / (n - 1)) * plotW;
      scoreCtx.strokeStyle = "#27ae60";
      scoreCtx.lineWidth = 1;
      scoreCtx.setLineDash([2, 3]);
      scoreCtx.beginPath();
      scoreCtx.moveTo(cx, PAD_T);
      scoreCtx.lineTo(cx, PAD_T + plotH);
      scoreCtx.stroke();
      scoreCtx.setLineDash([]);
    }
  }

  // Failed attempt frames
  if (seg.failed_attempt_frames) {
    for (const ff of seg.failed_attempt_frames) {
      const localIdx = labelData.frames.findIndex(f => f.frame === ff);
      if (localIdx >= 0 && n > 1) {
        const fx = PAD_L + (localIdx / (n - 1)) * plotW;
        scoreCtx.strokeStyle = "#e74c3c";
        scoreCtx.lineWidth = 1;
        scoreCtx.setLineDash([2, 2]);
        scoreCtx.beginPath();
        scoreCtx.moveTo(fx, PAD_T);
        scoreCtx.lineTo(fx, PAD_T + plotH);
        scoreCtx.stroke();
        scoreCtx.setLineDash([]);
      }
    }
  }
}

function drawPlayhead(n, PAD_L, PAD_T, plotW, plotH) {
  if (n <= 0) return;
  const px = PAD_L + (currentFrame / (n - 1)) * plotW;
  scoreCtx.strokeStyle = "rgba(255,255,255,0.7)";
  scoreCtx.lineWidth = 1.5;
  scoreCtx.setLineDash([4, 3]);
  scoreCtx.beginPath();
  scoreCtx.moveTo(px, PAD_T);
  scoreCtx.lineTo(px, PAD_T + plotH);
  scoreCtx.stroke();
  scoreCtx.setLineDash([]);
}

function drawXAxis(n, PAD_L, PAD_T, plotW, plotH) {
  const data = activeChart === "attempt" ? attemptData : scoreData;
  if (!data || !data.frame_indices) return;
  scoreCtx.fillStyle = "#555";
  scoreCtx.font = "9px monospace";
  scoreCtx.textAlign = "center";
  scoreCtx.textBaseline = "top";
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {
    const x = PAD_L + (i / (n - 1)) * plotW;
    scoreCtx.fillText(String(data.frame_indices[i]), x, PAD_T + plotH + 4);
  }
}

// =========================================================================
// Legend
// =========================================================================
function buildLegend() {
  if (activeChart === "attempt") {
    const components = [
      { key: "attempt_score", label: "Combined" },
      { key: "clip_contact", label: "CLIP Contact" },
      { key: "visual_change", label: "Visual Change" },
      { key: "state_change", label: "State Change" },
      { key: "gripper_change", label: "Gripper Change" },
    ];
    phaseLegend.innerHTML = components.map(c => `
      <div class="legend-item">
        <div class="legend-swatch" style="background:${ATTEMPT_COLORS[c.key]};"></div>
        ${c.label}
      </div>
    `).join("");
  } else {
    phaseLegend.innerHTML = CLIP_SCORE_KEYS.map((key, i) => `
      <div class="legend-item">
        <div class="legend-swatch" style="background:${CLIP_SCORE_COLORS[i]};"></div>
        ${PHASE_LABELS_NICE[key]}
      </div>
    `).join("");
  }
}

// =========================================================================
// Resize
// =========================================================================
window.addEventListener("resize", () => {
  drawScoreChart();
});

// =========================================================================
// Utility
// =========================================================================
async function fetchJSON(url) { return (await fetch(url)).json(); }

// =========================================================================
// Boot
// =========================================================================
init();
