// =========================================================================
// TOPReward Viewer — Interactive progress-curve and success-interval viewer
// =========================================================================

const CELL_W = 56;
const TL_H   = 88;
const THUMB_SRC = 64;

// ---- State ----
let allSegments = [];
let filteredSegments = [];
let currentIdx = 0;
let segMeta = null;         // current segment metadata
let progressData = null;    // progress_scores.json
let intervalData = null;    // successful_interval.json
let spriteImg = null;
let currentFrame = 0;
let animId = null;

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
const speedSlider    = $("#speed-slider");
const confBadge      = $("#confidence-badge");
const subtaskDisplay = $("#subtask-display");
const goalDisplay    = $("#task-goal-display");
const intervalInfo   = $("#interval-info");
const failedInfo     = $("#failed-info");
const keyframeStrip  = $("#keyframe-strip");
const chartLegend    = $("#chart-legend");
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
    segListScroll.innerHTML = '<div style="padding:20px;color:#666;">No segments found. Run pseudo_label_topreward.py first.</div>';
    return;
  }

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

  tlCanvas.addEventListener("click", onTimelineClick);

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") { e.preventDefault(); navigate(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); navigate(-1); }
    if (e.key === "r" || e.key === "R") resetVideo();
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
    const conf = (s.confidence || 0).toFixed(2);
    let confClass = "conf-none";
    if (s.confidence >= 0.7) confClass = "conf-high";
    else if (s.confidence >= 0.4) confClass = "conf-mid";
    else if (s.confidence > 0) confClass = "conf-low";

    const successIcon = s.success_detected ? "&#10003;" : "&#10007;";
    const successColor = s.success_detected ? "#2ecc71" : "#e74c3c";

    const div = document.createElement("div");
    div.className = "seg-item" + (i === currentIdx ? " active" : "");
    div.dataset.idx = i;
    div.innerHTML = `
      <div class="seg-item-env">${s.env_id} &middot; ep${s.episode_idx}
        <span style="color:${successColor};margin-left:4px;">${successIcon}</span>
      </div>
      <div class="seg-item-label">${s.subtask_label}</div>
      <div class="seg-item-meta">
        <span>${s.num_frames} frames</span>
        <span class="seg-item-conf ${confClass}">${conf}</span>
        <span>${s.failed_regions.length} failed</span>
      </div>
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
  segMeta = filteredSegments[idx];

  // Highlight
  segListScroll.querySelectorAll(".seg-item").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
  });
  const activeEl = segListScroll.querySelector(".seg-item.active");
  if (activeEl) activeEl.scrollIntoView({ block: "nearest" });

  segCounter.textContent = `${idx + 1} / ${filteredSegments.length}`;
  subtaskDisplay.textContent = `"${segMeta.subtask_label}"`;
  goalDisplay.textContent = segMeta.task_goal;

  // Confidence badge
  updateConfBadge(segMeta.confidence, segMeta.success_detected);

  // Interval info
  let iiHtml = "";
  if (segMeta.successful_interval_start_frame != null) {
    iiHtml += `<div class="ii-item"><span class="ii-dot" style="background:#2ecc71;"></span>Interval: ${segMeta.successful_interval_start_frame}&ndash;${segMeta.successful_interval_end_frame}</div>`;
  }
  if (segMeta.completion_start_frame != null) {
    iiHtml += `<div class="ii-item"><span class="ii-dot" style="background:#f39c12;"></span>Completion start: ${segMeta.completion_start_frame}</div>`;
  }
  iiHtml += `<div class="ii-item"><span class="ii-dot" style="background:#888;"></span>${segMeta.num_prefixes} prefixes</div>`;
  intervalInfo.innerHTML = iiHtml;

  // Failed regions
  if (segMeta.failed_regions.length > 0) {
    failedInfo.textContent = `Failed regions: ${segMeta.failed_regions.map(r => `[${r[0]}-${r[1]}]`).join(", ")}`;
  } else {
    failedInfo.textContent = "";
  }

  // Load data
  const [progress, interval] = await Promise.all([
    fetchJSON(`/api/segment/${segMeta.seg_id}/progress_scores`),
    fetchJSON(`/api/segment/${segMeta.seg_id}/interval`),
  ]);
  progressData = progress;
  intervalData = interval;

  // Video
  video.pause();
  video.src = `/api/segment/${segMeta.seg_id}/video`;
  video.playbackRate = parseFloat(speedSlider.value);
  video.load();
  currentFrame = 0;
  btnPlayPause.textContent = "Play";

  // Thumbnail strip
  spriteImg = new Image();
  spriteImg.src = `/api/segment/${segMeta.seg_id}/thumbstrip`;
  await new Promise(r => { spriteImg.onload = r; spriteImg.onerror = r; });

  // Keyframe thumbnails
  buildKeyframeStrip();

  // Size timeline canvas
  const n = segMeta.num_frames;
  tlCanvas.width = n * CELL_W;
  tlCanvas.height = TL_H;
  tlCanvas.style.width = n * CELL_W + "px";
  tlCanvas.style.height = TL_H + "px";

  drawTimeline();
  drawScoreChart();
  updateFrameInfo();
}

// =========================================================================
// Confidence badge
// =========================================================================
function updateConfBadge(confidence, success) {
  const c = (confidence || 0).toFixed(2);
  confBadge.textContent = success ? `conf ${c}` : `LOW ${c}`;
  confBadge.className = "conf-badge";
  if (!success) confBadge.classList.add("conf-low");
  else if (confidence >= 0.7) confBadge.classList.add("conf-high");
  else if (confidence >= 0.4) confBadge.classList.add("conf-mid");
  else confBadge.classList.add("conf-low");
}

// =========================================================================
// Keyframe strip
// =========================================================================
function buildKeyframeStrip() {
  keyframeStrip.innerHTML = "";
  if (!segMeta || !segMeta.selected_keyframes) return;

  // Parse start_frame from seg_id
  const startFrame = parseStartFrame(segMeta.seg_id);

  const labels = ["Early Context", "Ramp Start", "Transition", "Completion", "Post-Completion"];
  segMeta.selected_keyframes.forEach((kf, i) => {
    const container = document.createElement("div");
    container.style.textAlign = "center";

    const img = document.createElement("img");
    img.src = `/api/segment/${segMeta.seg_id}/frame/${kf}`;
    img.title = `Keyframe: frame ${kf + startFrame}`;
    img.addEventListener("click", () => seekToLocalFrame(kf));
    container.appendChild(img);

    const lbl = document.createElement("div");
    lbl.style.fontSize = "9px";
    lbl.style.color = "#888";
    lbl.style.marginTop = "2px";
    lbl.textContent = `${labels[i] || "KF"} (${kf + startFrame})`;
    container.appendChild(lbl);

    keyframeStrip.appendChild(container);
  });
}

function parseStartFrame(segId) {
  const parts = segId.rsplit ? segId.split("_f") : segId.split("_f");
  if (parts.length < 2) return 0;
  const range = parts[parts.length - 1];
  const dash = range.indexOf("-");
  if (dash < 0) return 0;
  return parseInt(range.substring(0, dash), 10) || 0;
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
  if (!segMeta) return;
  const n = segMeta.num_frames;
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
function navigate(delta) { loadSegment(currentIdx + delta); }

function seekToLocalFrame(localFrame) {
  if (!segMeta || !video.duration) return;
  const n = segMeta.num_frames;
  video.currentTime = (localFrame / n) * video.duration;
  currentFrame = localFrame;
  updateFrameInfo(); drawTimeline(); drawScoreChart();
}

// =========================================================================
// Frame info
// =========================================================================
function updateFrameInfo() {
  if (!segMeta) return;
  const n = segMeta.num_frames;
  const f = Math.min(currentFrame, n - 1);
  const startFrame = parseStartFrame(segMeta.seg_id);
  const globalIdx = startFrame + f;

  frameCounter.textContent = `Frame ${globalIdx} (${f + 1}/${n})`;

  // Color video border based on region
  videoBorder.className = "";
  const intStart = segMeta.successful_interval_start_frame || 0;
  const compStart = segMeta.completion_start_frame || n;

  if (f >= intStart && segMeta.success_detected) {
    if (f >= compStart) {
      videoBorder.classList.add("success-border");
    } else {
      videoBorder.classList.add("success-border");
      videoBorder.style.borderColor = "#e67e22"; // transition region
    }
  } else {
    // Check if in a failed region
    let inFailed = false;
    for (const [rs, re] of (segMeta.failed_regions || [])) {
      if (f >= rs && f <= re) { inFailed = true; break; }
    }
    if (inFailed) {
      videoBorder.classList.add("fail-border");
    } else {
      videoBorder.classList.add("pre-border");
    }
    videoBorder.style.borderColor = "";
  }

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
// Timeline (frame-level)
// =========================================================================
function drawTimeline() {
  if (!segMeta) return;
  const n = segMeta.num_frames;
  const W = tlCanvas.width;
  const H = TL_H;
  const PHASE_BAR_H = 14;
  const THUMB_Y = PHASE_BAR_H + 1;
  const THUMB_SIZE = CELL_W - 4;

  tlCtx.clearRect(0, 0, W, H);

  const intStart = segMeta.successful_interval_start_frame || 0;
  const compStart = segMeta.completion_start_frame || n;
  const intEnd = segMeta.successful_interval_end_frame || n - 1;
  const failedRegions = segMeta.failed_regions || [];

  // Phase color bar
  for (let i = 0; i < n; i++) {
    let color = "#444";  // pre-interval / discarded

    // Check failed regions
    let inFailed = false;
    for (const [rs, re] of failedRegions) {
      if (i >= rs && i <= re) { inFailed = true; break; }
    }

    if (inFailed) {
      color = "#e74c3c";
    } else if (i >= compStart && i <= intEnd && segMeta.success_detected) {
      color = "#2ecc71";  // completion
    } else if (i >= intStart && i < compStart && segMeta.success_detected) {
      color = "#e67e22";  // transition/ramp
    }

    tlCtx.fillStyle = color;
    tlCtx.fillRect(i * CELL_W, 0, CELL_W, PHASE_BAR_H);
  }

  // Thumbnails
  for (let i = 0; i < n; i++) {
    const x = i * CELL_W;
    if (spriteImg && spriteImg.complete && spriteImg.naturalWidth > 0) {
      tlCtx.drawImage(spriteImg, i * THUMB_SRC, 0, THUMB_SRC, THUMB_SRC,
                       x + 2, THUMB_Y, THUMB_SIZE, THUMB_SIZE);
    }
    tlCtx.strokeStyle = "#333";
    tlCtx.lineWidth = 0.5;
    tlCtx.strokeRect(x + 1, THUMB_Y, CELL_W - 2, THUMB_SIZE);
  }

  // Interval start marker
  if (segMeta.success_detected && intStart < n) {
    const sx = intStart * CELL_W + CELL_W / 2;
    tlCtx.strokeStyle = "#2ecc71";
    tlCtx.lineWidth = 2;
    tlCtx.setLineDash([3, 2]);
    tlCtx.beginPath(); tlCtx.moveTo(sx, 0); tlCtx.lineTo(sx, H); tlCtx.stroke();
    tlCtx.setLineDash([]);
    tlCtx.fillStyle = "#2ecc71";
    tlCtx.font = "bold 8px sans-serif";
    tlCtx.textAlign = "center";
    tlCtx.fillText("INT-START", sx, H - 10);
  }

  // Completion start marker
  if (segMeta.success_detected && compStart < n) {
    const cx = compStart * CELL_W + CELL_W / 2;
    tlCtx.strokeStyle = "#f39c12";
    tlCtx.lineWidth = 2;
    tlCtx.setLineDash([3, 2]);
    tlCtx.beginPath(); tlCtx.moveTo(cx, 0); tlCtx.lineTo(cx, H); tlCtx.stroke();
    tlCtx.setLineDash([]);
    tlCtx.fillStyle = "#f39c12";
    tlCtx.font = "bold 8px sans-serif";
    tlCtx.textAlign = "center";
    tlCtx.fillText("COMPLETE", cx, H - 2);
  }

  // Keyframe markers
  for (const kf of (segMeta.selected_keyframes || [])) {
    if (kf < n) {
      const kx = kf * CELL_W + CELL_W / 2;
      tlCtx.fillStyle = "#9b59b6";
      tlCtx.beginPath();
      tlCtx.moveTo(kx - 4, PHASE_BAR_H);
      tlCtx.lineTo(kx + 4, PHASE_BAR_H);
      tlCtx.lineTo(kx, PHASE_BAR_H + 6);
      tlCtx.closePath();
      tlCtx.fill();
    }
  }

  // Playhead
  const px = currentFrame * CELL_W + CELL_W / 2;
  tlCtx.strokeStyle = "rgba(255,255,255,0.9)";
  tlCtx.lineWidth = 2.5;
  tlCtx.beginPath(); tlCtx.moveTo(px, 0); tlCtx.lineTo(px, H); tlCtx.stroke();
  tlCtx.fillStyle = "#fff";
  tlCtx.beginPath();
  tlCtx.moveTo(px - 5, 0); tlCtx.lineTo(px + 5, 0); tlCtx.lineTo(px, 7);
  tlCtx.closePath(); tlCtx.fill();

  // Frame numbers
  const startFrame = parseStartFrame(segMeta.seg_id);
  tlCtx.fillStyle = "rgba(255,255,255,0.35)";
  tlCtx.font = "8px monospace";
  tlCtx.textAlign = "center";
  tlCtx.textBaseline = "bottom";
  const step = Math.max(1, Math.floor(n / 20));
  for (let i = 0; i < n; i += step) {
    tlCtx.fillText(String(startFrame + i), i * CELL_W + CELL_W / 2, H - 1);
  }
}

function onTimelineClick(e) {
  if (!segMeta) return;
  const rect = tlCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const f = Math.max(0, Math.min(segMeta.num_frames - 1, Math.floor(x / CELL_W)));
  seekToLocalFrame(f);
}

// =========================================================================
// Score chart — interactive progress curve
// =========================================================================
function drawScoreChart() {
  if (!progressData || !progressData.prefix_end_indices) return;

  const canvas = scoreCanvas;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width;
  const H = Math.max(160, rect.height - 40);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  scoreCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const prefixEnds = progressData.prefix_end_indices;
  const rawReward = progressData.reward_norm;
  const smooth = progressData.reward_norm;  // no smoothing
  const K = prefixEnds.length;

  const PAD_L = 44, PAD_R = 12, PAD_T = 14, PAD_B = 28;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  // Background
  scoreCtx.fillStyle = "#0d0d1a";
  scoreCtx.fillRect(0, 0, W, H);

  // x-axis: prefix endpoints mapped to frame index space
  const xMin = prefixEnds[0];
  const xMax = prefixEnds[K - 1];
  const xRange = xMax - xMin || 1;

  function xPos(frameIdx) { return PAD_L + ((frameIdx - xMin) / xRange) * plotW; }
  function yPos(val) { return PAD_T + plotH - val * plotH; }

  // Grid
  scoreCtx.strokeStyle = "#222";
  scoreCtx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD_T + (plotH * i / 4);
    scoreCtx.beginPath();
    scoreCtx.moveTo(PAD_L, y); scoreCtx.lineTo(PAD_L + plotW, y);
    scoreCtx.stroke();
    scoreCtx.fillStyle = "#555";
    scoreCtx.font = "9px monospace";
    scoreCtx.textAlign = "right";
    scoreCtx.textBaseline = "middle";
    scoreCtx.fillText((1 - i / 4).toFixed(2), PAD_L - 4, y);
  }

  // Failed region shading
  if (segMeta && segMeta.failed_regions) {
    for (const [rs, re] of segMeta.failed_regions) {
      const x1 = xPos(rs);
      const x2 = xPos(re);
      scoreCtx.fillStyle = "rgba(231, 76, 60, 0.15)";
      scoreCtx.fillRect(x1, PAD_T, x2 - x1, plotH);
    }
  }

  // Success interval shading
  if (segMeta && segMeta.success_detected) {
    const intStart = segMeta.successful_interval_start_frame;
    const intEnd = segMeta.successful_interval_end_frame;
    const x1 = xPos(Math.max(intStart, xMin));
    const x2 = xPos(Math.min(intEnd, xMax));
    scoreCtx.fillStyle = "rgba(46, 204, 113, 0.08)";
    scoreCtx.fillRect(x1, PAD_T, x2 - x1, plotH);
  }

  // Raw reward points
  for (let i = 0; i < K; i++) {
    const x = xPos(prefixEnds[i]);
    const y = yPos(rawReward[i]);
    scoreCtx.fillStyle = "rgba(100, 149, 237, 0.6)";
    scoreCtx.beginPath();
    scoreCtx.arc(x, y, 4, 0, Math.PI * 2);
    scoreCtx.fill();
  }

  // Smoothed curve
  if (smooth && smooth.length === K) {
    scoreCtx.strokeStyle = "#1a237e";
    scoreCtx.lineWidth = 2.5;
    scoreCtx.beginPath();
    for (let i = 0; i < K; i++) {
      const x = xPos(prefixEnds[i]);
      const y = yPos(smooth[i]);
      if (i === 0) scoreCtx.moveTo(x, y);
      else scoreCtx.lineTo(x, y);
    }
    scoreCtx.stroke();

    // Smooth points
    for (let i = 0; i < K; i++) {
      const x = xPos(prefixEnds[i]);
      const y = yPos(smooth[i]);
      scoreCtx.fillStyle = "#3949ab";
      scoreCtx.beginPath();
      scoreCtx.arc(x, y, 3, 0, Math.PI * 2);
      scoreCtx.fill();
    }
  }

  // Interval start vertical line
  if (segMeta && segMeta.success_detected && segMeta.successful_interval_start_frame != null) {
    const sx = xPos(segMeta.successful_interval_start_frame);
    scoreCtx.strokeStyle = "#2ecc71";
    scoreCtx.lineWidth = 1.5;
    scoreCtx.setLineDash([4, 3]);
    scoreCtx.beginPath(); scoreCtx.moveTo(sx, PAD_T); scoreCtx.lineTo(sx, PAD_T + plotH); scoreCtx.stroke();
    scoreCtx.setLineDash([]);
    // Label
    scoreCtx.fillStyle = "#2ecc71";
    scoreCtx.font = "bold 9px sans-serif";
    scoreCtx.textAlign = "center";
    scoreCtx.fillText("INT", sx, PAD_T - 3);
  }

  // Completion start vertical line
  if (segMeta && segMeta.completion_start_frame != null) {
    const cx = xPos(segMeta.completion_start_frame);
    scoreCtx.strokeStyle = "#f39c12";
    scoreCtx.lineWidth = 1.5;
    scoreCtx.setLineDash([4, 3]);
    scoreCtx.beginPath(); scoreCtx.moveTo(cx, PAD_T); scoreCtx.lineTo(cx, PAD_T + plotH); scoreCtx.stroke();
    scoreCtx.setLineDash([]);
    scoreCtx.fillStyle = "#f39c12";
    scoreCtx.font = "bold 9px sans-serif";
    scoreCtx.textAlign = "center";
    scoreCtx.fillText("COMP", cx, PAD_T - 3);
  }

  // Keyframe markers
  for (const kf of (segMeta.selected_keyframes || [])) {
    const kx = xPos(kf);
    if (kx >= PAD_L && kx <= PAD_L + plotW) {
      scoreCtx.fillStyle = "#9b59b6";
      scoreCtx.beginPath();
      scoreCtx.moveTo(kx - 4, PAD_T + plotH + 2);
      scoreCtx.lineTo(kx + 4, PAD_T + plotH + 2);
      scoreCtx.lineTo(kx, PAD_T + plotH - 4);
      scoreCtx.closePath();
      scoreCtx.fill();
    }
  }

  // Playhead
  const startFrame = parseStartFrame(segMeta.seg_id);
  const globalFrame = startFrame + currentFrame;
  if (globalFrame >= xMin && globalFrame <= xMax) {
    const px = xPos(globalFrame);
    scoreCtx.strokeStyle = "rgba(255,255,255,0.7)";
    scoreCtx.lineWidth = 1.5;
    scoreCtx.setLineDash([3, 3]);
    scoreCtx.beginPath(); scoreCtx.moveTo(px, PAD_T); scoreCtx.lineTo(px, PAD_T + plotH); scoreCtx.stroke();
    scoreCtx.setLineDash([]);
  }

  // X-axis labels
  scoreCtx.fillStyle = "#555";
  scoreCtx.font = "9px monospace";
  scoreCtx.textAlign = "center";
  scoreCtx.textBaseline = "top";
  const xStep = Math.max(1, Math.floor(K / 8));
  for (let i = 0; i < K; i += xStep) {
    const x = xPos(prefixEnds[i]);
    scoreCtx.fillText(String(prefixEnds[i]), x, PAD_T + plotH + 6);
  }
}

// =========================================================================
// Legend
// =========================================================================
function buildLegend() {
  const items = [
    { color: "rgba(100,149,237,0.6)", label: "Raw reward (norm)" },
    { color: "#3949ab", label: "Smoothed" },
    { color: "#2ecc71", label: "Interval start" },
    { color: "#f39c12", label: "Completion start" },
    { color: "rgba(231,76,60,0.4)", label: "Failed region" },
    { color: "#9b59b6", label: "Keyframe" },
  ];
  chartLegend.innerHTML = items.map(it => `
    <div class="legend-item">
      <div class="legend-swatch" style="background:${it.color};"></div>
      ${it.label}
    </div>
  `).join("");
}

// =========================================================================
// Resize
// =========================================================================
window.addEventListener("resize", drawScoreChart);

// =========================================================================
// Utility
// =========================================================================
async function fetchJSON(url) { return (await fetch(url)).json(); }

// =========================================================================
// Boot
// =========================================================================
init();
