// =========================================================================
// Frame Importance Annotator — Per-Episode with Video Player
// =========================================================================

const CELL_W = 64;   // px per frame in timeline
const CELL_H = 80;   // timeline canvas height (thumbnails + labels)
const THUMB_SRC = 64; // source thumbnail size in sprite sheet

// ---- State ----
let tasks = [];
let currentTask = null;
let episodes = [];
let epListIdx = 0;
let timelineMeta = [];
let annotations = {};
let segments = [];        // current episode's selected segments [[start,end], ...]
let spriteImg = null;
let isDragging = false;
let dragStartFrame = -1;
let dragCurrentFrame = -1;
let currentFrame = 0;     // current video frame index (synced with video)
let animFrameId = null;   // for the playhead tracking loop
let subgoalEntries = [];  // [{name, startIdx}, ...] ordered unique subgoals

// ---- DOM refs ----
const $ = (s) => document.querySelector(s);
const taskSelect   = $("#task-select");
const epPrev       = $("#ep-prev");
const epNext       = $("#ep-next");
const epLabel      = $("#ep-label");
const progressLbl  = $("#progress-label");
const video        = $("#video-player");
const videoBorder  = $("#video-border");
const frameCounter = $("#frame-counter");
const phaseBadge   = $("#phase-badge");
const subgoalText  = $("#subgoal-text");
const goalDisplay  = $("#goal-display");
const subgoalsList = $("#subgoals-list");
const btnPlayPause = $("#btn-play-pause");
const btnReset     = $("#btn-reset");
const speedSlider  = $("#speed-slider");
const speedLabel   = $("#speed-label");
const segmentsList = $("#segments-list");
const btnClearAll  = $("#btn-clear-all");
const btnSave      = $("#btn-save");
const btnSkip      = $("#btn-skip");
const timelineScroll = $("#timeline-scroll");
const canvas       = $("#timeline-canvas");
const ctx          = canvas.getContext("2d");

// =========================================================================
// Utilities
// =========================================================================
async function fetchJSON(url) { return (await fetch(url)).json(); }
function currentEpisode() { return episodes[epListIdx] || null; }

function frameIndexFromTime(time) {
  // Derive frame from video progress ratio — immune to encoding fps drift
  const ep = currentEpisode();
  if (!ep || !video.duration || !isFinite(video.duration)) return 0;
  const ratio = time / video.duration;
  return Math.min(Math.floor(ratio * ep.total_timesteps), ep.total_timesteps - 1);
}

function timeFromFrame(frame) {
  const ep = currentEpisode();
  if (!ep || !video.duration || !isFinite(video.duration) || ep.total_timesteps === 0) return 0;
  return (frame / ep.total_timesteps) * video.duration;
}

// =========================================================================
// Init
// =========================================================================
async function init() {
  tasks = await fetchJSON("/api/tasks");
  taskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join("");
  const defaultTask = tasks.includes("PatternLock") ? "PatternLock" : tasks[0];
  taskSelect.value = defaultTask;

  taskSelect.addEventListener("change", () => loadTask(taskSelect.value));
  epPrev.addEventListener("click", () => changeEpisode(-1));
  epNext.addEventListener("click", () => changeEpisode(1));
  btnPlayPause.addEventListener("click", togglePlayPause);
  btnReset.addEventListener("click", resetVideo);
  speedSlider.addEventListener("input", updateSpeed);
  btnClearAll.addEventListener("click", clearAllSegments);
  btnSave.addEventListener("click", saveAnnotation);
  btnSkip.addEventListener("click", skipToNextUnannotated);

  // Timeline mouse events
  canvas.addEventListener("mousedown", onTlDown);
  canvas.addEventListener("mousemove", onTlMove);
  canvas.addEventListener("mouseup", onTlUp);
  canvas.addEventListener("mouseleave", onTlLeave);

  // Click on timeline to seek video
  // (handled inside onTlUp when not dragging a range)

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlayPause(); }
    if (e.key === "s" || e.key === "S") { saveAnnotation(); }
    if (e.key === "r" || e.key === "R") { resetVideo(); }
    if (e.key === "ArrowRight") { e.preventDefault(); changeEpisode(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); changeEpisode(-1); }
  });

  // Video time update -> sync playhead
  video.addEventListener("timeupdate", onVideoTimeUpdate);
  // Also use requestAnimationFrame for smoother playhead
  startPlayheadLoop();

  await loadTask(defaultTask);
}

// =========================================================================
// Loading
// =========================================================================
async function loadTask(task) {
  currentTask = task;
  episodes = await fetchJSON(`/api/${task}/episodes`);
  annotations = await fetchJSON(`/api/annotations/${task}`);
  epListIdx = 0;
  await loadEpisode();
}

async function loadEpisode() {
  const ep = currentEpisode();
  if (!ep) return;
  epLabel.textContent = `${epListIdx + 1} / ${episodes.length}  (ep ${ep.idx})`;
  goalDisplay.textContent = ep.task_goal;

  // Load timeline metadata
  timelineMeta = await fetchJSON(`/api/${currentTask}/${ep.idx}/timeline_meta`);

  // Build ordered subgoals list (unique, in order of first appearance)
  subgoalEntries = [];
  const seen = new Set();
  for (let i = 0; i < timelineMeta.length; i++) {
    const sg = timelineMeta[i].subgoal;
    if (sg && !seen.has(sg)) {
      seen.add(sg);
      subgoalEntries.push({ name: sg, startIdx: i });
    }
  }
  renderSubgoalsList();

  // Load saved annotation
  const epKey = String(ep.idx);
  const saved = annotations.episodes?.[epKey];
  segments = saved?.selected_segments ? JSON.parse(JSON.stringify(saved.selected_segments)) : [];

  // Load video
  video.pause();
  video.src = `/api/${currentTask}/${ep.idx}/video`;
  video.playbackRate = parseFloat(speedSlider.value);
  video.load();
  currentFrame = 0;

  // Load sprite sheet
  spriteImg = new Image();
  spriteImg.src = `/api/${currentTask}/${ep.idx}/thumbstrip`;
  await new Promise(r => { spriteImg.onload = r; spriteImg.onerror = r; });

  // Size canvas
  const totalW = timelineMeta.length * CELL_W;
  canvas.width = totalW;
  canvas.height = CELL_H;
  canvas.style.width = totalW + "px";
  canvas.style.height = CELL_H + "px";

  drawTimeline();
  renderSegments();
  updateInfoBar();
  updateProgress();
}

// =========================================================================
// Video controls
// =========================================================================
function togglePlayPause() {
  if (video.paused) {
    video.play();
    btnPlayPause.textContent = "Pause";
  } else {
    video.pause();
    btnPlayPause.textContent = "Play";
  }
}

function resetVideo() {
  video.pause();
  video.currentTime = 0;
  currentFrame = 0;
  btnPlayPause.textContent = "Play";
  timelineScroll.scrollLeft = 0;
  updateInfoBar();
  drawTimeline();
}

function updateSpeed() {
  const spd = parseFloat(speedSlider.value);
  video.playbackRate = spd;
  speedLabel.textContent = spd.toFixed(1) + "x";
}

function onVideoTimeUpdate() {
  currentFrame = frameIndexFromTime(video.currentTime);
  updateInfoBar();
  drawTimeline();
  autoScrollTimeline();
}

function startPlayheadLoop() {
  function tick() {
    if (!video.paused) {
      currentFrame = frameIndexFromTime(video.currentTime);
      updateInfoBar();
      drawTimeline();
      autoScrollTimeline();
    }
    animFrameId = requestAnimationFrame(tick);
  }
  tick();
}

function autoScrollTimeline() {
  // Keep the playhead roughly centered in the visible scroll area,
  // but only scroll when it's about to leave the visible region.
  const playheadX = currentFrame * CELL_W + CELL_W / 2;
  const scrollLeft = timelineScroll.scrollLeft;
  const viewW = timelineScroll.clientWidth;
  const margin = viewW * 0.25; // start scrolling when playhead is in outer 25%

  if (playheadX < scrollLeft + margin) {
    timelineScroll.scrollLeft = playheadX - margin;
  } else if (playheadX > scrollLeft + viewW - margin) {
    timelineScroll.scrollLeft = playheadX - viewW + margin;
  }
}

function renderSubgoalsList() {
  subgoalsList.innerHTML = "";
  for (let i = 0; i < subgoalEntries.length; i++) {
    const li = document.createElement("li");
    li.className = "subgoal-item";
    li.dataset.sgIdx = i;
    li.innerHTML = `<span class="subgoal-idx">${i + 1}.</span> ${subgoalEntries[i].name}`;
    subgoalsList.appendChild(li);
  }
}

function updateInfoBar() {
  const ep = currentEpisode();
  if (!ep || timelineMeta.length === 0) return;

  const f = Math.min(currentFrame, timelineMeta.length - 1);
  const meta = timelineMeta[f];
  frameCounter.textContent = `Frame ${f} / ${ep.total_timesteps - 1}`;

  // Phase badge + border color
  if (meta.is_demo) {
    phaseBadge.textContent = "DEMO";
    phaseBadge.className = "demo";
    videoBorder.className = "demo";
  } else {
    phaseBadge.textContent = "EXEC";
    phaseBadge.className = "exec";
    videoBorder.className = "exec";
  }

  subgoalText.textContent = meta.subgoal || "";

  // Highlight active subgoal in the list
  const currentSg = meta.subgoal || "";
  const items = subgoalsList.querySelectorAll(".subgoal-item");
  for (const item of items) {
    const idx = parseInt(item.dataset.sgIdx);
    if (subgoalEntries[idx] && subgoalEntries[idx].name === currentSg) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  }
}

// =========================================================================
// Timeline drawing
// =========================================================================
function drawTimeline() {
  const W = canvas.width;
  const H = CELL_H;
  const HEADER = 16;  // phase bar at top
  const THUMB_Y = HEADER + 1;
  const THUMB_H = CELL_W - 4;  // leave room for frame numbers
  const LABEL_Y = H - 2;

  ctx.clearRect(0, 0, W, H);

  // Find demo/exec boundary
  let execStartIdx = timelineMeta.length;
  for (let i = 0; i < timelineMeta.length; i++) {
    if (!timelineMeta[i].is_demo) { execStartIdx = i; break; }
  }

  // Phase header bar
  ctx.fillStyle = "rgba(45,107,207,0.5)";
  ctx.fillRect(0, 0, execStartIdx * CELL_W, HEADER);
  ctx.fillStyle = "rgba(46,167,112,0.45)";
  ctx.fillRect(execStartIdx * CELL_W, 0, (timelineMeta.length - execStartIdx) * CELL_W, HEADER);

  // Phase labels
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if (execStartIdx > 0) {
    ctx.fillStyle = "#aaccff";
    ctx.fillText("DEMO", (execStartIdx * CELL_W) / 2, HEADER / 2);
  }
  ctx.fillStyle = "#90dbb5";
  ctx.fillText("EXECUTION", execStartIdx * CELL_W + ((timelineMeta.length - execStartIdx) * CELL_W) / 2, HEADER / 2);

  // Phase backgrounds
  ctx.fillStyle = "rgba(45,107,207,0.12)";
  ctx.fillRect(0, HEADER, execStartIdx * CELL_W, H - HEADER);
  ctx.fillStyle = "rgba(46,167,112,0.08)";
  ctx.fillRect(execStartIdx * CELL_W, HEADER, (timelineMeta.length - execStartIdx) * CELL_W, H - HEADER);

  // Demo/exec divider
  if (execStartIdx > 0 && execStartIdx < timelineMeta.length) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(execStartIdx * CELL_W, 0);
    ctx.lineTo(execStartIdx * CELL_W, H);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  // Draw thumbnails
  for (let i = 0; i < timelineMeta.length; i++) {
    const x = i * CELL_W;
    if (spriteImg && spriteImg.complete && spriteImg.naturalWidth > 0) {
      ctx.drawImage(spriteImg, i * THUMB_SRC, 0, THUMB_SRC, THUMB_SRC, x + 2, THUMB_Y, THUMB_H, THUMB_H);
    }
    // Subtle border
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.strokeRect(x + 2, THUMB_Y, THUMB_H, THUMB_H);

    // Subgoal boundary
    if (timelineMeta[i].is_boundary && i > 0) {
      ctx.strokeStyle = "rgba(255,255,0,0.5)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, HEADER);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Selection overlays
  for (const [start, end] of segments) {
    const sx = start * CELL_W;
    const sw = (end - start + 1) * CELL_W;
    ctx.fillStyle = "rgba(255,165,0,0.35)";
    ctx.fillRect(sx, THUMB_Y, sw, THUMB_H);
    ctx.strokeStyle = "rgba(255,165,0,0.8)";
    ctx.lineWidth = 2;
    ctx.strokeRect(sx + 1, THUMB_Y + 1, sw - 2, THUMB_H - 2);
    ctx.lineWidth = 1;
  }

  // Drag preview
  if (isDragging && dragStartFrame >= 0 && dragCurrentFrame >= 0) {
    const s = Math.min(dragStartFrame, dragCurrentFrame);
    const e = Math.max(dragStartFrame, dragCurrentFrame);
    const sx = s * CELL_W;
    const sw = (e - s + 1) * CELL_W;
    ctx.fillStyle = "rgba(255,165,0,0.2)";
    ctx.fillRect(sx, THUMB_Y, sw, THUMB_H);
    ctx.strokeStyle = "rgba(255,165,0,0.6)";
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 2;
    ctx.strokeRect(sx, THUMB_Y, sw, THUMB_H);
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
  }

  // Playhead marker (red line synced with video)
  if (timelineMeta.length > 0) {
    const px = currentFrame * CELL_W + CELL_W / 2;
    // Red line
    ctx.strokeStyle = "rgba(255,50,50,0.9)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.lineWidth = 1;
    // Triangle
    ctx.fillStyle = "rgba(255,50,50,0.9)";
    ctx.beginPath();
    ctx.moveTo(px - 6, 0);
    ctx.lineTo(px + 6, 0);
    ctx.lineTo(px, 8);
    ctx.closePath();
    ctx.fill();
  }

  // Frame numbers every 10
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (let i = 0; i < timelineMeta.length; i += 10) {
    ctx.fillText(String(i), i * CELL_W + CELL_W / 2, LABEL_Y);
  }
}

// =========================================================================
// Timeline mouse — drag to select segments, click to seek
// =========================================================================
function frameFromMouse(e) {
  const rect = canvas.getBoundingClientRect();
  // e.clientX is relative to viewport; rect.left accounts for scroll already
  // But canvas is inside a scrollable div, so we need scrollLeft offset
  const x = e.clientX - rect.left + timelineScroll.scrollLeft;
  // rect.left already includes the scroll position of the page, but
  // since the canvas is wider than the scroll container, we need to
  // account for the scroll container's own scroll.
  // Actually: getBoundingClientRect gives position relative to viewport.
  // The canvas is positioned at the start of the scroll container.
  // If the user scrolled, the canvas has moved left, so rect.left is
  // already shifted. So: x = e.clientX - rect.left is correct.
  const xFixed = e.clientX - rect.left;
  return Math.max(0, Math.min(timelineMeta.length - 1, Math.floor(xFixed / CELL_W)));
}

function onTlDown(e) {
  isDragging = true;
  dragStartFrame = frameFromMouse(e);
  dragCurrentFrame = dragStartFrame;
  drawTimeline();
}

function onTlMove(e) {
  if (!isDragging) return;
  dragCurrentFrame = frameFromMouse(e);
  drawTimeline();
}

function onTlUp(e) {
  if (!isDragging) return;
  isDragging = false;
  const s = Math.min(dragStartFrame, dragCurrentFrame);
  const eF = Math.max(dragStartFrame, dragCurrentFrame);

  if (s === eF) {
    // Single click -> seek video to this frame
    video.currentTime = timeFromFrame(s);
    currentFrame = s;
    updateInfoBar();
  } else {
    // Drag -> add segment
    addSegment(s, eF);
  }
  dragStartFrame = -1;
  dragCurrentFrame = -1;
  drawTimeline();
}

function onTlLeave(e) {
  if (isDragging) onTlUp(e);
}

// =========================================================================
// Segment management
// =========================================================================
function addSegment(start, end) {
  segments.push([start, end]);
  segments = mergeSegments(segments);
  renderSegments();
  drawTimeline();
}

function removeSegment(idx) {
  segments.splice(idx, 1);
  renderSegments();
  drawTimeline();
}

function clearAllSegments() {
  segments = [];
  renderSegments();
  drawTimeline();
}

function mergeSegments(segs) {
  if (segs.length <= 1) return segs;
  segs.sort((a, b) => a[0] - b[0]);
  const merged = [segs[0].slice()];
  for (let i = 1; i < segs.length; i++) {
    const last = merged[merged.length - 1];
    if (segs[i][0] <= last[1] + 1) {
      last[1] = Math.max(last[1], segs[i][1]);
    } else {
      merged.push(segs[i].slice());
    }
  }
  return merged;
}

function renderSegments() {
  segmentsList.innerHTML = "";
  if (segments.length === 0) {
    segmentsList.innerHTML = '<span class="muted">None yet - drag on the timeline</span>';
    return;
  }
  const ep = currentEpisode();
  const execStart = ep ? ep.exec_start_idx : Infinity;

  for (let i = 0; i < segments.length; i++) {
    const [s, e] = segments[i];
    // Determine phase label
    let phaseLabel = "";
    if (e < execStart) phaseLabel = "demo";
    else if (s >= execStart) phaseLabel = "exec";
    else phaseLabel = "mixed";

    const chip = document.createElement("div");
    chip.className = "segment-chip";
    chip.innerHTML = `
      <span class="chip-phase ${phaseLabel === 'demo' ? 'demo' : phaseLabel === 'exec' ? 'exec' : ''}">${phaseLabel}</span>
      frames ${s}\u2013${e} (${e - s + 1} frames)
      <span class="chip-x">\u00d7</span>
    `;
    chip.addEventListener("click", () => removeSegment(i));
    segmentsList.appendChild(chip);
  }
}

// =========================================================================
// Navigation
// =========================================================================
async function changeEpisode(delta) {
  await saveAnnotation();
  const newIdx = epListIdx + delta;
  if (newIdx < 0 || newIdx >= episodes.length) return;
  epListIdx = newIdx;
  await loadEpisode();
}

async function skipToNextUnannotated() {
  await saveAnnotation();
  for (let i = epListIdx + 1; i < episodes.length; i++) {
    const epKey = String(episodes[i].idx);
    const saved = annotations.episodes?.[epKey];
    if (!saved?.selected_segments || saved.selected_segments.length === 0) {
      epListIdx = i;
      await loadEpisode();
      return;
    }
  }
  // Wrap around
  for (let i = 0; i < epListIdx; i++) {
    const epKey = String(episodes[i].idx);
    const saved = annotations.episodes?.[epKey];
    if (!saved?.selected_segments || saved.selected_segments.length === 0) {
      epListIdx = i;
      await loadEpisode();
      return;
    }
  }
}

// =========================================================================
// Save / progress
// =========================================================================
async function saveAnnotation() {
  const ep = currentEpisode();
  if (!ep) return;

  // Update local cache
  const epKey = String(ep.idx);
  if (!annotations.episodes) annotations.episodes = {};
  if (!annotations.episodes[epKey]) annotations.episodes[epKey] = {};
  annotations.episodes[epKey].selected_segments = segments;

  await fetch(`/api/annotations/${currentTask}/${ep.idx}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_segments: segments }),
  });
  updateProgress();
}

function updateProgress() {
  let annotated = 0;
  for (const ep of episodes) {
    const saved = annotations.episodes?.[String(ep.idx)];
    if (saved?.selected_segments && saved.selected_segments.length > 0) annotated++;
  }
  progressLbl.textContent = `${annotated} / ${episodes.length} episodes annotated`;
}

// =========================================================================
// Boot
// =========================================================================
init();
