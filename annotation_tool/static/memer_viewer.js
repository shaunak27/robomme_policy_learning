// =========================================================================
// MeMER Keyframe Viewer
// =========================================================================

const CELL_W = 64;
const CELL_H = 80;
const THUMB_SRC = 64;

// ---- State ----
let tasks = [];
let currentTask = null;
let episodes = [];
let epListIdx = 0;
let timelineMeta = [];
let memerData = null;       // result from /memer_keyframes
let spriteImg = null;
let currentFrame = 0;
let animFrameId = null;
let subgoalEntries = [];

// ---- DOM refs ----
const $ = (s) => document.querySelector(s);
const taskSelect     = $("#task-select");
const epPrev         = $("#ep-prev");
const epNext         = $("#ep-next");
const epLabel        = $("#ep-label");
const kfCountLabel   = $("#keyframe-count");
const video          = $("#video-player");
const videoBorder    = $("#video-border");
const frameCounter   = $("#frame-counter");
const phaseBadge     = $("#phase-badge");
const subgoalText    = $("#subgoal-text");
const groundedText   = $("#grounded-subgoal-text");
const goalDisplay    = $("#goal-display");
const subgoalsList   = $("#subgoals-list");
const btnPlayPause   = $("#btn-play-pause");
const btnReset       = $("#btn-reset");
const speedSlider    = $("#speed-slider");
const speedLabel     = $("#speed-label");
const kfGallery      = $("#keyframe-gallery");
const timelineScroll = $("#timeline-scroll");
const canvas         = $("#timeline-canvas");
const ctx            = canvas.getContext("2d");

// =========================================================================
// Utilities
// =========================================================================
async function fetchJSON(url) { return (await fetch(url)).json(); }
function currentEpisode() { return episodes[epListIdx] || null; }

function frameIndexFromTime(time) {
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

  // Timeline click to seek
  canvas.addEventListener("click", onTlClick);

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlayPause(); }
    if (e.key === "r" || e.key === "R") { resetVideo(); }
    if (e.key === "ArrowRight") { e.preventDefault(); changeEpisode(1); }
    if (e.key === "ArrowLeft") { e.preventDefault(); changeEpisode(-1); }
  });

  video.addEventListener("timeupdate", onVideoTimeUpdate);
  startPlayheadLoop();

  await loadTask(defaultTask);
}

// =========================================================================
// Loading
// =========================================================================
async function loadTask(task) {
  currentTask = task;
  episodes = await fetchJSON(`/api/${task}/episodes`);
  epListIdx = 0;
  await loadEpisode();
}

async function loadEpisode() {
  const ep = currentEpisode();
  if (!ep) return;
  epLabel.textContent = `${epListIdx + 1} / ${episodes.length}  (ep ${ep.idx})`;
  goalDisplay.textContent = ep.task_goal;

  // Load timeline meta + memer keyframes in parallel
  const [meta, memer, _] = await Promise.all([
    fetchJSON(`/api/${currentTask}/${ep.idx}/timeline_meta`),
    fetchJSON(`/api/${currentTask}/${ep.idx}/memer_keyframes`),
    loadSpriteSheet(ep),
  ]);
  timelineMeta = meta;
  memerData = memer;

  kfCountLabel.textContent = `${memerData.keyframes.length} MeMER keyframes`;

  // Build ordered subgoals list (with grounded variant)
  subgoalEntries = [];
  const seen = new Set();
  for (let i = 0; i < timelineMeta.length; i++) {
    const sg = timelineMeta[i].subgoal;
    if (sg && !seen.has(sg)) {
      seen.add(sg);
      subgoalEntries.push({
        name: sg,
        grounded: timelineMeta[i].grounded_subgoal || "",
        startIdx: i,
      });
    }
  }
  renderSubgoalsList();
  renderKeyframeGallery();

  // Load video
  video.pause();
  video.src = `/api/${currentTask}/${ep.idx}/video`;
  video.playbackRate = parseFloat(speedSlider.value);
  video.load();
  currentFrame = 0;

  // Size canvas
  const totalW = timelineMeta.length * CELL_W;
  canvas.width = totalW;
  canvas.height = CELL_H;
  canvas.style.width = totalW + "px";
  canvas.style.height = CELL_H + "px";

  drawTimeline();
  updateInfoBar();
}

async function loadSpriteSheet(ep) {
  spriteImg = new Image();
  spriteImg.src = `/api/${currentTask}/${ep.idx}/thumbstrip`;
  await new Promise(r => { spriteImg.onload = r; spriteImg.onerror = r; });
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
  updateKeyframeHighlight();
}

function startPlayheadLoop() {
  function tick() {
    if (!video.paused) {
      currentFrame = frameIndexFromTime(video.currentTime);
      updateInfoBar();
      drawTimeline();
      autoScrollTimeline();
      updateKeyframeHighlight();
    }
    animFrameId = requestAnimationFrame(tick);
  }
  tick();
}

function autoScrollTimeline() {
  const playheadX = currentFrame * CELL_W + CELL_W / 2;
  const scrollLeft = timelineScroll.scrollLeft;
  const viewW = timelineScroll.clientWidth;
  const margin = viewW * 0.25;
  if (playheadX < scrollLeft + margin) {
    timelineScroll.scrollLeft = playheadX - margin;
  } else if (playheadX > scrollLeft + viewW - margin) {
    timelineScroll.scrollLeft = playheadX - viewW + margin;
  }
}

// =========================================================================
// Subgoals list
// =========================================================================
function renderSubgoalsList() {
  subgoalsList.innerHTML = "";
  for (let i = 0; i < subgoalEntries.length; i++) {
    const li = document.createElement("li");
    li.className = "subgoal-item";
    li.dataset.sgIdx = i;
    const grounded = subgoalEntries[i].grounded;
    li.innerHTML = `<span class="subgoal-idx">${i + 1}.</span> ${subgoalEntries[i].name}`
      + (grounded ? `<span class="subgoal-grounded">${grounded}</span>` : "");
    subgoalsList.appendChild(li);
  }
}

function updateInfoBar() {
  const ep = currentEpisode();
  if (!ep || timelineMeta.length === 0) return;

  const f = Math.min(currentFrame, timelineMeta.length - 1);
  const meta = timelineMeta[f];
  frameCounter.textContent = `Frame ${f} / ${ep.total_timesteps - 1}`;

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
  groundedText.textContent = meta.grounded_subgoal || "";

  // Highlight active subgoal
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
// Keyframe gallery
// =========================================================================
function renderKeyframeGallery() {
  kfGallery.innerHTML = "";
  if (!memerData || memerData.keyframes.length === 0) {
    kfGallery.innerHTML = '<span class="muted">No keyframes detected</span>';
    return;
  }

  const ep = currentEpisode();
  for (const kf of memerData.keyframes) {
    const card = document.createElement("div");
    card.className = "kf-card";
    card.dataset.idx = kf.idx;

    const badges = kf.provenance.map(p => {
      const label = p === "transition" ? "T" : p === "local_minimum" ? "M" : p === "midpoint" ? "I" : "G";
      return `<span class="kf-badge ${p}">${label}</span>`;
    }).join("");

    card.innerHTML = `
      <img src="/api/${currentTask}/${ep.idx}/${kf.idx}/frame/front_rgb" alt="Frame ${kf.idx}" loading="lazy">
      <div class="kf-card-info">
        <span class="kf-frame-label">Frame ${kf.idx}</span>
        <span class="kf-subgoal-label" title="${kf.simple_subgoal}">${kf.simple_subgoal || "-"}</span>
        <span class="kf-grounded-label" title="${kf.grounded_subgoal}">${kf.grounded_subgoal || "-"}</span>
        <div class="kf-badges">${badges}</div>
      </div>
    `;

    card.addEventListener("click", () => {
      video.currentTime = timeFromFrame(kf.idx);
      currentFrame = kf.idx;
      updateInfoBar();
      drawTimeline();
      updateKeyframeHighlight();
    });

    kfGallery.appendChild(card);
  }
}

function updateKeyframeHighlight() {
  if (!memerData) return;
  const cards = kfGallery.querySelectorAll(".kf-card");
  // Find closest keyframe to current frame
  let closestIdx = -1;
  let closestDist = Infinity;
  for (const kf of memerData.keyframes) {
    const dist = Math.abs(kf.idx - currentFrame);
    if (dist < closestDist) {
      closestDist = dist;
      closestIdx = kf.idx;
    }
  }

  for (const card of cards) {
    const idx = parseInt(card.dataset.idx);
    if (idx === closestIdx && closestDist <= 2) {
      card.classList.add("active");
      // Scroll card into view if needed
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } else {
      card.classList.remove("active");
    }
  }
}

// =========================================================================
// Timeline drawing
// =========================================================================
function drawTimeline() {
  const W = canvas.width;
  const H = CELL_H;
  const HEADER = 16;
  const THUMB_Y = HEADER + 1;
  const THUMB_H = CELL_W - 4;
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
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.strokeRect(x + 2, THUMB_Y, THUMB_H, THUMB_H);

    // Env subgoal boundary (dashed yellow)
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

  // MeMER keyframe markers
  if (memerData) {
    for (const kf of memerData.keyframes) {
      const x = kf.idx * CELL_W + CELL_W / 2;

      // Pick color by primary provenance
      let color = "#608060"; // merged
      if (kf.provenance.includes("transition")) color = "#e07020";
      else if (kf.provenance.includes("local_minimum")) color = "#20a0c0";
      else if (kf.provenance.includes("midpoint")) color = "#b030b0";

      // Solid vertical line
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, HEADER);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.lineWidth = 1;

      // Diamond marker at top
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, HEADER);
      ctx.lineTo(x + 5, HEADER + 5);
      ctx.lineTo(x, HEADER + 10);
      ctx.lineTo(x - 5, HEADER + 5);
      ctx.closePath();
      ctx.fill();
    }
  }

  // Playhead
  if (timelineMeta.length > 0) {
    const px = currentFrame * CELL_W + CELL_W / 2;
    ctx.strokeStyle = "rgba(255,50,50,0.9)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();
    ctx.lineWidth = 1;

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
// Timeline click to seek
// =========================================================================
function onTlClick(e) {
  const rect = canvas.getBoundingClientRect();
  const xFixed = e.clientX - rect.left;
  const frame = Math.max(0, Math.min(timelineMeta.length - 1, Math.floor(xFixed / CELL_W)));
  video.currentTime = timeFromFrame(frame);
  currentFrame = frame;
  updateInfoBar();
  drawTimeline();
  updateKeyframeHighlight();
}

// =========================================================================
// Navigation
// =========================================================================
async function changeEpisode(delta) {
  const newIdx = epListIdx + delta;
  if (newIdx < 0 || newIdx >= episodes.length) return;
  epListIdx = newIdx;
  await loadEpisode();
}

// =========================================================================
// Boot
// =========================================================================
init();