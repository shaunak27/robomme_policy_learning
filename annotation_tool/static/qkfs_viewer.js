// QKFS Eval Viewer — interactive timeline + metrics

const SEG_COLORS = [
  '#66bb6a','#42a5f5','#ffa726','#ab47bc','#ef5350',
  '#26c6da','#ffee58','#ec407a','#8d6e63','#78909c',
  '#9ccc65','#5c6bc0','#ff7043','#26a69a','#d4e157',
];
const COMPLETED_COLOR = '#555';

let summaryData = {};
let currentTask = '';
let currentEpisode = -1;
let episodeInfo = null;   // {total_frames, segments}
let lastResult = null;    // last QKFS run result
let lastTargets = null;   // last targets API result

// Video playback state
const playbackState = {};  // keyed by stripId

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  summaryData = await fetchJSON('/api/summary');
  renderOverallTable(summaryData.overall || {});

  const tasks = await fetchJSON('/api/tasks');
  const sel = document.getElementById('task-select');
  tasks.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => loadTask(sel.value));

  document.getElementById('btn-prev-ep').addEventListener('click', () => stepEpisode(-1));
  document.getElementById('btn-next-ep').addEventListener('click', () => stepEpisode(1));
  document.getElementById('step-slider').addEventListener('input', onSliderChange);

  initPlayback();

  if (tasks.length > 0) loadTask(tasks[0]);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    console.warn(`fetchJSON ${url} → ${r.status}`);
    return { _error: r.status };
  }
  return r.json();
}

// ---------------------------------------------------------------------------
// Task / Episode loading
// ---------------------------------------------------------------------------
async function loadTask(task) {
  currentTask = task;
  document.getElementById('task-label').textContent = task;

  // Per-task metrics
  const taskMetrics = (summaryData.per_task || {})[task] || {};
  renderTaskTable(taskMetrics);

  // Load episodes
  const eps = await fetchJSON(`/api/${task}/episodes`);
  const sel = document.getElementById('episode-select');
  sel.innerHTML = '';
  eps.forEach(ep => {
    const obj = (typeof ep === 'object') ? ep : {id: ep, split: 'train'};
    const opt = document.createElement('option');
    opt.value = obj.id;
    const tag = obj.split === 'eval' ? ' [EVAL]' : '';
    opt.textContent = `Episode ${obj.id}${tag}`;
    if (obj.split === 'eval') opt.style.color = '#e67e22';
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => loadEpisode(parseInt(sel.value)));

  const firstEp = (typeof eps[0] === 'object') ? eps[0].id : eps[0];
  if (eps.length > 0) loadEpisode(firstEp);
}

async function loadEpisode(ep) {
  currentEpisode = ep;
  document.getElementById('episode-select').value = ep;
  setBadge('Loading...');

  episodeInfo = await fetchJSON(`/api/${currentTask}/${ep}/episode_info`);
  if (episodeInfo.error) {
    setBadge('No data');
    return;
  }

  renderSegmentBar(episodeInfo);
  setupSlider(episodeInfo);

  // Auto-run at midpoint of an exec segment with sources
  const segs = episodeInfo.segments;
  let defaultStep = Math.floor(episodeInfo.total_frames / 2);
  for (const seg of segs) {
    if (seg.phase === 'exec' && !isCompleted(seg.label) && seg.idx > 0) {
      defaultStep = Math.floor((seg.start_frame + seg.end_frame) / 2);
      break;
    }
  }
  document.getElementById('step-slider').value = defaultStep;
  runAtStep(defaultStep);
}

function stepEpisode(dir) {
  const sel = document.getElementById('episode-select');
  const opts = Array.from(sel.options);
  const idx = opts.findIndex(o => o.selected);
  const next = idx + dir;
  if (next >= 0 && next < opts.length) {
    sel.selectedIndex = next;
    loadEpisode(parseInt(opts[next].value));
  }
}

// ---------------------------------------------------------------------------
// Segment bar
// ---------------------------------------------------------------------------
function renderSegmentBar(info) {
  const bar = document.getElementById('segment-bar');
  const labels = document.getElementById('segment-labels');
  bar.innerHTML = '';
  labels.innerHTML = '';

  const total = info.total_frames;
  const segs = info.segments;

  segs.forEach((seg, i) => {
    const width = ((seg.end_frame - seg.start_frame + 1) / total) * 100;
    const block = document.createElement('div');
    block.className = 'seg-block';
    block.style.width = width + '%';
    const color = isCompleted(seg.label) ? COMPLETED_COLOR : SEG_COLORS[seg.idx % SEG_COLORS.length];
    block.style.background = color;
    block.dataset.segIdx = seg.idx;
    block.title = `Seg ${seg.idx}: ${seg.label} (${seg.start_frame}-${seg.end_frame})`;

    // Truncated label
    const shortLabel = seg.label.length > 15 ? seg.label.slice(0, 14) + '\u2026' : seg.label;
    if (width > 8) block.textContent = shortLabel;

    bar.appendChild(block);

    // Label below
    const lbl = document.createElement('div');
    lbl.className = 'seg-label-item';
    lbl.style.width = width + '%';
    lbl.textContent = width > 5 ? `${seg.start_frame}` : '';
    labels.appendChild(lbl);
  });

  document.getElementById('ep-info-label').textContent =
    `(${total} frames, ${segs.length} segments)`;
}

function highlightSourceSegments(sourceIdxs) {
  const blocks = document.querySelectorAll('.seg-block');
  const sourceSet = new Set(sourceIdxs);
  blocks.forEach(b => {
    const idx = parseInt(b.dataset.segIdx);
    b.style.opacity = sourceSet.has(idx) ? '1.0' : '0.25';
  });
}

// ---------------------------------------------------------------------------
// Step slider + QKFS execution
// ---------------------------------------------------------------------------
function setupSlider(info) {
  const slider = document.getElementById('step-slider');
  slider.min = 1;
  slider.max = info.total_frames - 1;
}

function onSliderChange() {
  const step = parseInt(document.getElementById('step-slider').value);
  document.getElementById('step-value').textContent = step;
  // Debounce: only fire after 200ms pause
  clearTimeout(window._sliderTimer);
  window._sliderTimer = setTimeout(() => runAtStep(step), 200);
}

async function runAtStep(step) {
  document.getElementById('step-value').textContent = step;
  setBadge('Running QKFS...');

  // Fetch QKFS results and training targets in parallel
  const [result, targets] = await Promise.all([
    fetchJSON(`/api/${currentTask}/${currentEpisode}/run_qkfs?step=${step}`),
    fetchJSON(`/api/${currentTask}/${currentEpisode}/targets?step=${step}`),
  ]);
  lastResult = result;
  lastTargets = targets;

  highlightSourceSegments(result.source_seg_indices || []);
  renderTimeline(result, targets);
  renderStepMetrics(result);
  renderCurrentObs(result);
  renderTargetStrip(targets);
  renderFrameStrips(result);
  setBadge('Ready');
}

// ---------------------------------------------------------------------------
// Timeline canvas
// ---------------------------------------------------------------------------
function renderTimeline(result, targets) {
  const canvas = document.getElementById('timeline-canvas');
  const container = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;

  const H = 220;
  canvas.width = container.clientWidth * dpr;
  canvas.height = H * dpr;
  canvas.style.width = container.clientWidth + 'px';
  canvas.style.height = H + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = container.clientWidth;

  ctx.clearRect(0, 0, W, H);

  if (!episodeInfo) return;
  const total = episodeInfo.total_frames;
  const segs = episodeInfo.segments;
  const sourceSet = new Set(result.source_seg_indices || []);

  const xScale = (frame) => (frame / total) * (W - 40) + 20;
  const rowY = { target: 40, qkfs: 70, uniform: 100, recency: 130 };
  const labelY = { target: 32, qkfs: 62, uniform: 92, recency: 122 };

  // Draw segment backgrounds
  segs.forEach(seg => {
    const x0 = xScale(seg.start_frame);
    const x1 = xScale(seg.end_frame + 1);
    const color = isCompleted(seg.label) ? COMPLETED_COLOR : SEG_COLORS[seg.idx % SEG_COLORS.length];
    const isSource = sourceSet.has(seg.idx);
    ctx.fillStyle = color;
    ctx.globalAlpha = isSource ? 0.35 : 0.08;
    ctx.fillRect(x0, 20, x1 - x0, H - 40);
    ctx.globalAlpha = 1.0;

    // Segment boundary lines
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x0, 20);
    ctx.lineTo(x0, H - 20);
    ctx.stroke();
  });

  // TOPReward windows (shaded background regions)
  if (targets && targets.topreward_info) {
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = '#FFD700';
    Object.values(targets.topreward_info).forEach(tr => {
      const x0 = xScale(tr.window_start);
      const x1 = xScale(tr.window_end);
      ctx.fillRect(x0, 20, x1 - x0, H - 40);
    });
    ctx.globalAlpha = 1.0;
  }

  // Row labels
  ctx.fillStyle = '#4CAF50'; ctx.font = '11px sans-serif';
  ctx.fillText('Target', 0, labelY.target);
  ctx.fillStyle = '#4db8ff';
  ctx.fillText('QKFS', 0, labelY.qkfs);
  ctx.fillStyle = '#999';
  ctx.fillText('Unif', 0, labelY.uniform);
  ctx.fillStyle = '#FF9800';
  ctx.fillText('Rec', 0, labelY.recency);

  // Draw selections
  function drawSelections(frames, y, color, height) {
    ctx.fillStyle = color;
    frames.forEach(fi => {
      const x = xScale(fi);
      ctx.fillRect(x - 1, y - height/2, 2, height);
    });
  }

  // Target frames row
  if (targets && targets.target_frames) {
    drawSelections(targets.target_frames, rowY.target, '#4CAF50', 20);
  }

  drawSelections(result.qkfs || [], rowY.qkfs, '#2196F3', 24);
  drawSelections(result.uniform || [], rowY.uniform, '#999', 18);
  drawSelections(result.recency || [], rowY.recency, '#FF9800', 18);

  // Draw density keyframe markers (cyan diamonds on the target row)
  if (targets && targets.density_keyframes_abs) {
    ctx.fillStyle = '#00E5FF';
    Object.values(targets.density_keyframes_abs).flat().forEach(fi => {
      const x = xScale(fi);
      ctx.beginPath();
      ctx.moveTo(x, rowY.target - 12);
      ctx.lineTo(x + 4, rowY.target - 8);
      ctx.lineTo(x, rowY.target - 4);
      ctx.lineTo(x - 4, rowY.target - 8);
      ctx.closePath();
      ctx.fill();
    });
  }

  // Draw TOPReward keyframe markers (gold triangles above target row)
  if (targets && targets.topreward_info) {
    ctx.fillStyle = '#FFD700';
    Object.values(targets.topreward_info).forEach(tr => {
      (tr.keyframes || []).forEach(fi => {
        const x = xScale(fi);
        ctx.beginPath();
        ctx.moveTo(x, 155);
        ctx.lineTo(x - 4, 165);
        ctx.lineTo(x + 4, 165);
        ctx.closePath();
        ctx.fill();
      });
    });
    // Labels for TR markers
    ctx.fillStyle = '#FFD700';
    ctx.font = '9px sans-serif';
    ctx.fillText('TR kf', W - 35, 163);
  }

  // Current timestep line
  const stepX = xScale(result.step_idx);
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.beginPath();
  ctx.moveTo(stepX, 15);
  ctx.lineTo(stepX, H - 15);
  ctx.stroke();
  ctx.setLineDash([]);

  // Step label
  ctx.fillStyle = '#e94560';
  ctx.font = 'bold 11px sans-serif';
  ctx.fillText(`t=${result.step_idx}`, stepX + 4, 14);

  // Frame axis
  ctx.fillStyle = '#666';
  ctx.font = '10px sans-serif';
  for (let f = 0; f <= total; f += Math.max(50, Math.round(total / 10))) {
    const x = xScale(f);
    ctx.fillText(f, x - 8, H - 5);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, H - 20);
    ctx.lineTo(x, H - 16);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Step metrics table
// ---------------------------------------------------------------------------
function renderStepMetrics(result) {
  const tbody = document.querySelector('#step-metrics-table tbody');
  tbody.innerHTML = '';

  const metrics = result.metrics || {};
  const methods = [
    { key: 'qkfs', label: 'QKFS', cls: 'qkfs-row' },
    { key: 'uniform', label: 'Uniform', cls: 'uniform-row' },
    { key: 'recency', label: 'Recency', cls: 'recency-row' },
  ];

  methods.forEach(m => {
    const d = metrics[m.key] || {};
    const nSelected = (result[m.key] || []).length;
    const inSource = Math.round((d.precision || 0) * nSelected);
    const tr = document.createElement('tr');
    tr.className = m.cls;
    tr.innerHTML = `
      <td>${m.label}</td>
      <td>${pct(d.precision)}</td>
      <td>${pct(d.recall)}</td>
      <td>${inSource} / ${nSelected}</td>
    `;
    tbody.appendChild(tr);
  });

  // Source info
  const sourceInfo = document.getElementById('source-info');
  const srcSegs = result.source_seg_indices || [];
  if (srcSegs.length > 0 && episodeInfo) {
    const segLabels = srcSegs.map(idx => {
      const seg = episodeInfo.segments.find(s => s.idx === idx);
      return seg ? `seg${idx} "${seg.label}"` : `seg${idx}`;
    });
    sourceInfo.innerHTML = `<strong>Rule sources (${srcSegs.length}):</strong> ${segLabels.join(', ')}`;
  } else {
    sourceInfo.innerHTML = '<em>No source subtasks at this timestep</em>';
  }
}

// ---------------------------------------------------------------------------
// Summary tables
// ---------------------------------------------------------------------------
function renderOverallTable(overall) {
  const tbody = document.querySelector('#overall-table tbody');
  tbody.innerHTML = '';
  ['qkfs', 'uniform', 'recency'].forEach(method => {
    const d = overall[method] || {};
    const cls = method + '-row';
    const tr = document.createElement('tr');
    tr.className = cls;
    tr.innerHTML = `
      <td>${method.toUpperCase()}</td>
      <td>${pct(d.precision)}</td>
      <td>${pct(d.recall)}</td>
      <td>${fmt(d.correlation)}</td>
      <td>${fmt(d.temporal_std)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTaskTable(taskData) {
  const tbody = document.querySelector('#task-table tbody');
  tbody.innerHTML = '';
  ['qkfs', 'uniform', 'recency'].forEach(method => {
    const d = taskData[method] || {};
    const cls = method + '-row';
    const tr = document.createElement('tr');
    tr.className = cls;
    tr.innerHTML = `
      <td>${method.toUpperCase()}</td>
      <td>${pct(d.precision)}</td>
      <td>${pct(d.recall)}</td>
      <td>${fmt(d.correlation)}</td>
      <td>${fmt(d.temporal_std)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Frame thumbnails
// ---------------------------------------------------------------------------

function renderCurrentObs(result) {
  const container = document.getElementById('current-obs-frame');
  const label = document.getElementById('current-t-label');
  container.innerHTML = '';
  label.textContent = result.step_idx;

  const img = document.createElement('img');
  img.src = `/api/${currentTask}/${currentEpisode}/${result.step_idx}/frame`;
  img.title = `Current observation t=${result.step_idx}`;
  container.appendChild(img);
}

function renderFrameStrips(result) {
  const sourceSet = new Set(result.source_seg_indices || []);
  const segs = episodeInfo ? episodeInfo.segments : [];

  renderOneStrip('qkfs-strip', 'qkfs-strip-info', result.qkfs || [], sourceSet, segs, result.metrics?.qkfs);
  renderOneStrip('uniform-strip', 'uniform-strip-info', result.uniform || [], sourceSet, segs, result.metrics?.uniform);
  renderOneStrip('recency-strip', 'recency-strip-info', result.recency || [], sourceSet, segs, result.metrics?.recency);
}

function renderOneStrip(stripId, infoId, frameIndices, sourceSet, segments, metrics) {
  const strip = document.getElementById(stripId);
  const info = document.getElementById(infoId);
  strip.innerHTML = '';

  if (!frameIndices.length) {
    info.textContent = '(no frames)';
    return;
  }

  // Info label
  const p = metrics ? metrics.precision : 0;
  const r = metrics ? metrics.recall : 0;
  info.textContent = `(${frameIndices.length} frames, prec=${(p*100).toFixed(0)}%, recall=${(r*100).toFixed(0)}%)`;

  // Create thumbnails
  frameIndices.forEach(fi => {
    const seg = findSegment(fi, segments);
    const inSource = seg && sourceSet.has(seg.idx);
    const segColor = seg ? (isCompleted(seg.label) ? COMPLETED_COLOR : SEG_COLORS[seg.idx % SEG_COLORS.length]) : '#333';

    const div = document.createElement('div');
    div.className = 'frame-thumb ' + (inSource ? 'in-source' : 'not-in-source');
    div.title = `Frame ${fi}` + (seg ? ` | seg${seg.idx}: "${seg.label}"` : '') + (inSource ? ' [SOURCE]' : '');

    const img = document.createElement('img');
    img.src = `/api/${currentTask}/${currentEpisode}/${fi}/frame`;
    img.loading = 'lazy';
    div.appendChild(img);

    // Segment color indicator bar
    const bar = document.createElement('div');
    bar.className = 'seg-indicator';
    bar.style.background = segColor;
    div.appendChild(bar);

    // Frame index label
    const lbl = document.createElement('div');
    lbl.className = 'frame-label';
    lbl.textContent = `f${fi}`;
    if (seg) lbl.textContent += ` s${seg.idx}`;
    div.appendChild(lbl);

    strip.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// Target frames strip (training signal)
// ---------------------------------------------------------------------------
function renderTargetStrip(targets) {
  const strip = document.getElementById('target-strip');
  const info = document.getElementById('target-strip-info');
  strip.innerHTML = '';

  if (!targets || !targets.has_target || !targets.target_frames || targets.target_frames.length === 0) {
    info.textContent = '(no target at this timestep)';
    return;
  }

  const frameIndices = targets.target_frames;
  const segs = episodeInfo ? episodeInfo.segments : [];
  const sourceSet = new Set(targets.source_seg_indices || []);

  // Collect all topreward keyframes and density keyframes into sets for quick lookup
  const trKeyframes = new Set();
  if (targets.topreward_info) {
    Object.values(targets.topreward_info).forEach(tr => {
      (tr.keyframes || []).forEach(kf => trKeyframes.add(kf));
    });
  }
  const dkKeyframes = new Set();
  if (targets.density_keyframes_abs) {
    Object.values(targets.density_keyframes_abs).forEach(kfs => {
      kfs.forEach(kf => dkKeyframes.add(kf));
    });
  }

  info.textContent = `(${frameIndices.length} frames from target distribution)`;

  frameIndices.forEach(fi => {
    const seg = findSegment(fi, segs);
    const inSource = seg && sourceSet.has(seg.idx);
    const segColor = seg ? (isCompleted(seg.label) ? COMPLETED_COLOR : SEG_COLORS[seg.idx % SEG_COLORS.length]) : '#333';

    const div = document.createElement('div');
    div.className = 'frame-thumb target-frame';
    if (inSource) div.classList.add('in-source');

    // Highlight TOPReward and density keyframes
    const isTR = trKeyframes.has(fi);
    const isDK = dkKeyframes.has(fi);
    if (isTR) div.classList.add('topreward-kf');
    else if (isDK) div.classList.add('density-kf');

    let title = `Frame ${fi}`;
    if (seg) title += ` | seg${seg.idx}: "${seg.label}"`;
    if (isTR) title += ' [TOPReward KF]';
    if (isDK) title += ' [Density KF]';
    div.title = title;

    const img = document.createElement('img');
    img.src = `/api/${currentTask}/${currentEpisode}/${fi}/frame`;
    img.loading = 'lazy';
    div.appendChild(img);

    // Segment color indicator bar
    const bar = document.createElement('div');
    bar.className = 'seg-indicator';
    bar.style.background = segColor;
    div.appendChild(bar);

    // Frame index label
    const lbl = document.createElement('div');
    lbl.className = 'frame-label';
    lbl.textContent = `f${fi}`;
    if (seg) lbl.textContent += ` s${seg.idx}`;
    if (isTR) lbl.textContent += ' TR';
    if (isDK) lbl.textContent += ' DK';
    div.appendChild(lbl);

    strip.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// Video playback
// ---------------------------------------------------------------------------
function initPlayback() {
  document.querySelectorAll('.play-btn').forEach(btn => {
    const stripId = btn.dataset.strip;
    playbackState[stripId] = { playing: false, timer: null, idx: 0 };

    btn.addEventListener('click', () => togglePlayback(stripId));
  });

  document.querySelectorAll('.speed-select').forEach(sel => {
    sel.addEventListener('change', () => {
      const stripId = sel.dataset.strip;
      if (playbackState[stripId] && playbackState[stripId].playing) {
        stopPlayback(stripId);
        startPlayback(stripId);
      }
    });
  });

  // Close overlay on click or ESC
  const overlay = document.getElementById('playback-overlay');
  overlay.addEventListener('click', closePlaybackOverlay);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closePlaybackOverlay();
  });
}

function togglePlayback(stripId) {
  const state = playbackState[stripId];
  if (state.playing) {
    stopPlayback(stripId);
    closePlaybackOverlay();
  } else {
    startPlayback(stripId);
  }
}

function startPlayback(stripId) {
  const state = playbackState[stripId];
  const strip = document.getElementById(stripId);
  const thumbs = strip.querySelectorAll('.frame-thumb');
  if (thumbs.length === 0) return;

  state.playing = true;
  state.idx = 0;
  const btn = document.querySelector(`.play-btn[data-strip="${stripId}"]`);
  btn.classList.add('playing');
  btn.innerHTML = '&#9724; Stop';

  const speedSel = document.querySelector(`.speed-select[data-strip="${stripId}"]`);
  const interval = parseInt(speedSel.value);

  const overlay = document.getElementById('playback-overlay');
  overlay.classList.add('active');

  function tick() {
    if (!state.playing) return;

    // Clear previous highlight
    thumbs.forEach(t => t.classList.remove('playback-active'));

    if (state.idx >= thumbs.length) {
      stopPlayback(stripId);
      closePlaybackOverlay();
      return;
    }

    const thumb = thumbs[state.idx];
    thumb.classList.add('playback-active');
    thumb.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });

    // Show in overlay
    const img = thumb.querySelector('img');
    const overlayImg = document.getElementById('playback-img');
    const overlayLabel = document.getElementById('playback-label');
    overlayImg.src = img.src;
    overlayLabel.textContent = thumb.title || `Frame ${state.idx}`;

    state.idx++;
    state.timer = setTimeout(tick, interval);
  }

  tick();
}

function stopPlayback(stripId) {
  const state = playbackState[stripId];
  state.playing = false;
  if (state.timer) { clearTimeout(state.timer); state.timer = null; }

  const btn = document.querySelector(`.play-btn[data-strip="${stripId}"]`);
  btn.classList.remove('playing');
  btn.innerHTML = '&#9654; Play';

  const strip = document.getElementById(stripId);
  strip.querySelectorAll('.frame-thumb').forEach(t => t.classList.remove('playback-active'));
}

function closePlaybackOverlay() {
  const overlay = document.getElementById('playback-overlay');
  overlay.classList.remove('active');
  // Stop all active playbacks
  Object.keys(playbackState).forEach(stripId => {
    if (playbackState[stripId].playing) stopPlayback(stripId);
  });
}

function findSegment(frameIdx, segments) {
  for (const seg of segments) {
    if (seg.start_frame <= frameIdx && frameIdx <= seg.end_frame) return seg;
  }
  return null;
}

// Collapsible sections
document.addEventListener('click', e => {
  const h3 = e.target.closest('h3.collapsible');
  if (!h3) return;
  const section = h3.closest('.frame-strip-section');
  if (section) section.classList.toggle('collapsed');
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function isCompleted(label) {
  return label.includes('all tasks completed') || label === 'complete';
}

function pct(v) {
  return v != null ? (v * 100).toFixed(1) + '%' : '-';
}

function fmt(v) {
  return v != null ? v.toFixed(3) : '-';
}

function setBadge(text) {
  document.getElementById('status-badge').textContent = text;
}

// Resize handler
window.addEventListener('resize', () => {
  if (lastResult) renderTimeline(lastResult, lastTargets);
});

init();
