/* RoboMME Subtask Taxonomy Viewer */

const SUBTASK_COLORS = [
  '#4ecca3', '#e06c75', '#61afef', '#c678dd', '#e5c07b',
  '#56b6c2', '#be5046', '#98c379', '#d19a66', '#c882e7',
  '#6ba3e8', '#f0a170', '#e57373', '#81c784', '#64b5f6',
  '#ffb74d', '#a1887f', '#90a4ae', '#f06292', '#aed581',
];

let taskList = [];
let currentTask = null;
let currentPhase = 'exec';

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', async () => {
  const tasks = await fetchJSON('/api/tasks');
  taskList = tasks;
  renderTaskSidebar(tasks);

  // Compute totals
  const totalEps = tasks.reduce((s, t) => s + t.num_episodes, 0);
  document.getElementById('stat-tasks').textContent = `${tasks.length} tasks`;
  document.getElementById('stat-episodes').textContent = `${totalEps} episodes`;
});

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

// ---------- Sidebar ----------
function renderTaskSidebar(tasks) {
  const container = document.getElementById('task-list');
  container.innerHTML = '';
  for (const t of tasks) {
    const div = document.createElement('div');
    div.className = 'task-item';
    div.dataset.name = t.name;
    div.innerHTML = `
      <div>${t.name}${t.has_video_demo ? '<span class="demo-badge">VIDEO</span>' : ''}${t.needs_review ? '<span class="review-badge">REVIEW</span>' : ''}</div>
      <div class="task-meta">${t.num_exec_subtasks} subtasks &middot; ${t.num_episodes} eps</div>
    `;
    div.addEventListener('click', () => selectTask(t.name));
    container.appendChild(div);
  }
}

async function selectTask(taskName) {
  // Update sidebar active
  document.querySelectorAll('.task-item').forEach(el => {
    el.classList.toggle('active', el.dataset.name === taskName);
  });

  const data = await fetchJSON(`/api/task/${taskName}`);
  currentTask = data;
  currentPhase = 'exec';
  renderDetail(data);
}

// ---------- Detail panel ----------
function renderDetail(data) {
  const panel = document.getElementById('detail-panel');
  panel.innerHTML = '';

  // Header
  const header = document.createElement('div');
  header.className = 'task-header';
  header.innerHTML = `
    <h2>${data.task_name}</h2>
    <span class="episode-count">${data.num_episodes} episodes</span>
    <div class="goal-text">${data.task_goal}</div>
  `;
  panel.appendChild(header);

  // Phase tabs (if demo exists)
  if (data.has_video_demo && data.demo) {
    const tabs = document.createElement('div');
    tabs.className = 'phase-tabs';
    tabs.innerHTML = `
      <div class="phase-tab exec-tab ${currentPhase === 'exec' ? 'active' : ''}" data-phase="exec">Execution</div>
      <div class="phase-tab demo-tab ${currentPhase === 'demo' ? 'active' : ''}" data-phase="demo">Video Demo</div>
    `;
    tabs.querySelectorAll('.phase-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        currentPhase = tab.dataset.phase;
        renderDetail(data);
      });
    });
    panel.appendChild(tabs);
  }

  const phaseData = currentPhase === 'demo' && data.demo ? data.demo : data.exec;

  // Rules editor (always show for exec phase) — at top for easy reference
  if (currentPhase === 'exec') {
    panel.appendChild(renderRulesEditor(data.task_name, phaseData));
    // Applied sampling results
    const samplingCard = renderSamplingResults(data.task_name, phaseData);
    panel.appendChild(samplingCard);
    // Edit instructions
    panel.appendChild(renderEditInstructions(data.task_name));
    // Sampling density per category
    panel.appendChild(renderSamplingDensity(data.task_name));
    // Frame-level density results
    panel.appendChild(renderDensityResults(data.task_name, phaseData));
  }

  // Canonical sequence
  panel.appendChild(renderCanonicalSequences(phaseData));

  // Subtask table
  panel.appendChild(renderSubtaskTable(phaseData));

  // Position timeline
  panel.appendChild(renderPositionTimeline(phaseData));

  // Coverage chart
  panel.appendChild(renderCoverageChart(phaseData));

  // Transition graph
  panel.appendChild(renderTransitionGraph(phaseData));

  // Dependencies (exec only)
  if (currentPhase === 'exec' && phaseData.dependencies && phaseData.dependencies.length > 0) {
    panel.appendChild(renderDependencies(phaseData));
  }

}

// ---------- Canonical sequences ----------
function renderCanonicalSequences(phaseData) {
  const card = makeCard('Canonical Sequences', 'Most common subtask orderings across episodes');
  const seqs = phaseData.canonical_sequences || [];
  if (seqs.length === 0) {
    card.innerHTML += '<div class="no-data">No sequences found</div>';
    return card;
  }

  const subtaskLabels = Object.keys(phaseData.subtasks);
  const colorMap = buildColorMap(subtaskLabels);

  for (const seq of seqs.slice(0, 5)) {
    const flow = document.createElement('div');
    flow.className = 'sequence-flow';
    seq.sequence.forEach((label, i) => {
      if (i > 0) {
        const arrow = document.createElement('span');
        arrow.className = 'seq-arrow';
        arrow.textContent = '\u2192';
        flow.appendChild(arrow);
      }
      const item = document.createElement('span');
      item.className = 'seq-item';
      item.style.background = hexToRgba(colorMap[label] || '#888', 0.25);
      item.style.border = `1px solid ${hexToRgba(colorMap[label] || '#888', 0.5)}`;
      item.style.color = colorMap[label] || '#888';
      item.textContent = label;
      flow.appendChild(item);
    });
    const count = document.createElement('span');
    count.className = 'seq-count';
    count.textContent = `(${seq.count} episodes)`;
    flow.appendChild(count);
    card.appendChild(flow);
  }
  return card;
}

// ---------- Subtask table ----------
function renderSubtaskTable(phaseData) {
  const card = makeCard('Subtask Details', 'Per-subtask statistics sorted by typical position');
  const subtasks = phaseData.subtasks;
  const entries = Object.entries(subtasks).sort((a, b) => a[1].median_position - b[1].median_position);
  const colorMap = buildColorMap(Object.keys(subtasks));

  const table = document.createElement('table');
  table.className = 'subtask-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>#</th>
        <th>Subtask</th>
        <th>Frequency</th>
        <th>Avg Coverage</th>
        <th>Median Position</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement('tbody');

  entries.forEach(([label, stats], i) => {
    const freqClass = stats.frequency >= 0.9 ? 'freq-high' : stats.frequency >= 0.5 ? 'freq-medium' : 'freq-low';
    const coveragePct = (stats.avg_coverage * 100).toFixed(1);
    const positionPct = (stats.median_position * 100).toFixed(0);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:#666; font-size:12px;">${i + 1}</td>
      <td>
        <span class="subtask-label" style="border-left: 3px solid ${colorMap[label]}; padding-left: 8px;">
          ${label}
        </span>
      </td>
      <td>
        <span class="freq-badge ${freqClass}">${(stats.frequency * 100).toFixed(0)}%</span>
        <span style="font-size:11px; color:#666; margin-left:4px;">(${stats.episode_count} eps)</span>
      </td>
      <td>
        <span class="coverage-bar-bg">
          <span class="coverage-bar-fill" style="width:${Math.min(coveragePct, 100)}%; background:${colorMap[label]};"></span>
        </span>
        <span class="coverage-value">${coveragePct}%</span>
        <span style="font-size:10px; color:#666;">\u00b1${(stats.std_coverage * 100).toFixed(1)}%</span>
      </td>
      <td style="text-align:center; color:#aaa;">${positionPct}%</td>
    `;
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  card.appendChild(table);
  return card;
}

// ---------- Position timeline ----------
function renderPositionTimeline(phaseData) {
  const card = makeCard('Subtask Ordering', 'Median position of each subtask within the episode (left=start, right=end)');
  const subtasks = phaseData.subtasks;
  const entries = Object.entries(subtasks).sort((a, b) => a[1].median_position - b[1].median_position);
  const colorMap = buildColorMap(Object.keys(subtasks));

  const container = document.createElement('div');
  container.style.padding = '8px 0';

  // Scale labels
  const scaleRow = document.createElement('div');
  scaleRow.style.cssText = 'display:flex; margin-left:210px; margin-bottom:8px; font-size:10px; color:#666;';
  scaleRow.innerHTML = '<span>Start (0%)</span><span style="margin-left:auto;">End (100%)</span>';
  container.appendChild(scaleRow);

  entries.forEach(([label, stats]) => {
    const row = document.createElement('div');
    row.className = 'position-row';

    const name = document.createElement('div');
    name.className = 'pos-name';
    name.textContent = label;

    const track = document.createElement('div');
    track.className = 'pos-track';

    const dot = document.createElement('div');
    dot.className = 'pos-dot';
    dot.style.left = `${stats.median_position * 100}%`;
    dot.style.background = colorMap[label];
    dot.title = `Median: ${(stats.median_position * 100).toFixed(0)}%, Avg: ${(stats.avg_position * 100).toFixed(0)}%`;
    track.appendChild(dot);

    row.appendChild(name);
    row.appendChild(track);
    container.appendChild(row);
  });

  card.appendChild(container);
  return card;
}

// ---------- Coverage chart ----------
function renderCoverageChart(phaseData) {
  const card = makeCard('Frame Coverage Distribution', 'Average % of episode frames each subtask occupies');

  const subtasks = phaseData.subtasks;
  const entries = Object.entries(subtasks).sort((a, b) => b[1].avg_coverage - a[1].avg_coverage);
  const colorMap = buildColorMap(Object.keys(subtasks));

  const container = document.createElement('div');
  container.id = 'coverage-chart-container';

  const canvas = document.createElement('canvas');
  canvas.width = 800;
  canvas.height = 260;
  container.appendChild(canvas);
  card.appendChild(container);

  // Draw after append so dimensions are known
  requestAnimationFrame(() => drawCoverageChart(canvas, entries, colorMap));
  return card;
}

function drawCoverageChart(canvas, entries, colorMap) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const margin = { top: 20, right: 20, bottom: 80, left: 50 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  ctx.clearRect(0, 0, W, H);

  if (entries.length === 0) return;

  const maxVal = Math.max(...entries.map(([, s]) => s.avg_coverage)) * 1.15;
  const barW = Math.min(40, plotW / entries.length - 4);
  const gap = (plotW - barW * entries.length) / (entries.length + 1);

  // Grid lines
  ctx.strokeStyle = '#2a2a3a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = margin.top + plotH - (plotH * i / 4);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(W - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${(maxVal * i / 4 * 100).toFixed(0)}%`, margin.left - 6, y + 3);
  }

  // Bars
  entries.forEach(([label, stats], i) => {
    const x = margin.left + gap + i * (barW + gap);
    const barH = (stats.avg_coverage / maxVal) * plotH;
    const y = margin.top + plotH - barH;

    ctx.fillStyle = colorMap[label] || '#4ecca3';
    ctx.beginPath();
    roundedRect(ctx, x, y, barW, barH, 3);
    ctx.fill();

    // Error bar (std)
    const stdH = (stats.std_coverage / maxVal) * plotH;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    const cx = x + barW / 2;
    ctx.beginPath();
    ctx.moveTo(cx, y - stdH);
    ctx.lineTo(cx, y + stdH);
    ctx.moveTo(cx - 4, y - stdH);
    ctx.lineTo(cx + 4, y - stdH);
    ctx.moveTo(cx - 4, y + stdH);
    ctx.lineTo(cx + 4, y + stdH);
    ctx.stroke();

    // Label
    ctx.save();
    ctx.translate(x + barW / 2, margin.top + plotH + 8);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#aaa';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(truncate(label, 25), 0, 0);
    ctx.restore();
  });
}

// ---------- Transition graph ----------
function renderTransitionGraph(phaseData) {
  const card = makeCard('Transition Graph', 'Subtask-to-subtask transitions (thicker = more frequent)');

  const transitions = phaseData.transitions || [];
  if (transitions.length === 0) {
    card.innerHTML += '<div class="no-data">No transitions found</div>';
    return card;
  }

  const subtasks = Object.keys(phaseData.subtasks);
  const colorMap = buildColorMap(subtasks);

  // Sort subtasks by median position
  const sorted = [...subtasks].sort((a, b) =>
    (phaseData.subtasks[a].median_position || 0) - (phaseData.subtasks[b].median_position || 0)
  );

  const container = document.createElement('div');
  container.id = 'transition-graph-container';

  const nodeH = 32;
  const nodeW = 180;
  const nodeGap = 16;
  const graphW = 700;
  const leftPad = 20;
  const graphH = sorted.length * (nodeH + nodeGap) + 40;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', graphW);
  svg.setAttribute('height', graphH);
  svg.setAttribute('viewBox', `0 0 ${graphW} ${graphH}`);

  // Defs for arrowheads
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#666"/>
  </marker>`;
  svg.appendChild(defs);

  // Node positions
  const nodePositions = {};
  sorted.forEach((label, i) => {
    const y = 20 + i * (nodeH + nodeGap);
    nodePositions[label] = { x: leftPad, y, w: nodeW, h: nodeH };
  });

  // Draw edges
  const maxCount = Math.max(...transitions.map(t => t.count));
  transitions.forEach(t => {
    const from = nodePositions[t.from];
    const to = nodePositions[t.to];
    if (!from || !to) return;

    const strokeW = Math.max(1.5, (t.count / maxCount) * 6);
    const opacity = Math.max(0.3, t.count / maxCount);

    const x1 = from.x + from.w;
    const y1 = from.y + from.h / 2;
    const x2 = to.x + to.w;
    const y2 = to.y + to.h / 2;

    // Curved path going to the right then back
    const curveOffset = 40 + Math.abs(y2 - y1) * 0.15;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const d = `M ${x1} ${y1} C ${x1 + curveOffset} ${y1}, ${x2 + curveOffset} ${y2}, ${x2} ${y2}`;
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', colorMap[t.from] || '#666');
    path.setAttribute('stroke-width', strokeW);
    path.setAttribute('opacity', opacity);
    path.setAttribute('marker-end', 'url(#arrowhead)');

    // Tooltip
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${t.from} \u2192 ${t.to}: ${t.count} times`;
    path.appendChild(title);

    svg.appendChild(path);

    // Count label on edge
    const midX = x1 + curveOffset * 0.7;
    const midY = (y1 + y2) / 2;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', midX);
    text.setAttribute('y', midY);
    text.setAttribute('fill', '#888');
    text.setAttribute('font-size', '10');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.textContent = t.count;
    svg.appendChild(text);
  });

  // Draw nodes
  sorted.forEach(label => {
    const pos = nodePositions[label];
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', pos.x);
    rect.setAttribute('y', pos.y);
    rect.setAttribute('width', pos.w);
    rect.setAttribute('height', pos.h);
    rect.setAttribute('rx', 4);
    rect.setAttribute('fill', hexToRgba(colorMap[label] || '#888', 0.2));
    rect.setAttribute('stroke', colorMap[label] || '#888');
    rect.setAttribute('stroke-width', 1.5);
    svg.appendChild(rect);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x + 10);
    text.setAttribute('y', pos.y + pos.h / 2);
    text.setAttribute('fill', '#e0e0e0');
    text.setAttribute('font-size', '12');
    text.setAttribute('dominant-baseline', 'middle');
    text.textContent = truncate(label, 24);
    svg.appendChild(text);
  });

  container.appendChild(svg);
  card.appendChild(container);
  return card;
}

// ---------- Dependencies ----------
function renderDependencies(phaseData) {
  const card = makeCard('Dependencies', 'Subtask A always precedes subtask B (>= 95% of co-occurrences)');
  const deps = phaseData.dependencies || [];

  if (deps.length === 0) {
    card.innerHTML += '<div class="no-data">No strict ordering dependencies found</div>';
    return card;
  }

  const ul = document.createElement('ul');
  ul.className = 'dep-list';

  deps.sort((a, b) => b.strength - a.strength).forEach(dep => {
    const li = document.createElement('li');
    li.className = 'dep-item';
    li.innerHTML = `
      <span>${dep.before}</span>
      <span class="dep-arrow">\u2192</span>
      <span>${dep.after}</span>
      <span class="dep-strength">${(dep.strength * 100).toFixed(0)}% (n=${dep.co_occurrences})</span>
    `;
    ul.appendChild(li);
  });

  card.appendChild(ul);
  return card;
}

// ---------- Rules editor ----------
function renderRulesEditor(taskName, phaseData) {
  const card = makeCard(
    'Context Dependency Rules',
    'Which previously completed subtasks are necessary context for solving the current subtask? Write rules in free-form text.'
  );
  card.classList.add('rules-card');

  const rulesList = document.createElement('div');
  rulesList.className = 'rules-list';
  rulesList.id = 'rules-list';
  card.appendChild(rulesList);

  const actions = document.createElement('div');
  actions.className = 'rules-actions';
  actions.innerHTML = `
    <button class="btn-add" id="btn-add-rule">+ Add rule</button>
    <button class="btn-save" id="btn-save-rules">Save</button>
    <span class="rules-status" id="rules-status"></span>
  `;
  card.appendChild(actions);

  // Load existing rules then populate
  fetchJSON(`/api/rules/${taskName}`).then(data => {
    const rules = data.rules || [];
    if (rules.length === 0) {
      addRuleEntry(rulesList, '');
    } else {
      rules.forEach(r => addRuleEntry(rulesList, r));
    }
    if (data.updated) {
      const status = document.getElementById('rules-status');
      const when = new Date(data.updated);
      status.textContent = `Last saved: ${when.toLocaleString()}`;
    }
  });

  // Add rule button
  card.querySelector('#btn-add-rule').addEventListener('click', () => {
    addRuleEntry(rulesList, '');
    // Focus the new textarea
    const areas = rulesList.querySelectorAll('textarea');
    areas[areas.length - 1].focus();
  });

  // Save button
  card.querySelector('#btn-save-rules').addEventListener('click', async () => {
    const areas = rulesList.querySelectorAll('textarea');
    const rules = [];
    areas.forEach(ta => {
      const val = ta.value.trim();
      if (val) rules.push(val);
    });
    const res = await fetch(`/api/rules/${taskName}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules }),
    });
    if (res.ok) {
      const status = document.getElementById('rules-status');
      status.textContent = 'Saved!';
      status.classList.add('saved');
      setTimeout(() => {
        status.textContent = `Last saved: ${new Date().toLocaleString()}`;
        status.classList.remove('saved');
      }, 2000);
    }
  });

  return card;
}

function addRuleEntry(container, text) {
  const idx = container.querySelectorAll('.rule-entry').length + 1;
  const entry = document.createElement('div');
  entry.className = 'rule-entry';
  entry.innerHTML = `
    <span class="rule-idx">${idx}.</span>
    <textarea rows="2" placeholder="e.g. &quot;pick up third cube&quot; requires knowing that the first and second cubes have already been picked up and placed in the bin">${escapeHtml(text)}</textarea>
    <button class="rule-remove" title="Remove rule">&times;</button>
  `;
  entry.querySelector('.rule-remove').addEventListener('click', () => {
    entry.remove();
    renumberRules(container);
  });
  // Auto-grow textarea
  const ta = entry.querySelector('textarea');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = ta.scrollHeight + 'px';
  });
  container.appendChild(entry);
  // Trigger auto-grow for pre-filled text
  if (text) {
    ta.style.height = 'auto';
    ta.style.height = ta.scrollHeight + 'px';
  }
}

function renumberRules(container) {
  container.querySelectorAll('.rule-entry').forEach((entry, i) => {
    entry.querySelector('.rule-idx').textContent = `${i + 1}.`;
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------- Sampling results ----------
function renderSamplingResults(taskName, phaseData) {
  const card = makeCard(
    'Applied Sampling Rules',
    'How the rules map to concrete episodes — which past segments each exec subtask samples from'
  );
  card.classList.add('sampling-card');

  const container = document.createElement('div');
  container.id = 'sampling-results';
  container.innerHTML = '<div class="no-data" style="padding:16px;">Loading...</div>';
  card.appendChild(container);

  fetchJSON(`/api/sampling/${taskName}`).then(episodes => {
    container.innerHTML = '';
    if (episodes.length === 0) {
      container.innerHTML = '<div class="no-data" style="padding:16px;">No sampling results yet. Run scripts/apply_rules_programmatic.py first.</div>';
      return;
    }

    const subtaskLabels = Object.keys(phaseData.subtasks);
    const colorMap = buildColorMap(subtaskLabels);

    // Episode selector
    const controls = document.createElement('div');
    controls.className = 'sampling-controls';
    const selector = document.createElement('select');
    selector.id = 'sampling-ep-select';
    episodes.forEach((ep, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `Episode ${ep.episode_idx}`;
      selector.appendChild(opt);
    });
    controls.appendChild(selector);

    const navLeft = document.createElement('button');
    navLeft.textContent = '\u25C0';
    navLeft.addEventListener('click', () => {
      selector.value = Math.max(0, parseInt(selector.value) - 1);
      selector.dispatchEvent(new Event('change'));
    });
    const navRight = document.createElement('button');
    navRight.textContent = '\u25B6';
    navRight.addEventListener('click', () => {
      selector.value = Math.min(episodes.length - 1, parseInt(selector.value) + 1);
      selector.dispatchEvent(new Event('change'));
    });
    controls.insertBefore(navLeft, selector);
    controls.appendChild(navRight);

    const epCount = document.createElement('span');
    epCount.style.cssText = 'font-size:12px; color:#888; margin-left:8px;';
    epCount.textContent = `${episodes.length} episodes`;
    controls.appendChild(epCount);

    container.appendChild(controls);

    const display = document.createElement('div');
    display.id = 'sampling-display';
    container.appendChild(display);

    function renderEpisode(idx) {
      const ep = episodes[idx];
      display.innerHTML = '';

      const goalDiv = document.createElement('div');
      goalDiv.className = 'sampling-goal';
      goalDiv.textContent = ep.task_goal;
      display.appendChild(goalDiv);

      // Segment timeline
      const timeline = document.createElement('div');
      timeline.className = 'sampling-timeline';

      const segments = ep.segments;
      const totalFrames = segments[segments.length - 1].end_frame + 1;

      segments.forEach(seg => {
        const bar = document.createElement('div');
        bar.className = `sampling-seg ${seg.phase}`;
        const width = ((seg.num_frames || (seg.end_frame - seg.start_frame + 1)) / totalFrames * 100);
        bar.style.width = `${Math.max(width, 0.5)}%`;

        // Color by label
        const c = colorMap[seg.label] || '#666';
        bar.style.background = seg.phase === 'demo'
          ? hexToRgba(c, 0.3)
          : hexToRgba(c, 0.6);
        bar.style.borderLeft = `2px solid ${c}`;
        bar.title = `[${seg.idx}] ${seg.phase}: ${seg.label} (${seg.start_frame}-${seg.end_frame})`;
        timeline.appendChild(bar);
      });
      display.appendChild(timeline);

      // Legend for timeline
      const legend = document.createElement('div');
      legend.style.cssText = 'display:flex; gap:12px; font-size:10px; color:#666; margin-bottom:10px;';
      legend.innerHTML = '<span>Lighter = demo</span><span>Darker = exec</span>';
      display.appendChild(legend);

      // Mapping table
      const map = ep.sampling_map;
      if (!map || Object.keys(map).length === 0) {
        display.innerHTML += '<div class="no-data" style="padding:10px;">No mapping data</div>';
        return;
      }

      const table = document.createElement('div');
      table.className = 'sampling-map';

      // Header
      const headerRow = document.createElement('div');
      headerRow.className = 'smap-header';
      headerRow.innerHTML = `
        <span class="smap-col-target">Exec Subtask</span>
        <span class="smap-col-arrow"></span>
        <span class="smap-col-sources">Samples From</span>
      `;
      table.appendChild(headerRow);

      Object.keys(map).sort((a, b) => parseInt(a) - parseInt(b)).forEach(segIdxStr => {
        const segIdx = parseInt(segIdxStr);
        const seg = segments.find(s => s.idx === segIdx);
        if (!seg) return;

        const sources = map[segIdxStr];
        const row = document.createElement('div');
        row.className = 'smap-row';

        // Target segment
        const targetCol = document.createElement('div');
        targetCol.className = 'smap-col-target';
        const targetChip = document.createElement('span');
        targetChip.className = 'smap-chip target';
        const tc = colorMap[seg.label] || '#888';
        targetChip.style.borderColor = tc;
        targetChip.style.background = hexToRgba(tc, 0.15);
        targetChip.innerHTML = `<span class="smap-idx">[${seg.idx}]</span> ${seg.label}`;
        targetCol.appendChild(targetChip);
        row.appendChild(targetCol);

        // Arrow
        const arrowCol = document.createElement('div');
        arrowCol.className = 'smap-col-arrow';
        arrowCol.textContent = sources.length > 0 ? '\u2190' : '';
        row.appendChild(arrowCol);

        // Source segments
        const sourcesCol = document.createElement('div');
        sourcesCol.className = 'smap-col-sources';
        if (sources.length === 0) {
          const noMem = document.createElement('span');
          noMem.className = 'smap-no-memory';
          noMem.textContent = 'no memory';
          sourcesCol.appendChild(noMem);
        } else {
          sources.forEach(srcIdx => {
            const srcSeg = segments.find(s => s.idx === srcIdx);
            if (!srcSeg) return;
            const chip = document.createElement('span');
            chip.className = `smap-chip source ${srcSeg.phase}`;
            const sc = colorMap[srcSeg.label] || '#888';
            chip.style.borderColor = sc;
            chip.style.background = hexToRgba(sc, srcSeg.phase === 'demo' ? 0.08 : 0.15);
            chip.innerHTML = `<span class="smap-idx">[${srcIdx}]</span><span class="smap-phase">${srcSeg.phase}</span>${srcSeg.label}`;
            sourcesCol.appendChild(chip);
          });
        }
        row.appendChild(sourcesCol);
        table.appendChild(row);
      });

      display.appendChild(table);
    }

    selector.addEventListener('change', () => renderEpisode(parseInt(selector.value)));
    renderEpisode(0);
  });

  return card;
}

// ---------- Edit instructions ----------
function renderEditInstructions(taskName) {
  const card = makeCard(
    'Edit Instructions',
    'After reviewing the applied rules above, note any corrections or adjustments here'
  );
  card.classList.add('edit-card');

  const textarea = document.createElement('textarea');
  textarea.id = 'edit-instructions-text';
  textarea.className = 'edit-textarea';
  textarea.rows = 4;
  textarea.placeholder = 'e.g. "pick up container should NOT include the static demo segment" or "press button should also get all prior exec segments"';
  card.appendChild(textarea);

  const actions = document.createElement('div');
  actions.className = 'rules-actions';
  actions.innerHTML = `
    <button class="btn-save" id="btn-save-edits">Save</button>
    <span class="rules-status" id="edit-status"></span>
  `;
  card.appendChild(actions);

  // Load existing
  fetchJSON(`/api/edit_instructions/${taskName}`).then(data => {
    textarea.value = data.text || '';
    if (data.updated) {
      const status = document.getElementById('edit-status');
      status.textContent = `Last saved: ${new Date(data.updated).toLocaleString()}`;
    }
    // Auto-grow
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  });

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  });

  card.querySelector('#btn-save-edits').addEventListener('click', async () => {
    const res = await fetch(`/api/edit_instructions/${taskName}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: textarea.value }),
    });
    if (res.ok) {
      const status = document.getElementById('edit-status');
      status.textContent = 'Saved!';
      status.classList.add('saved');
      setTimeout(() => {
        status.textContent = `Last saved: ${new Date().toLocaleString()}`;
        status.classList.remove('saved');
      }, 2000);
    }
  });

  return card;
}

// ---------- Density results (frame-level) ----------
function renderDensityResults(taskName, phaseData) {
  const card = makeCard(
    'Frame Selection Preview',
    'Which exact frames are selected for each exec subtask — verify the density rules produce correct selections'
  );
  card.classList.add('density-results-card');

  const container = document.createElement('div');
  container.id = 'density-results';
  container.innerHTML = '<div class="no-data" style="padding:16px;">Loading...</div>';
  card.appendChild(container);

  fetchJSON(`/api/density/${taskName}`).then(episodes => {
    container.innerHTML = '';
    if (episodes.length === 0) {
      container.innerHTML = '<div class="no-data" style="padding:16px;">No density results yet. Run scripts/apply_density_rules.py first.</div>';
      return;
    }

    const subtaskLabels = Object.keys(phaseData.subtasks);
    const colorMap = buildColorMap(subtaskLabels);

    // Episode selector with nav
    const controls = document.createElement('div');
    controls.className = 'sampling-controls';
    const selector = document.createElement('select');
    selector.id = 'density-ep-select';
    episodes.forEach((ep, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `Episode ${ep.episode_idx}`;
      selector.appendChild(opt);
    });
    controls.appendChild(selector);

    const navLeft = document.createElement('button');
    navLeft.textContent = '\u25C0';
    navLeft.addEventListener('click', () => {
      selector.value = Math.max(0, parseInt(selector.value) - 1);
      selector.dispatchEvent(new Event('change'));
    });
    const navRight = document.createElement('button');
    navRight.textContent = '\u25B6';
    navRight.addEventListener('click', () => {
      selector.value = Math.min(episodes.length - 1, parseInt(selector.value) + 1);
      selector.dispatchEvent(new Event('change'));
    });
    controls.insertBefore(navLeft, selector);
    controls.appendChild(navRight);

    const epCount = document.createElement('span');
    epCount.style.cssText = 'font-size:12px; color:#888; margin-left:8px;';
    epCount.textContent = `${episodes.length} episodes`;
    controls.appendChild(epCount);

    container.appendChild(controls);

    const display = document.createElement('div');
    display.id = 'density-display';
    container.appendChild(display);

    function renderEpisode(idx) {
      const ep = episodes[idx];
      display.innerHTML = '';

      const goalDiv = document.createElement('div');
      goalDiv.className = 'sampling-goal';
      goalDiv.textContent = ep.task_goal;
      display.appendChild(goalDiv);

      const segments = ep.segments;
      const totalFrames = segments[segments.length - 1].end_frame + 1;
      const frameSelections = ep.frame_selections || {};
      const keyframesBySeg = ep.keyframes_by_seg || {};

      // Per exec segment: show timeline with selected frames highlighted
      Object.keys(frameSelections).sort((a, b) => parseInt(a) - parseInt(b)).forEach(segIdxStr => {
        const segIdx = parseInt(segIdxStr);
        const seg = segments.find(s => s.idx === segIdx);
        if (!seg) return;

        const sel = frameSelections[segIdxStr];
        const frameIndices = sel.frame_indices || [];
        const sourceSegIndices = sel.source_segments || [];

        const rowDiv = document.createElement('div');
        rowDiv.className = 'density-result-row';

        // Label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'density-result-label';
        const c = colorMap[seg.label] || '#888';
        labelDiv.innerHTML = `<span class="smap-idx">[${segIdx}]</span> <span style="color:${c}; font-weight:600;">${seg.label}</span>`;
        const metaSpan = document.createElement('span');
        metaSpan.className = 'density-result-meta';
        metaSpan.textContent = `${frameIndices.length} frames selected | ${sourceSegIndices.length} source segs | step=${sel.step_idx}`;
        labelDiv.appendChild(metaSpan);
        rowDiv.appendChild(labelDiv);

        // Timeline bar showing full episode with selected frames marked
        const timelineDiv = document.createElement('div');
        timelineDiv.className = 'density-timeline-container';

        // Background: segment bars
        const segBar = document.createElement('div');
        segBar.className = 'density-timeline-bg';
        segments.forEach(s => {
          const bar = document.createElement('div');
          bar.className = `density-seg-bg ${s.phase}`;
          const width = (s.num_frames / totalFrames * 100);
          bar.style.width = `${Math.max(width, 0.3)}%`;
          const sc = colorMap[s.label] || '#666';
          bar.style.background = s.phase === 'demo'
            ? hexToRgba(sc, 0.15)
            : hexToRgba(sc, 0.25);
          bar.title = `[${s.idx}] ${s.phase}: ${s.label}`;
          segBar.appendChild(bar);
        });
        timelineDiv.appendChild(segBar);

        // Overlay: selected frame markers
        const markerLayer = document.createElement('div');
        markerLayer.className = 'density-marker-layer';

        // Color-code markers by which source segment they fall in
        frameIndices.forEach(fIdx => {
          const marker = document.createElement('div');
          marker.className = 'density-frame-marker';
          marker.style.left = `${(fIdx / totalFrames) * 100}%`;

          // Find which segment this frame belongs to
          const parentSeg = segments.find(s => fIdx >= s.start_frame && fIdx <= s.end_frame);
          if (parentSeg) {
            const mc = colorMap[parentSeg.label] || '#888';
            marker.style.background = mc;
            // Check if this is a keyframe
            const segKfs = keyframesBySeg[String(parentSeg.idx)];
            if (segKfs) {
              const relFrame = fIdx - parentSeg.start_frame;
              if (segKfs.includes(relFrame)) {
                marker.classList.add('is-keyframe');
              }
            }
          }
          marker.title = `frame ${fIdx}`;
          markerLayer.appendChild(marker);
        });
        timelineDiv.appendChild(markerLayer);

        // Current segment indicator
        const curIndicator = document.createElement('div');
        curIndicator.className = 'density-current-seg';
        curIndicator.style.left = `${(seg.start_frame / totalFrames) * 100}%`;
        curIndicator.style.width = `${(seg.num_frames / totalFrames) * 100}%`;
        timelineDiv.appendChild(curIndicator);

        rowDiv.appendChild(timelineDiv);

        // Legend line: source segments
        if (sourceSegIndices.length > 0) {
          const srcDiv = document.createElement('div');
          srcDiv.className = 'density-source-list';
          srcDiv.innerHTML = '<span class="density-src-label">Sources:</span>';
          sourceSegIndices.forEach(srcIdx => {
            const srcSeg = segments.find(s => s.idx === srcIdx);
            if (!srcSeg) return;
            const chip = document.createElement('span');
            const sc = colorMap[srcSeg.label] || '#888';
            chip.className = `smap-chip source ${srcSeg.phase}`;
            chip.style.borderColor = sc;
            chip.style.background = hexToRgba(sc, srcSeg.phase === 'demo' ? 0.08 : 0.15);
            chip.innerHTML = `<span class="smap-idx">[${srcIdx}]</span><span class="smap-phase">${srcSeg.phase}</span>${truncate(srcSeg.label, 30)}`;
            srcDiv.appendChild(chip);
          });
          rowDiv.appendChild(srcDiv);
        }

        display.appendChild(rowDiv);
      });
    }

    selector.addEventListener('change', () => renderEpisode(parseInt(selector.value)));
    renderEpisode(0);
  });

  return card;
}

// ---------- Sampling density per category ----------
function renderSamplingDensity(taskName) {
  const card = makeCard(
    'Sampling Density',
    'Define how densely to sample from each subtask category. Categories group variants like colors/ordinals into generic patterns.'
  );
  card.classList.add('density-card');

  const container = document.createElement('div');
  container.id = 'density-categories';
  container.innerHTML = '<div class="no-data" style="padding:16px;">Loading categories...</div>';
  card.appendChild(container);

  // Load categories and saved densities in parallel
  Promise.all([
    fetchJSON(`/api/categories/${taskName}`),
    fetchJSON(`/api/sampling_density/${taskName}`),
  ]).then(([categories, savedData]) => {
    container.innerHTML = '';
    if (categories.length === 0) {
      container.innerHTML = '<div class="no-data" style="padding:16px;">No subtask categories found.</div>';
      return;
    }

    const densities = savedData.densities || {};

    // Group by phase
    const execCats = categories.filter(c => c.phase === 'exec');
    const demoCats = categories.filter(c => c.phase === 'demo');

    function renderPhaseGroup(label, phaseCats, phaseKey) {
      if (phaseCats.length === 0) return;
      const groupHeader = document.createElement('div');
      groupHeader.className = 'density-phase-header';
      groupHeader.innerHTML = `<span class="density-phase-tag ${phaseKey}">${label}</span> <span class="density-phase-count">${phaseCats.length} categories</span>`;
      container.appendChild(groupHeader);

      phaseCats.forEach(cat => {
        const catKey = `${cat.phase}::${cat.generic_label}`;
        const row = document.createElement('div');
        row.className = 'density-row';

        // Category label + members
        const labelDiv = document.createElement('div');
        labelDiv.className = 'density-label';
        const genericSpan = document.createElement('span');
        genericSpan.className = 'density-generic';
        genericSpan.textContent = cat.generic_label;
        labelDiv.appendChild(genericSpan);

        if (cat.members.length > 1 || cat.members[0] !== cat.generic_label) {
          const membersDiv = document.createElement('div');
          membersDiv.className = 'density-members';
          membersDiv.textContent = cat.members.join(', ');
          labelDiv.appendChild(membersDiv);
        }

        const countSpan = document.createElement('span');
        countSpan.className = 'density-ep-count';
        countSpan.textContent = `${cat.total_episode_count} eps`;
        labelDiv.appendChild(countSpan);

        row.appendChild(labelDiv);

        // Textarea for density description
        const ta = document.createElement('textarea');
        ta.className = 'density-textarea';
        ta.rows = 2;
        ta.placeholder = 'e.g. "dense: every 2nd frame" or "sparse: 1 keyframe per segment" or "uniform across all source segments"';
        ta.value = densities[catKey] || '';
        ta.dataset.catKey = catKey;
        ta.addEventListener('input', () => {
          ta.style.height = 'auto';
          ta.style.height = ta.scrollHeight + 'px';
        });
        // Auto-grow for pre-filled
        if (ta.value) {
          requestAnimationFrame(() => {
            ta.style.height = 'auto';
            ta.style.height = ta.scrollHeight + 'px';
          });
        }
        row.appendChild(ta);
        container.appendChild(row);
      });
    }

    renderPhaseGroup('Execution', execCats, 'exec');
    renderPhaseGroup('Video Demo', demoCats, 'demo');

    // Save button + status
    const actions = document.createElement('div');
    actions.className = 'rules-actions';
    actions.innerHTML = `
      <button class="btn-save" id="btn-save-density">Save</button>
      <span class="rules-status" id="density-status">${savedData.updated ? 'Last saved: ' + new Date(savedData.updated).toLocaleString() : ''}</span>
    `;
    container.appendChild(actions);

    container.querySelector('#btn-save-density').addEventListener('click', async () => {
      const allTextareas = container.querySelectorAll('.density-textarea');
      const newDensities = {};
      allTextareas.forEach(ta => {
        const val = ta.value.trim();
        if (val) newDensities[ta.dataset.catKey] = val;
      });
      const res = await fetch(`/api/sampling_density/${taskName}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ densities: newDensities }),
      });
      if (res.ok) {
        const status = document.getElementById('density-status');
        status.textContent = 'Saved!';
        status.classList.add('saved');
        setTimeout(() => {
          status.textContent = `Last saved: ${new Date().toLocaleString()}`;
          status.classList.remove('saved');
        }, 2000);
      }
    });
  });

  return card;
}

// ---------- Helpers ----------
function makeCard(title, subtitle) {
  const card = document.createElement('div');
  card.className = 'card';
  const h = document.createElement('h3');
  h.textContent = title;
  card.appendChild(h);
  if (subtitle) {
    const sub = document.createElement('div');
    sub.style.cssText = 'font-size:11px; color:#666; margin-bottom:10px; margin-top:-4px;';
    sub.textContent = subtitle;
    card.appendChild(sub);
  }
  return card;
}

function buildColorMap(labels) {
  const map = {};
  labels.forEach((l, i) => { map[l] = SUBTASK_COLORS[i % SUBTASK_COLORS.length]; });
  return map;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max - 1) + '\u2026' : str;
}

function roundedRect(ctx, x, y, w, h, r) {
  r = Math.min(r, h / 2, w / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}
