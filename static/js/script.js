/**
 * Smart Classroom AI — Teacher Command Center
 * script.js v3.0.0
 *
 * Responsibilities:
 *  1. Poll /data_stream (SSE) → update aggregate classroom metrics
 *  2. Poll /attendance_live  → update live attendance roster panel
 *  3. Drive Chart.js with rolling class-average focus baseline
 *  4. Session timer
 *  5. Report modal with attendance table + focus analytics
 *  6. Gear menu: export CSV / JSON / report / reset
 */

'use strict';

/* ══════════════════════════════════════════════════════════════
   DOM refs
   ══════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const elFps        = $('val-fps');
const elScore      = $('val-score');
const elEmotion    = $('val-emotion');
const elGaze       = $('val-gaze');
const elBoredom    = $('val-boredom');
const elBoredomBar = $('boredom-bar-fill');
const elPresent    = $('val-present');
const elBadgeCount = $('badge-count');
const elBadgeFaces = $('badge-faces');
const elFaceCount  = $('overlay-face-count');
const elClock      = $('session-clock');
const elRosterList = $('roster-list');
const elRosterEmpty= $('roster-empty');
const elRosterCount= $('roster-count');
const elPhoneAlerts= $('badge-phone');   // phone-alert badge (may be null until DOM ready)

/* ══════════════════════════════════════════════════════════════
   Chart.js — Class Focus Timeline
   ══════════════════════════════════════════════════════════════ */
const MAX_POINTS = 60;

const focusChart = new Chart($('focus-chart').getContext('2d'), {
  type: 'line',
  data: {
    labels  : [],
    datasets: [{
      label          : 'Class Avg Focus',
      data           : [],
      borderColor    : '#06b6d4',
      backgroundColor: 'rgba(6,182,212,0.08)',
      borderWidth    : 2,
      pointRadius    : 0,
      tension        : 0.4,
      fill           : true,
    }],
  },
  options: {
    responsive        : true,
    maintainAspectRatio: false,
    animation         : false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` Focus: ${ctx.parsed.y.toFixed(1)}%`
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#4a5280', font: { family: 'JetBrains Mono', size: 10 },
                 maxTicksLimit: 8 },
        grid : { color: 'rgba(255,255,255,0.03)' },
      },
      y: {
        min  : 0,
        max  : 100,
        ticks: { color: '#4a5280', font: { family: 'JetBrains Mono', size: 10 },
                 callback: v => v + '%' },
        grid : { color: 'rgba(255,255,255,0.05)' },
      },
    },
  },
});

function pushChartPoint(focusScore) {
  const now    = new Date();
  const label  = `${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
  const ds     = focusChart.data;

  ds.labels.push(label);
  ds.datasets[0].data.push(focusScore);

  if (ds.labels.length > MAX_POINTS) {
    ds.labels.shift();
    ds.datasets[0].data.shift();
  }

  // Colour the line based on the mean focus of the visible window
  const vis  = ds.datasets[0].data;
  const mean = vis.reduce((a, b) => a + b, 0) / vis.length;
  ds.datasets[0].borderColor     = mean >= 60 ? '#06b6d4' : '#f59e0b';
  ds.datasets[0].backgroundColor = mean >= 60 ? 'rgba(6,182,212,0.08)' : 'rgba(245,158,11,0.08)';

  focusChart.update('none');
}

/* ══════════════════════════════════════════════════════════════
   Session Clock
   ══════════════════════════════════════════════════════════════ */
const _sessionStart = Date.now();

function tickClock() {
  const elapsed = Math.floor((Date.now() - _sessionStart) / 1000);
  const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  elClock.textContent = `${h}:${m}:${s}`;
}
setInterval(tickClock, 1000);

/* ══════════════════════════════════════════════════════════════
   SSE — /data_stream (classroom aggregate metrics)
   ══════════════════════════════════════════════════════════════ */
let _lastFocusPush    = 0;
let _phoneAlertShown  = false;   // prevents toast spam

function connectStream() {
  const es = new EventSource('/data_stream');

  es.addEventListener('metrics', e => {
    try {
      const m = JSON.parse(e.data);
      applyMetrics(m);
    } catch (_) { /* ignore malformed events */ }
  });

  es.onerror = () => {
    es.close();
    setTimeout(connectStream, 3000);
  };
}

function applyMetrics(m) {
  /* FPS */
  elFps.textContent = `${m.fps?.toFixed(1) ?? '—'} FPS`;

  /* Class average focus */
  const focus = m.focus_score ?? 0;
  elScore.textContent = `${focus.toFixed(1)}%`;

  /* Push to chart every ~2 s */
  const now = Date.now();
  if (now - _lastFocusPush > 2000) {
    pushChartPoint(focus);
    _lastFocusPush = now;
  }

  /* Face count badges */
  const fc = m.face_count ?? 0;
  elFaceCount.textContent = fc;
  elBadgeFaces.textContent = fc;

  /* Dominant emotion */
  elEmotion.textContent = m.emotion ?? '—';

  /* Gaze status */
  const gazeState = m.gaze_state ?? m.gaze ?? '—';
  elGaze.textContent = gazeState;
  elGaze.className   = 'live-value' +
    (gazeState === 'Centered' ? ' gaze-centered' : ' gaze-distracted');

  /* Room boredom */
  const bPct = ((m.boredom ?? 0) * 100);
  elBoredom.textContent = `${bPct.toFixed(0)}%`;
  elBoredomBar.style.width      = `${Math.min(bPct, 100)}%`;
  elBoredomBar.style.background = bPct > 60
    ? '#ef4444'
    : bPct > 35 ? '#f59e0b' : '#06b6d4';

  /* Phone alerts */
  const alerts = m.phone_alerts ?? 0;
  if (elPhoneAlerts) {
    elPhoneAlerts.textContent = alerts;
    const badge = elPhoneAlerts.closest('.header-badge');
    if (badge) {
      badge.style.display = alerts > 0 ? 'flex' : 'none';
    }
  }
  // One-shot toast when alert fires
  if (alerts > 0 && !_phoneAlertShown) {
    _phoneAlertShown = true;
    showToast(`⚠️ ${alerts} student${alerts > 1 ? 's' : ''} using phone!`, 'error');
  } else if (alerts === 0) {
    _phoneAlertShown = false;
  }
}

connectStream();

/* ══════════════════════════════════════════════════════════════
   Attendance Roster Poll — /attendance_live every 3 s
   ══════════════════════════════════════════════════════════════ */
const _knownPrns = new Set();   // track who we've already added to DOM

async function pollAttendance() {
  try {
    const res  = await fetch('/attendance_live');
    const data = await res.json();

    const students = data.students ?? [];
    const total    = data.total_present ?? 0;

    /* Update header badge & bottom card */
    elBadgeCount.textContent = total;
    elPresent.textContent    = total;
    elRosterCount.textContent = `${total} student${total !== 1 ? 's' : ''}`;

    /* Show / hide empty placeholder */
    elRosterEmpty.style.display = total === 0 ? 'flex' : 'none';

    /* Append new entries (do NOT rebuild from scratch → no flicker) */
    students.forEach(s => {
      if (_knownPrns.has(s.prn)) return;   // already rendered
      _knownPrns.add(s.prn);

      const li = document.createElement('li');
      li.className = 'roster-entry';
      li.id        = `roster-${s.prn}`;

      const [datePart, timePart] = (s.marked_at ?? '').split(' ');
      const displayTime = timePart ?? s.marked_at ?? '';

      li.innerHTML = `
        <div class="roster-dot"></div>
        <span class="roster-name">${escHtml(s.name)}</span>
        <span class="roster-prn">${escHtml(s.prn)}</span>
        <span class="roster-time">${escHtml(displayTime)}</span>
      `;
      elRosterList.prepend(li);   // newest at top
    });

  } catch (err) {
    console.warn('[attendance] Poll failed:', err);
  }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

pollAttendance();
setInterval(pollAttendance, 3000);

/* ══════════════════════════════════════════════════════════════
   Gear menu toggle
   ══════════════════════════════════════════════════════════════ */
const btnGear    = $('btn-gear');
const dropMenu   = $('dropdown-menu');

btnGear.addEventListener('click', e => {
  e.stopPropagation();
  dropMenu.classList.toggle('hidden');
});
document.addEventListener('click', () => dropMenu.classList.add('hidden'));

/* ══════════════════════════════════════════════════════════════
   Toast helper
   ══════════════════════════════════════════════════════════════ */
const toastEl = $('toast');
let _toastTimer;

function showToast(msg, type = 'info') {
  toastEl.textContent = msg;
  toastEl.className   = `show ${type}`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toastEl.classList.remove('show'), 3500);
}

/* ══════════════════════════════════════════════════════════════
   Export / Reset buttons
   ══════════════════════════════════════════════════════════════ */
$('btn-export').addEventListener('click', async () => {
  dropMenu.classList.add('hidden');
  try {
    const r = await fetch('/export_session', { method: 'POST' });
    const d = await r.json();
    showToast(`CSV saved: ${d.path}`, 'success');
  } catch { showToast('Export failed.', 'error'); }
});

$('btn-export-r').addEventListener('click', async () => {
  dropMenu.classList.add('hidden');
  try {
    const r = await fetch('/export_report', { method: 'POST' });
    const d = await r.json();
    showToast(`JSON saved: ${d.path}`, 'success');
  } catch { showToast('Export failed.', 'error'); }
});

$('btn-reset').addEventListener('click', async () => {
  dropMenu.classList.add('hidden');
  if (!confirm('Reset session? All in-memory logs will be cleared.')) return;
  try {
    await fetch('/reset_session', { method: 'POST' });
    /* Clear chart */
    focusChart.data.labels = [];
    focusChart.data.datasets[0].data = [];
    focusChart.update('none');
    /* Clear roster */
    _knownPrns.clear();
    elRosterList.innerHTML = '';
    elRosterEmpty.style.display = 'flex';
    elBadgeCount.textContent = 0;
    elPresent.textContent = 0;
    elRosterCount.textContent = '0 students';
    showToast('Session reset.', 'info');
  } catch { showToast('Reset failed.', 'error'); }
});

/* ══════════════════════════════════════════════════════════════
   Session Report Modal
   ══════════════════════════════════════════════════════════════ */
const modal      = $('report-modal');
const modalBody  = $('modal-body');
const btnReport  = $('btn-report');
const btnClose   = $('modal-close');

btnReport.addEventListener('click', () => {
  dropMenu.classList.add('hidden');
  openModal();
});
btnClose.addEventListener('click', closeModal);
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

function openModal() {
  modal.hidden = false;
  modal.classList.remove('hidden');
  modalBody.innerHTML = '<p class="modal-loading">Loading report…</p>';
  fetchReport();
}
function closeModal() {
  modal.classList.add('hidden');
  modal.hidden = true;
}

async function fetchReport() {
  try {
    const res  = await fetch('/session_report');
    const data = await res.json();
    renderReport(data);
  } catch {
    modalBody.innerHTML = '<p style="color:#ef4444">Failed to load report.</p>';
  }
}

function renderReport(d) {
  if (d.error) {
    modalBody.innerHTML = `<p style="color:#f59e0b">${escHtml(d.error)}</p>`;
    return;
  }

  const f = d.focus ?? {};
  const e = d.emotion ?? {};
  const b = d.boredom ?? {};
  const att = d.attendance ?? {};
  const gazeKeys = Object.keys(d.gaze_distribution ?? {});

  /* Attendance rows */
  const attStudents = att.students ?? [];
  const attRows = attStudents.length
    ? attStudents.map(s => `
        <tr>
          <td>${escHtml(s.name)}</td>
          <td style="font-family:monospace;color:#8891c0">${escHtml(s.prn)}</td>
          <td><span class="attendance-badge">✓ Present</span></td>
          <td style="font-family:monospace;font-size:0.72rem;color:#4a5280">${escHtml(s.marked_at)}</td>
        </tr>`).join('')
    : `<tr><td colspan="4" style="text-align:center;color:#4a5280;padding:16px">
         No students identified this session.
       </td></tr>`;

  /* Gaze table rows */
  const gazeRows = gazeKeys.map(k =>
    `<p><strong>${escHtml(k)}</strong> — ${d.gaze_distribution[k]}%</p>`
  ).join('');

  modalBody.innerHTML = `
    <div class="report-grid">
      <!-- Focus Block -->
      <div class="report-block">
        <h3>📊 Class Focus Analysis</h3>
        <p>Duration: <strong>${d.session_duration_sec ?? 0}s</strong></p>
        <p>Records: <strong>${d.total_records ?? 0}</strong></p>
        <p>Avg Focus: <strong>${f.mean ?? '—'}%</strong></p>
        <p>Peak: <strong>${f.max ?? '—'}%</strong> · Min: <strong>${f.min ?? '—'}%</strong></p>
        <p>Focused: <strong>${f.pct_focused ?? 0}%</strong> · Distracted: <strong>${f.pct_distracted ?? 0}%</strong></p>
        <p>Drowsy Alerts: <strong style="color:${(d.drowsiness_alerts ?? 0) > 0 ? '#ef4444' : '#10b981'}">${d.drowsiness_alerts ?? 0}</strong></p>
      </div>

      <!-- Emotion / Boredom Block -->
      <div class="report-block">
        <h3>🎭 Emotion & Boredom</h3>
        <p>Peak Emotion: <strong>${escHtml(e.peak ?? '—')}</strong></p>
        ${Object.entries(e.distribution ?? {}).map(([k,v]) =>
          `<p>${escHtml(k)}: <strong>${v}%</strong></p>`).join('')}
        <br>
        <p>Avg Boredom: <strong>${((b.mean ?? 0)*100).toFixed(1)}%</strong></p>
        <p>Peak Boredom: <strong>${((b.max ?? 0)*100).toFixed(1)}%</strong></p>
      </div>

      <!-- Gaze Block -->
      <div class="report-block">
        <h3>👁 Gaze Distribution</h3>
        ${gazeRows || '<p style="color:#4a5280">No gaze data.</p>'}
      </div>

      <!-- Attendance Summary Block -->
      <div class="report-block">
        <h3>✅ Attendance Summary</h3>
        <p>Total Identified: <strong style="color:#10b981">${att.total_present ?? 0} student(s)</strong></p>
      </div>
    </div>

    <!-- Full Attendance Table -->
    <div class="report-block" style="margin-top:0">
      <h3>📋 Attendance Sheet</h3>
      <table class="attendance-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>PRN</th>
            <th>Status</th>
            <th>Time Marked</th>
          </tr>
        </thead>
        <tbody>${attRows}</tbody>
      </table>
    </div>
  `;
}
