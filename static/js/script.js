/**
 * script.js  --  Smart Classroom AI Dashboard
 * Electric Cyan Bento Box Layout
 */

"use strict";

const dom = {
  // Chart values
  score      : document.getElementById("val-score"),
  scoreBar   : document.getElementById("score-bar-fill"),
  
  // Video metrics
  fps        : document.getElementById("val-fps"),
  
  // Bento Cards
  ear        : document.getElementById("val-ear"),
  emotion    : document.getElementById("val-emotion"),
  gaze       : document.getElementById("val-gaze"),
  boredom    : document.getElementById("val-boredom"),
  boredomBar : document.getElementById("boredom-bar-fill"),

  toast      : document.getElementById("toast"),

  // Buttons inside Gear menu
  btnGear    : document.getElementById("btn-gear"),
  dropdown   : document.getElementById("dropdown-menu"),
  btnReport  : document.getElementById("btn-report"),
  btnExport  : document.getElementById("btn-export"),
  btnExportR : document.getElementById("btn-export-r"),
  btnReset   : document.getElementById("btn-reset"),

  modal      : document.getElementById("report-modal"),
  modalBody  : document.getElementById("modal-body"),
  modalClose : document.getElementById("modal-close"),
};

// Toggle Gear Menu
dom.btnGear.addEventListener("click", (e) => {
  e.stopPropagation();
  dom.dropdown.classList.toggle("hidden");
});
document.addEventListener("click", (e) => {
  if (!dom.dropdown.contains(e.target) && e.target !== dom.btnGear) {
    dom.dropdown.classList.add("hidden");
  }
});

// ---------------------------------------------------------------------------
// Chart.js  --  Focus Score Graph
// ---------------------------------------------------------------------------
const MAX_POINTS = 60;
const chartCtx  = document.getElementById("focus-chart").getContext("2d");
const CYAN_COLOR = "#06b6d4";
const CYAN_BG = "rgba(6, 182, 212, 0.15)";
const CYAN_GLOW = "rgba(6, 182, 212, 0.4)";

const chartData = {
  labels  : [],
  datasets: [
    {
      label          : "Focus Score %",
      data           : [],
      pointRadius    : [],
      pointBackgroundColor: [],
      borderColor    : CYAN_COLOR,
      backgroundColor: CYAN_BG,
      borderWidth    : 2,
      tension        : 0.40,
      fill           : true,
    },
    {
      label          : "Session Average %",
      data           : [],
      borderColor    : "rgba(255,255,255,0.3)",
      backgroundColor: "transparent",
      borderWidth    : 1.5,
      tension        : 0.40,
      pointRadius    : 0,
      borderDash     : [5, 4],
      fill           : false,
    },
  ],
};

const focusChart = new Chart(chartCtx, {
  type   : "line",
  data   : chartData,
  options: {
    animation          : false,
    responsive         : true,
    maintainAspectRatio: false,
    interaction        : { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0f172a",
        titleColor     : "#f8fafc",
        bodyColor      : "#94a3b8",
        borderColor    : "rgba(255,255,255,0.1)",
        borderWidth    : 1,
        padding        : 10,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: "#64748b", maxTicksLimit: 8, font: { size: 10, family: "'JetBrains Mono', monospace" } },
        grid : { color: "rgba(255,255,255,0.05)" },
        border: { color: "transparent" },
      },
      y: {
        min  : 0,
        max  : 100,
        ticks: { color: "#64748b", stepSize: 25, font: { family: "'JetBrains Mono', monospace" } },
        grid : { color: "rgba(255,255,255,0.05)" },
        border: { color: "transparent" },
      },
    },
  },
});

const scoreHistory = [];

function pushChart(score, state) {
  const timeLabel = new Date().toLocaleTimeString("en", { hour12: false });
  scoreHistory.push(score);
  if (scoreHistory.length > MAX_POINTS) scoreHistory.shift();
  const avg = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;

  chartData.labels.push(timeLabel);
  chartData.datasets[0].data.push(score);
  chartData.datasets[1].data.push(parseFloat(avg.toFixed(1)));
  
  if (state === "PHONE SUSPECTED") {
      chartData.datasets[0].pointRadius.push(6);
      chartData.datasets[0].pointBackgroundColor.push("#FF0000");
  } else {
      chartData.datasets[0].pointRadius.push(0);
      chartData.datasets[0].pointBackgroundColor.push("transparent");
  }

  if (chartData.labels.length > MAX_POINTS) {
    chartData.labels.shift();
    chartData.datasets[0].data.shift();
    chartData.datasets[1].data.shift();
    chartData.datasets[0].pointRadius.shift();
    chartData.datasets[0].pointBackgroundColor.shift();
  }
  focusChart.update("none");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function fmt(v, dp = 2, unit = "") {
  return (v == null) ? "—" : Number(v).toFixed(dp) + unit;
}

let toastTimeout;
function showToast(msg, type = "info", ms = 3000) {
  dom.toast.textContent = msg;
  dom.toast.className = `show ${type}`;
  clearTimeout(toastTimeout);
  if (type !== "info") {
    toastTimeout = setTimeout(() => { dom.toast.className = ""; }, ms);
  }
}

// ---------------------------------------------------------------------------
// Main poll
// ---------------------------------------------------------------------------
async function poll() {
  try {
    const resp = await fetch("/data_stream", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();

    // FPS
    dom.fps.textContent = fmt(d.fps, 1, " FPS");

    // Chart
    const score = d.focus_score ?? 0;
    const state = d.gaze_state || "N/A";
    dom.score.textContent = fmt(score, 0, "%");
    pushChart(score, state);

    // Emotion
    dom.emotion.textContent = d.emotion || "—";

    // Cards
    dom.ear.textContent = d.ear ? fmt(d.ear, 2) : "—";
    
    // Gaze Direction
    if (state === "Centered") {
      dom.gaze.textContent = "Optimal";
      dom.gaze.className = "live-value gaze-centered";
    } else if (state === "PHONE SUSPECTED") {
      dom.gaze.textContent = "PHONE SUSPECTED";
      dom.gaze.className = "live-value gaze-distracted";
    } else if (state === "Looking Away") {
      dom.gaze.textContent = "Looking Away";
      dom.gaze.className = "live-value gaze-distracted";
    } else {
      dom.gaze.textContent = "—";
      dom.gaze.className = "live-value";
    }

    // Boredom
    const boredomPct = Math.max(0, Math.min(100, (d.boredom || 0) * 100));
    dom.boredom.textContent = fmt(boredomPct, 0, "%");
    dom.boredomBar.style.width = boredomPct + "%";

  } catch (err) {
    // console.warn("[poll connection]", err.message);
  }
}

// ---------------------------------------------------------------------------
// Modals and Reporting
// ---------------------------------------------------------------------------
function renderReport(data) {
  if (data.error) {
    dom.modalBody.innerHTML = `<p class="modal-loading">${data.error}</p>`;
    return;
  }
  const f = data.focus || {};
  const e = data.emotion || {};
  const g = data.gaze_distribution || {};
  const b = data.boredom || {};

  const distRows = Object.entries(e.distribution || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}%</td></tr>`).join("");

  const gazeRows = Object.entries(g)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}%</td></tr>`).join("");

  dom.modalBody.innerHTML = `
    <div class="report-grid">
      <div class="report-block">
        <h3>Session Telemetry</h3>
        <p>Duration : <strong>${data.session_duration_sec}s</strong></p>
        <p>Records  : <strong>${data.total_records}</strong></p>
      </div>
      <div class="report-block">
        <h3>Focus Algorithm</h3>
        <p>Mean Focus  : <strong style="color:var(--cyan)">${f.mean}%</strong></p>
        <p>Distracted  : <strong>${f.pct_distracted}%</strong></p>
      </div>
      <div class="report-block">
        <h3>Emotion Distribution</h3>
        <table class="report-table">${distRows}</table>
      </div>
      <div class="report-block">
        <h3>Gaze Distribution</h3>
        <table class="report-table">${gazeRows}</table>
      </div>
    </div>`;
}

dom.btnReport.addEventListener("click", async () => {
  dom.dropdown.classList.add("hidden");
  dom.modal.classList.remove("hidden");
  dom.modalBody.innerHTML = '<p class="modal-loading">Loading telemetry report…</p>';
  try {
    const r = await fetch("/session_report", { cache: "no-store" });
    renderReport(await r.json());
  } catch { dom.modalBody.innerHTML = '<p class="modal-loading">Failed to load.</p>'; }
});

dom.modalClose.addEventListener("click", () => { dom.modal.classList.add("hidden"); });
dom.modal.addEventListener("click", e => { if (e.target === dom.modal) dom.modal.classList.add("hidden"); });

dom.btnExport.addEventListener("click", async () => {
  dom.dropdown.classList.add("hidden");
  showToast("Exporting CSV…", "info");
  try {
    const r = await fetch("/export_session", { method: "POST" });
    const d = await r.json();
    showToast(d.status === "ok" ? `Saved: ${d.file}` : "Export failed.", d.status === "ok" ? "success" : "error");
  } catch { showToast("Export error.", "error"); }
});

dom.btnExportR.addEventListener("click", async () => {
  dom.dropdown.classList.add("hidden");
  showToast("Exporting JSON…", "info");
  try {
    const r = await fetch("/export_report", { method: "POST" });
    const d = await r.json();
    showToast(d.status === "ok" ? `Saved: ${d.file}` : "Export failed.", d.status === "ok" ? "success" : "error");
  } catch { showToast("Export error.", "error"); }
});

dom.btnReset.addEventListener("click", async () => {
  dom.dropdown.classList.add("hidden");
  if (!confirm("Clear the session log? This cannot be undone.")) return;
  try {
    await fetch("/reset_session", { method: "POST" });
    scoreHistory.length = 0;
    chartData.labels  = [];
    chartData.datasets.forEach(ds => ds.data = []);
    focusChart.update();
    showToast("Session telemetry cleared.", "success");
  } catch { showToast("Reset error.", "error"); }
});

// START
poll();
setInterval(poll, 900);
