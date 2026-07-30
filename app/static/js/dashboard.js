/**
 * CropGuard AI — Dashboard Module
 * Fetches detection data and renders Chart.js charts + history table.
 */

let chartInstances = {};

window.loadDashboard = async function() {
  if (window.CropGuard?.currentTab !== 'dashboard') return;

  const days = parseInt(document.getElementById('dashDays')?.value || '30', 10);

  try {
    const [timeline, detections] = await Promise.all([
      apiGet(`/api/timeline?days=${days}`),
      apiGet(`/api/detections?days=${days}&limit=20`),
    ]);

    renderSummaryStats(timeline.summary);
    renderTimelineChart(timeline.timeline);
    renderDiseaseChart(timeline.disease_breakdown);
    renderSeverityChart(timeline.severity_breakdown);
    renderHistoryTable(detections.detections);

    // Load alerts if location available
    if (window.CropGuard.userLocation) {
      loadAlertsBanner(window.CropGuard.userLocation);
    }

  } catch (err) {
    console.error('[Dashboard] Error:', err);
    showToast('Failed to load dashboard data.', 'error');
  }
};

// ── Summary stats ──────────────────────────────────────────────────────────────
function renderSummaryStats(summary) {
  if (!summary) return;
  setEl('statTotal',     summary.total       ?? 0);
  setEl('statHealthy',   summary.healthy      ?? 0);
  setEl('statDiseased',  summary.diseased     ?? 0);
  setEl('statHealthRate',`${summary.health_rate ?? 0}%`);
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Chart helpers ──────────────────────────────────────────────────────────────
function destroyChart(key) {
  if (chartInstances[key]) {
    chartInstances[key].destroy();
    delete chartInstances[key];
  }
}

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: 'hsl(220, 15%, 65%)',
        font: { family: "'Inter', sans-serif", size: 12 },
      },
    },
    tooltip: {
      backgroundColor: 'hsl(220, 20%, 10%)',
      borderColor: 'hsl(220, 14%, 22%)',
      borderWidth: 1,
      titleColor: 'hsl(220, 20%, 95%)',
      bodyColor: 'hsl(220, 15%, 65%)',
      padding: 12,
    },
  },
  scales: {
    x: {
      grid:  { color: 'hsla(220, 14%, 22%, 0.5)' },
      ticks: { color: 'hsl(220, 15%, 55%)', font: { size: 11 } },
    },
    y: {
      grid:  { color: 'hsla(220, 14%, 22%, 0.5)' },
      ticks: { color: 'hsl(220, 15%, 55%)', font: { size: 11 } },
      beginAtZero: true,
    },
  },
};

// ── Timeline chart ─────────────────────────────────────────────────────────────
function renderTimelineChart(data) {
  const canvas = document.getElementById('timelineChart');
  if (!canvas) return;
  destroyChart('timeline');

  if (!data || data.length === 0) {
    showEmptyChart(canvas, 'No detections in this period');
    return;
  }

  const labels   = data.map((d) => formatDateShort(d.date));
  const diseased = data.map((d) => d.diseased || 0);
  const healthy  = data.map((d) => d.healthy  || 0);

  chartInstances['timeline'] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label:           'Diseased',
          data:            diseased,
          backgroundColor: 'hsla(4, 85%, 60%, 0.7)',
          borderColor:     'hsl(4, 85%, 60%)',
          borderWidth:     1,
          borderRadius:    4,
        },
        {
          label:           'Healthy',
          data:            healthy,
          backgroundColor: 'hsla(142, 65%, 45%, 0.7)',
          borderColor:     'hsl(142, 65%, 45%)',
          borderWidth:     1,
          borderRadius:    4,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { ...CHART_DEFAULTS.plugins.legend, position: 'top' },
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        x: { ...CHART_DEFAULTS.scales.x, stacked: true },
        y: { ...CHART_DEFAULTS.scales.y, stacked: true },
      },
    },
  });
}

// ── Disease breakdown doughnut ─────────────────────────────────────────────────
function renderDiseaseChart(data) {
  const canvas = document.getElementById('diseaseChart');
  if (!canvas) return;
  destroyChart('disease');

  if (!data || data.length === 0) {
    showEmptyChart(canvas, 'No diseases detected');
    return;
  }

  const palette = [
    'hsl(4,   85%, 60%)', 'hsl(38,  95%, 58%)', 'hsl(205, 85%, 58%)',
    'hsl(270, 65%, 65%)', 'hsl(142, 65%, 50%)', 'hsl(322, 70%, 60%)',
    'hsl(180, 65%, 50%)', 'hsl(60,  85%, 55%)',  'hsl(25, 90%, 60%)',
    'hsl(240, 65%, 60%)',
  ];

  chartInstances['disease'] = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels:   data.map((d) => d.disease),
      datasets: [{
        data:            data.map((d) => d.count),
        backgroundColor: data.map((_, i) => palette[i % palette.length]),
        borderColor:     'hsl(220, 18%, 13%)',
        borderWidth:     2,
        hoverOffset:     8,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: undefined,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { ...CHART_DEFAULTS.plugins.legend, position: 'right' },
      },
      cutout: '65%',
    },
  });
}

// ── Severity radar/bar ─────────────────────────────────────────────────────────
function renderSeverityChart(data) {
  const canvas = document.getElementById('severityChart');
  if (!canvas) return;
  destroyChart('severity');

  if (!data || data.length === 0) {
    showEmptyChart(canvas, 'No severity data');
    return;
  }

  const colorMap = {
    early:    'hsla(142, 65%, 45%, 0.8)',
    moderate: 'hsla(38,  95%, 58%, 0.8)',
    severe:   'hsla(4,   85%, 60%, 0.8)',
    none:     'hsla(205, 85%, 58%, 0.8)',
  };

  chartInstances['severity'] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: data.map((d) => capitalize(d.severity)),
      datasets: [{
        label:           'Detections',
        data:            data.map((d) => d.count),
        backgroundColor: data.map((d) => colorMap[d.severity] || 'hsla(220, 14%, 40%, 0.8)'),
        borderRadius:    6,
        borderWidth:     0,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: 'y',
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: false },
      },
    },
  });
}

// ── History table ──────────────────────────────────────────────────────────────
function renderHistoryTable(detections) {
  const tbody = document.getElementById('historyTableBody');
  const empty = document.getElementById('historyEmpty');
  if (!tbody) return;

  if (!detections || detections.length === 0) {
    tbody.innerHTML = '';
    if (empty) empty.style.display = 'block';
    return;
  }
  if (empty) empty.style.display = 'none';

  tbody.innerHTML = detections.map((d) => {
    const sev  = d.is_healthy ? 'healthy' : (d.severity || 'unknown');
    const conf = d.confidence ? `${Math.round(d.confidence * 100)}%` : '—';
    return `
      <tr>
        <td>${formatDate(d.timestamp)}</td>
        <td><strong>${escHtml(d.crop || '—')}</strong></td>
        <td>${escHtml(d.disease || 'N/A')}</td>
        <td><span class="severity-badge ${sev}">${capitalize(sev)}</span></td>
        <td>${conf}</td>
        <td style="color:var(--clr-text-muted);font-size:12px">${escHtml(d.location_name || '—')}</td>
      </tr>
    `;
  }).join('');
}

// ── Alerts banner ──────────────────────────────────────────────────────────────
async function loadAlertsBanner(loc) {
  try {
    const data = await apiGet(`/api/alerts?lat=${loc.lat}&lng=${loc.lng}`);
    const container = document.getElementById('alertsContainer');
    if (!container || !data.alerts?.length) return;

    container.innerHTML = data.alerts.map((a) => `
      <div class="alert-banner">
        <span class="alert-banner-icon">🚨</span>
        <div class="alert-banner-body">
          <h4>${escHtml(a.disease)} outbreak near your area</h4>
          <p>${escHtml(a.crop || '')} · ${a.detection_count} cases within ${a.radius_km}km · ${a.distance_km}km from you</p>
        </div>
      </div>
    `).join('');
  } catch {
    // Silently fail — alerts are non-critical
  }
}

// ── Period selector ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('dashDays')?.addEventListener('change', loadDashboard);
  document.getElementById('refreshDash')?.addEventListener('click', loadDashboard);
  document.getElementById('getLocationBtn')?.addEventListener('click', async () => {
    const loc = await requestLocation();
    const el  = document.getElementById('locationStatus');
    if (loc) {
      if (el) el.textContent = `📍 ${loc.lat.toFixed(3)}, ${loc.lng.toFixed(3)}`;
      showToast('Location acquired.', 'success');
      loadAlertsBanner(loc);
    } else {
      showToast('Location unavailable — check permissions.', 'error');
    }
  });
});

// ── Helpers ────────────────────────────────────────────────────────────────────
function showEmptyChart(canvas, message) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'hsl(220, 15%, 45%)';
  ctx.font = '13px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}
function capitalize(str) { return str ? str.charAt(0).toUpperCase() + str.slice(1) : ''; }
function escHtml(str) {
  return String(str ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
