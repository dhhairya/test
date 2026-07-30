/**
 * CropGuard AI — Upload Module
 * Handles drag-drop, file selection, camera capture, API submission,
 * and rendering the prediction result card.
 */

window.initUploadModule = function() {
  const dropZone     = document.getElementById('dropZone');
  const fileInput    = document.getElementById('fileInput');
  const cameraBtn    = document.getElementById('cameraBtn');
  const analyzeBtn   = document.getElementById('analyzeBtn');
  const clearBtn     = document.getElementById('clearBtn');
  const preview      = document.getElementById('uploadPreview');
  const previewImg   = document.getElementById('previewImg');
  const previewRemove= document.getElementById('previewRemove');
  const resultSection= document.getElementById('resultSection');

  let selectedFile = null;

  // ── File selection ──────────────────────────────────────────────────────────
  fileInput?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) setFile(file);
  });

  // ── Drag and drop ───────────────────────────────────────────────────────────
  dropZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });

  // ── Camera capture ──────────────────────────────────────────────────────────
  cameraBtn?.addEventListener('click', () => {
    const camInput = document.createElement('input');
    camInput.type = 'file';
    camInput.accept = 'image/*';
    camInput.capture = 'environment';
    camInput.onchange = (e) => {
      const file = e.target.files?.[0];
      if (file) setFile(file);
    };
    camInput.click();
  });

  // ── Remove preview ──────────────────────────────────────────────────────────
  previewRemove?.addEventListener('click', clearFile);
  clearBtn?.addEventListener('click', clearFile);

  // ── Analyze ─────────────────────────────────────────────────────────────────
  analyzeBtn?.addEventListener('click', () => {
    if (selectedFile) submitAnalysis(selectedFile);
  });

  // ── Core functions ──────────────────────────────────────────────────────────

  function setFile(file) {
    const allowed = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowed.includes(file.type)) {
      showToast('Unsupported file type. Use JPEG, PNG, or WebP.', 'error');
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showToast('File too large. Maximum size is 16 MB.', 'error');
      return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      preview?.classList.add('visible');
      dropZone.style.display = 'none';
      if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '🔍 <span>Analyze Leaf</span>';
      }
    };
    reader.readAsDataURL(file);

    // Clear previous result
    if (resultSection) resultSection.innerHTML = '';
  }

  function clearFile() {
    selectedFile = null;
    if (previewImg) previewImg.src = '';
    preview?.classList.remove('visible');
    dropZone.style.display = '';
    if (fileInput) fileInput.value = '';
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '🔍 <span>Analyze Leaf</span>';
    }
    if (resultSection) resultSection.innerHTML = '';
  }

  async function submitAnalysis(file) {
    if (!window.CropGuard.isOnline) {
      // Queue for offline processing
      const queued = await window.OfflineQueue?.enqueue?.(file);
      if (queued) {
        showToast('Offline: detection queued. Will submit when online.', 'info');
        renderOfflineQueued();
        return;
      }
    }

    showSpinner('Analyzing crop leaf…');
    if (analyzeBtn) analyzeBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('image', file);

      // Attach location if available
      const loc = window.CropGuard.userLocation;
      if (loc) {
        formData.append('lat', loc.lat);
        formData.append('lng', loc.lng);
      }

      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();

      if (!resp.ok || data.error) {
        renderError(data.error || 'Analysis failed. Please try again.');
        return;
      }

      window.CropGuard.lastDetection = data;
      renderResult(data, previewImg?.src);

    } catch (err) {
      if (!navigator.onLine) {
        const queued = await window.OfflineQueue?.enqueue?.(file);
        showToast('Connection lost — detection queued for later sync.', 'info');
        if (queued) renderOfflineQueued();
      } else {
        renderError('Network error. Check your connection and try again.');
      }
    } finally {
      hideSpinner();
      if (analyzeBtn) analyzeBtn.disabled = false;
    }
  }

  // ── Render functions ─────────────────────────────────────────────────────────

  function renderResult(data, imageDataUrl) {
    if (!resultSection) return;

    if (data.status === 'low_confidence') {
      renderLowConfidence(data, imageDataUrl);
      return;
    }

    const isHealthy  = data.is_healthy;
    const badgeClass = isHealthy ? 'healthy' : 'diseased';
    const badgeText  = isHealthy ? '✅ Healthy Crop' : '⚠️ Disease Detected';
    const confPct    = Math.round((data.confidence || 0) * 100);
    const severity   = data.severity || 'early';

    const demoNotice = data.demo_mode
      ? `<div class="demo-notice">🔬 Demo mode — using mock classifier. Train the model for real predictions.</div>`
      : '';

    resultSection.innerHTML = `
      ${demoNotice}
      <div class="result-card" id="resultCard">
        <div class="result-header">
          ${imageDataUrl ? `<img src="${imageDataUrl}" class="leaf-thumb" alt="Leaf photo">` : ''}
          <div class="result-meta">
            <div class="result-status-badge ${badgeClass}">${badgeText}</div>
            <div class="result-crop">${escHtml(data.crop || '—')}</div>
            <div class="result-disease">${escHtml(data.disease || '—')}</div>
          </div>
        </div>

        <div class="result-body">
          <div class="confidence-meter">
            <div class="confidence-label">
              <span>Model Confidence</span>
              <strong>${confPct}%</strong>
            </div>
            <div class="confidence-track">
              <div class="confidence-fill" id="confFill" style="width:0%"></div>
            </div>
          </div>

          <div class="result-details">
            <div class="detail-pill">
              <div class="label">Crop</div>
              <div class="value">${escHtml(data.crop || '—')}</div>
            </div>
            <div class="detail-pill">
              <div class="label">Severity</div>
              <div class="value ${isHealthy ? '' : severity}">${isHealthy ? 'None' : capitalize(severity)}</div>
            </div>
            <div class="detail-pill">
              <div class="label">Confidence</div>
              <div class="value">${confPct}%</div>
            </div>
            <div class="detail-pill">
              <div class="label">Detected At</div>
              <div class="value">${formatDate(data.timestamp)}</div>
            </div>
          </div>

          ${data.description ? `
            <div class="result-description">
              <strong>About this disease:</strong> ${escHtml(data.description)}
            </div>
          ` : ''}

          <div class="result-actions">
            ${!isHealthy && data.detection_id ? `
              <button class="btn btn-primary" onclick="loadProgressionPanel(${data.detection_id})">
                📈 View Progression
              </button>
              <button class="btn btn-outline" onclick="loadRecommendations(${data.detection_id})">
                💊 Get Recommendations
              </button>
            ` : ''}
            <button class="btn btn-secondary" onclick="switchTab('dashboard')">
              📊 View Dashboard
            </button>
          </div>
        </div>

        <div id="progressionPanel"></div>
        <div id="recommendationsPanel"></div>
      </div>
    `;

    // Animate confidence bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = document.getElementById('confFill');
        if (fill) fill.style.width = `${confPct}%`;
      }, 100);
    });

    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderLowConfidence(data, imageDataUrl) {
    if (!resultSection) return;
    const confPct = Math.round((data.confidence || 0) * 100);

    resultSection.innerHTML = `
      <div class="result-card" id="resultCard">
        <div class="result-header">
          ${imageDataUrl ? `<img src="${imageDataUrl}" class="leaf-thumb" alt="Leaf photo">` : ''}
          <div class="result-meta">
            <div class="result-status-badge low-confidence">⚠️ Low Confidence</div>
            <div class="result-crop">Unable to Identify</div>
            <div class="result-disease">Confidence ${confPct}% (threshold: ${Math.round(data.threshold * 100)}%)</div>
          </div>
        </div>
        <div class="result-body">
          <div class="confidence-meter">
            <div class="confidence-label">
              <span>Model Confidence</span>
              <strong>${confPct}%</strong>
            </div>
            <div class="confidence-track">
              <div class="confidence-fill low" style="width:${confPct}%"></div>
            </div>
          </div>
          <p style="color:var(--clr-text-secondary);font-size:14px;margin-bottom:var(--space-4)">
            ${escHtml(data.message)}
          </p>
        </div>
        <div class="low-conf-warning">
          <h4>📸 Tips for a better photo</h4>
          <ul class="tips-list">
            ${(data.tips || []).map((t) => `<li>${escHtml(t)}</li>`).join('')}
          </ul>
          <button class="btn btn-primary" style="margin-top:var(--space-4)" onclick="clearUploadResult()">
            📷 Try Again
          </button>
        </div>
      </div>
    `;
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderError(message) {
    if (!resultSection) return;
    resultSection.innerHTML = `
      <div class="alert-banner">
        <span class="alert-banner-icon">❌</span>
        <div class="alert-banner-body">
          <h4>Analysis Failed</h4>
          <p>${escHtml(message)}</p>
        </div>
      </div>
    `;
  }

  function renderOfflineQueued() {
    if (!resultSection) return;
    resultSection.innerHTML = `
      <div class="alert-banner" style="border-color:hsla(205,85%,58%,0.2);background:hsla(205,85%,58%,0.07)">
        <span class="alert-banner-icon">📶</span>
        <div class="alert-banner-body">
          <h4 style="color:var(--clr-info)">Detection Queued</h4>
          <p>This photo has been saved locally and will be submitted automatically when you reconnect to the internet.</p>
        </div>
      </div>
    `;
  }

  window.clearUploadResult = clearFile;
};

// ── Progression panel loader ───────────────────────────────────────────────────
window.loadProgressionPanel = async function(detectionId) {
  const panel = document.getElementById('progressionPanel');
  if (!panel) return;
  panel.innerHTML = `<div style="padding:var(--space-5);color:var(--clr-text-muted);text-align:center">⏳ Loading progression forecast…</div>`;

  try {
    const data = await apiGet(`/api/progression/${detectionId}`);
    const prog = data.progression;
    if (!prog?.windows) return;

    const windows = prog.windows.map((w) => `
      <div class="detail-pill" style="text-align:center">
        <div class="label">${w.days}-Day Forecast</div>
        <div class="value ${w.stage}" style="font-size:18px;margin:4px 0">${capitalize(w.stage)}</div>
        <div style="font-size:12px;color:var(--clr-text-muted)">${Math.round(w.probability * 100)}% probability</div>
        <div style="font-size:11px;color:var(--clr-text-muted)">Spread: ${w.spread_risk}</div>
      </div>
    `).join('');

    panel.innerHTML = `
      <div style="padding:var(--space-5);border-top:1px solid var(--clr-border-subtle)">
        <h4 style="font-size:14px;font-weight:600;margin-bottom:var(--space-4);color:var(--clr-text-secondary)">
          📈 Disease Progression Forecast
          ${prog.stub ? '<span style="font-size:11px;color:var(--clr-text-muted)">(stub data)</span>' : ''}
        </h4>
        <div class="result-details" style="grid-template-columns:repeat(3,1fr)">${windows}</div>
        ${prog.weather_factor ? `<p style="font-size:12px;color:var(--clr-text-muted);margin-top:var(--space-3)">
          Weather factor: <strong>${prog.weather_factor}</strong>
        </p>` : ''}
      </div>
    `;
  } catch (e) {
    panel.innerHTML = `<div style="padding:var(--space-4);color:var(--clr-text-muted)">Could not load progression data.</div>`;
  }
};

// ── Recommendations panel loader ───────────────────────────────────────────────
window.loadRecommendations = async function(detectionId) {
  const panel = document.getElementById('recommendationsPanel');
  if (!panel) return;
  panel.innerHTML = `<div style="padding:var(--space-5);color:var(--clr-text-muted);text-align:center">⏳ Loading recommendations…</div>`;

  try {
    const data = await apiGet(`/api/recommendations/${detectionId}`);
    const recs = data.recommendations;
    if (!recs?.recommendations) return;

    const priorityColors = { critical: 'var(--clr-danger)', high: 'var(--clr-accent)', medium: 'var(--clr-primary)', low: 'var(--clr-text-muted)' };

    const items = recs.recommendations.map((r) => `
      <div style="display:flex;gap:var(--space-3);padding:var(--space-3);border-bottom:1px solid var(--clr-border-subtle);align-items:flex-start">
        <div style="width:8px;height:8px;border-radius:50%;background:${priorityColors[r.priority]||'gray'};flex-shrink:0;margin-top:5px"></div>
        <div>
          <div style="font-size:13px;font-weight:500">${escHtml(r.action)}</div>
          <div style="font-size:11px;color:var(--clr-text-muted);margin-top:2px">⏱ ${escHtml(r.timing)} · ${r.type}</div>
        </div>
      </div>
    `).join('');

    const pesticide = recs.pesticide_forecast;

    panel.innerHTML = `
      <div style="padding:var(--space-5);border-top:1px solid var(--clr-border-subtle)">
        <h4 style="font-size:14px;font-weight:600;margin-bottom:var(--space-4);color:var(--clr-text-secondary)">
          💊 Recommended Actions
        </h4>
        ${recs.weather_note ? `<div class="demo-notice" style="margin-bottom:var(--space-3)">${escHtml(recs.weather_note)}</div>` : ''}
        <div style="border:1px solid var(--clr-border-subtle);border-radius:var(--radius-md);overflow:hidden;margin-bottom:var(--space-4)">${items}</div>
        ${pesticide ? `
          <div class="detail-pill">
            <div class="label">Pesticide Forecast</div>
            <div class="value" style="font-size:13px;margin-top:4px">${escHtml(pesticide.type)}: ${escHtml(pesticide.active_ingredient)}</div>
            <div style="font-size:12px;color:var(--clr-text-muted);margin-top:4px">⏱ ${escHtml(pesticide.timing_window)}</div>
            <div style="font-size:12px;color:var(--clr-text-muted)">📦 ${escHtml(pesticide.quantity_per_acre)} per acre</div>
          </div>
        ` : ''}
      </div>
    `;
  } catch (e) {
    panel.innerHTML = `<div style="padding:var(--space-4);color:var(--clr-text-muted)">Could not load recommendations.</div>`;
  }
};

// ── Utils ──────────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}
