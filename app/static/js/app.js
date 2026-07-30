/**
 * CropGuard AI — Main Application Shell
 * Tab routing, SW registration, online/offline detection, toast system.
 */

// ── State ─────────────────────────────────────────────────────────────────────
window.CropGuard = {
  currentTab:   'upload',
  isOnline:     navigator.onLine,
  lastDetection: null,
  userLocation:  null,
};

// ── DOM Ready ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initServiceWorker();
  initNavigation();
  initOnlineStatus();
  initOfflineModule();
  initUploadModule();
  loadDashboard();

  // Check for tab param in URL
  const urlTab = new URLSearchParams(location.search).get('tab');
  if (urlTab) switchTab(urlTab);
});

// ── Service Worker ─────────────────────────────────────────────────────────────
function initServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker
      .register('/sw.js', { scope: '/' })
      .then((reg) => {
        console.log('[App] SW registered, scope:', reg.scope);

        // Listen for SW sync triggers
        navigator.serviceWorker.addEventListener('message', (e) => {
          if (e.data?.type === 'SW_SYNC_TRIGGER') {
            window.OfflineQueue?.syncNow?.();
          }
        });
      })
      .catch((err) => console.warn('[App] SW registration failed:', err));
  }
}

// ── Tab Navigation ─────────────────────────────────────────────────────────────
function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}

function switchTab(tabName) {
  const tabs  = document.querySelectorAll('.nav-tab');
  const pages = document.querySelectorAll('.page');

  tabs.forEach((t) => t.classList.toggle('active', t.dataset.tab === tabName));
  pages.forEach((p) => p.classList.toggle('active', p.dataset.page === tabName));

  window.CropGuard.currentTab = tabName;

  if (tabName === 'dashboard') loadDashboard();
}

// ── Online / Offline Status ────────────────────────────────────────────────────
function initOnlineStatus() {
  const badge = document.getElementById('offlineBadge');

  function update() {
    const online = navigator.onLine;
    window.CropGuard.isOnline = online;
    badge?.classList.toggle('visible', !online);

    if (online) {
      showToast('Back online — syncing pending uploads…', 'success');
      window.OfflineQueue?.syncNow?.();
    } else {
      showToast('You are offline. Uploads will be queued.', 'info');
    }
  }

  window.addEventListener('online',  update);
  window.addEventListener('offline', update);

  // Set initial state silently
  window.CropGuard.isOnline = navigator.onLine;
  if (!navigator.onLine) badge?.classList.add('visible');
}

// ── Toast Notifications ────────────────────────────────────────────────────────
window.showToast = function(message, type = 'info', durationMs = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || '🔔'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = '0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, durationMs);
};

// ── Spinner ────────────────────────────────────────────────────────────────────
window.showSpinner = function(text = 'Analyzing crop…') {
  const el = document.getElementById('spinnerOverlay');
  if (el) {
    el.classList.add('visible');
    const textEl = el.querySelector('.spinner-text');
    if (textEl) textEl.textContent = text;
  }
};

window.hideSpinner = function() {
  document.getElementById('spinnerOverlay')?.classList.remove('visible');
};

// ── User Location ──────────────────────────────────────────────────────────────
window.requestLocation = function() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const loc = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        window.CropGuard.userLocation = loc;
        resolve(loc);
      },
      () => resolve(null),
      { timeout: 5000, maximumAge: 60000 }
    );
  });
};

// ── Date Formatting ────────────────────────────────────────────────────────────
window.formatDate = function(isoString) {
  if (!isoString) return '—';
  try {
    return new Intl.DateTimeFormat('en-IN', {
      day:    '2-digit',
      month:  'short',
      year:   'numeric',
      hour:   '2-digit',
      minute: '2-digit',
    }).format(new Date(isoString));
  } catch { return isoString; }
};

window.formatDateShort = function(isoString) {
  if (!isoString) return '—';
  try {
    return new Intl.DateTimeFormat('en-IN', {
      day: '2-digit', month: 'short'
    }).format(new Date(isoString));
  } catch { return isoString; }
};

// ── API helper ─────────────────────────────────────────────────────────────────
window.apiGet = async function(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
};
