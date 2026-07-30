/**
 * CropGuard AI — Offline Queue Module
 * Uses IndexedDB to store pending detections when the user is offline.
 * Automatically syncs when connectivity is restored.
 */

const DB_NAME    = 'cropguard-offline';
const DB_VERSION = 1;
const STORE_NAME = 'pending-detections';

let db = null;

// ── Init ───────────────────────────────────────────────────────────────────────
window.initOfflineModule = function() {
  openDB()
    .then(() => {
      console.log('[Offline] IndexedDB ready');
      updateQueueBadge();

      // Sync when coming online
      window.addEventListener('online', () => {
        console.log('[Offline] Online — syncing queue…');
        syncNow();
      });

      // Handle SW sync trigger
      window.addEventListener('message', (e) => {
        if (e.data?.type === 'SW_SYNC_TRIGGER') syncNow();
      });
    })
    .catch((err) => console.error('[Offline] IndexedDB error:', err));

  // Expose API on the global OfflineQueue object
  window.OfflineQueue = { enqueue, syncNow, getPendingCount };
};

// ── IndexedDB open ─────────────────────────────────────────────────────────────
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (e) => {
      const idb = e.target.result;
      if (!idb.objectStoreNames.contains(STORE_NAME)) {
        const store = idb.createObjectStore(STORE_NAME, {
          keyPath:       'id',
          autoIncrement: true,
        });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };

    req.onsuccess = (e) => {
      db = e.target.result;
      resolve(db);
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

// ── Enqueue a file for later upload ───────────────────────────────────────────
async function enqueue(file) {
  if (!db) return false;

  try {
    const arrayBuffer = await file.arrayBuffer();
    const record = {
      fileName:  file.name,
      fileType:  file.type,
      fileData:  arrayBuffer,
      lat:       window.CropGuard?.userLocation?.lat ?? null,
      lng:       window.CropGuard?.userLocation?.lng ?? null,
      timestamp: new Date().toISOString(),
      attempts:  0,
    };

    await idbPut(record);
    updateQueueBadge();
    console.log('[Offline] Detection queued:', file.name);
    return true;
  } catch (err) {
    console.error('[Offline] Enqueue error:', err);
    return false;
  }
}

// ── Sync all queued detections ─────────────────────────────────────────────────
window.syncNow = async function syncNow() {
  if (!navigator.onLine || !db) return;

  const pending = await getAllPending();
  if (pending.length === 0) return;

  console.log(`[Offline] Syncing ${pending.length} queued detection(s)…`);
  showToast(`Syncing ${pending.length} offline detection(s)…`, 'info');

  let synced = 0;

  for (const record of pending) {
    try {
      const blob     = new Blob([record.fileData], { type: record.fileType });
      const file     = new File([blob], record.fileName, { type: record.fileType });
      const formData = new FormData();
      formData.append('image', file);

      if (record.lat) formData.append('lat', record.lat);
      if (record.lng) formData.append('lng', record.lng);

      const resp = await fetch('/predict', { method: 'POST', body: formData });

      if (resp.ok || resp.status === 200) {
        await idbDelete(record.id);
        synced++;
      } else {
        await idbUpdate(record.id, { attempts: (record.attempts || 0) + 1 });
      }
    } catch (err) {
      console.warn('[Offline] Sync failed for record', record.id, err);
      await idbUpdate(record.id, { attempts: (record.attempts || 0) + 1 });
    }
  }

  updateQueueBadge();

  if (synced > 0) {
    showToast(`✅ Synced ${synced} detection(s) successfully.`, 'success');
    // Refresh dashboard if it's open
    if (window.CropGuard?.currentTab === 'dashboard') {
      setTimeout(() => window.loadDashboard?.(), 500);
    }
  }

  // Remove records that have failed too many times
  const stillPending = await getAllPending();
  for (const r of stillPending) {
    if ((r.attempts || 0) >= 5) {
      await idbDelete(r.id);
      console.warn('[Offline] Dropped record after 5 failed attempts:', r.id);
    }
  }
};

async function getPendingCount() {
  if (!db) return 0;
  const all = await getAllPending();
  return all.length;
}

// ── Update queue badge in UI ───────────────────────────────────────────────────
async function updateQueueBadge() {
  const count   = await getPendingCount();
  const badge   = document.getElementById('queueBadge');
  if (!badge) return;
  badge.textContent = count > 0 ? `${count} queued` : '';
  badge.style.display = count > 0 ? 'inline' : 'none';
}

// ── IndexedDB helpers ──────────────────────────────────────────────────────────
function idbPut(record) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const req = tx.objectStore(STORE_NAME).add(record);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

function idbDelete(id) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const req = tx.objectStore(STORE_NAME).delete(id);
    req.onsuccess = () => resolve();
    req.onerror   = () => reject(req.error);
  });
}

function idbUpdate(id, updates) {
  return new Promise((resolve, reject) => {
    const tx    = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const getReq = store.get(id);
    getReq.onsuccess = () => {
      const updated = { ...getReq.result, ...updates };
      const putReq  = store.put(updated);
      putReq.onsuccess = () => resolve();
      putReq.onerror   = () => reject(putReq.error);
    };
    getReq.onerror = () => reject(getReq.error);
  });
}

function getAllPending() {
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE_NAME, 'readonly');
    const req = tx.objectStore(STORE_NAME).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}
