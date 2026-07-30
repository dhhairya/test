/**
 * CropGuard AI — Service Worker
 * Handles offline caching and background sync of detection uploads.
 */

const CACHE_VERSION = 'cropguard-v1';
const STATIC_CACHE  = `${CACHE_VERSION}-static`;
const API_CACHE     = `${CACHE_VERSION}-api`;

const STATIC_ASSETS = [
  '/',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/js/upload.js',
  '/static/js/dashboard.js',
  '/static/js/offline.js',
  '/manifest.json',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js',
];

const CACHEABLE_API = [
  '/api/timeline',
  '/api/detections',
  '/api/alerts',
];

// ── Install ───────────────────────────────────────────────────────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('[SW] Some static assets failed to cache:', err);
      });
    })
  );
  self.skipWaiting();
});

// ── Activate (clean old caches) ────────────────────────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith('cropguard-') && k !== STATIC_CACHE && k !== API_CACHE)
          .map((k) => {
            console.log('[SW] Deleting old cache:', k);
            return caches.delete(k);
          })
      )
    )
  );
  self.clients.claim();
});

// ── Fetch Strategy ────────────────────────────────────────────────────────────
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests except /predict (handled by background sync)
  if (request.method !== 'GET') return;

  // Skip chrome-extension and non-http requests
  if (!url.protocol.startsWith('http')) return;

  // API routes: Network-first, cache fallback
  if (isApiRequest(url)) {
    event.respondWith(networkFirstWithCache(request));
    return;
  }

  // Static assets: Cache-first, network fallback
  event.respondWith(cacheFirstWithNetwork(request));
});

// ── Background Sync ───────────────────────────────────────────────────────────
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-detections') {
    console.log('[SW] Background sync: uploading queued detections');
    event.waitUntil(syncQueuedDetections());
  }
});

// ── Push Notifications (Phase 3 hook) ─────────────────────────────────────────
self.addEventListener('push', (event) => {
  if (!event.data) return;
  const data = event.data.json();
  event.waitUntil(
    self.registration.showNotification('🌿 CropGuard Alert', {
      body:  data.message || 'A disease outbreak has been detected near your area.',
      icon:  '/static/icons/icon-192.png',
      badge: '/static/icons/badge-72.png',
      tag:   'outbreak-alert',
      data:  { url: data.url || '/' },
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(clients.openWindow(event.notification.data?.url || '/'));
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function isApiRequest(url) {
  return CACHEABLE_API.some((path) => url.pathname.startsWith(path));
}

async function networkFirstWithCache(request) {
  const cache = await caches.open(API_CACHE);
  try {
    const networkResponse = await fetch(request.clone());
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch {
    const cached = await cache.match(request);
    if (cached) {
      console.log('[SW] Offline: serving cached API response for', request.url);
      return cached;
    }
    return new Response(JSON.stringify({
      error: 'You are offline. Cached data not available.',
      offline: true,
    }), { status: 503, headers: { 'Content-Type': 'application/json' } });
  }
}

async function cacheFirstWithNetwork(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const networkResponse = await fetch(request.clone());
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch {
    // Return offline page fallback
    const offlineFallback = await caches.match('/');
    return offlineFallback || new Response('Offline', { status: 503 });
  }
}

async function syncQueuedDetections() {
  // Read queued uploads from IndexedDB and replay them against /predict
  // The main logic lives in offline.js — this is the SW side trigger.
  const clients = await self.clients.matchAll({ type: 'window' });
  clients.forEach((client) => {
    client.postMessage({ type: 'SW_SYNC_TRIGGER' });
  });
}
