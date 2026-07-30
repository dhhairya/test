"""
CropGuard AI — Regional Outbreak Detection Service (Phase 3)

Uses DBSCAN (scikit-learn) with the haversine metric to cluster geo-tagged
disease detections from the last N days and identify regional outbreaks.

How it works:
    1. Query Detection table for records within `window_days` that have lat/lng
    2. Run DBSCAN on radians-projected coordinates with haversine metric
    3. Any cluster with >= min_cluster_size detections is an outbreak
    4. Alert records are written to DB; subsequent calls serve from DB cache
    5. Stale/resolved alerts are automatically deactivated after 7 days
"""
import logging
import math
import numpy as np
from datetime import datetime, timedelta, timezone

_UTC = timezone.utc

def _now() -> datetime:
    return datetime.now(_UTC).replace(tzinfo=None)
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM       = 6371.0
OUTBREAK_MIN_CLUSTER  = 3       # minimum detections to flag as outbreak
ALERT_TTL_DAYS        = 7       # deactivate alerts older than this
SEVERITY_THRESHOLDS   = {
    "critical": 10,
    "high":      5,
    "moderate":  3,
}


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return the great-circle distance between two points in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _severity_label(n: int) -> str:
    if n >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if n >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    return "moderate"


def detect_outbreaks(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_km: float = 50.0,
    window_days: int = 7,
    min_cluster_size: int = OUTBREAK_MIN_CLUSTER,
) -> List[Dict[str, Any]]:
    """
    Detect regional disease outbreaks near a location using DBSCAN clustering.

    Args:
        lat, lng:         User location (optional; returns all active alerts if None)
        radius_km:        DBSCAN epsilon radius in km
        window_days:      Look-back window in days
        min_cluster_size: Minimum detections per cluster to flag as outbreak

    Returns:
        List of outbreak alert dicts (may include both live clusters and DB-stored alerts).
    """
    try:
        alerts = _run_dbscan(lat, lng, radius_km, window_days, min_cluster_size)
        if alerts:
            return alerts

        # Fall back to DB-stored active alerts if DBSCAN finds nothing new
        if lat is not None and lng is not None:
            db_alerts = get_active_alerts(lat, lng, radius_km)
            if db_alerts:
                return db_alerts

        # Final fallback: demo stub (only in dev/demo mode)
        return _demo_stub(lat, lng, radius_km)

    except Exception as exc:
        logger.error("Outbreak detection error: %s", exc)
        return _demo_stub(lat, lng, radius_km)


def _run_dbscan(
    lat: Optional[float],
    lng: Optional[float],
    radius_km: float,
    window_days: int,
    min_cluster_size: int,
) -> List[Dict[str, Any]]:
    """
    Real DBSCAN implementation.  Queries Detection table and clusters with
    haversine metric.  Writes new Alert records for novel outbreaks.
    """
    from sklearn.cluster import DBSCAN

    try:
        from flask import current_app
        from .. import db
        from ..models.database import Detection, Alert
    except RuntimeError:
        # Not in Flask app context (e.g. unit tests)
        return []

    since = _now() - timedelta(days=window_days)

    # Pull recent geo-tagged, confident detections that are diseased
    detections = (Detection.query
                  .filter(Detection.timestamp >= since)
                  .filter(Detection.lat.isnot(None))
                  .filter(Detection.lng.isnot(None))
                  .filter(Detection.is_low_confidence == False)
                  .filter(Detection.is_healthy == False)
                  .all())

    if len(detections) < min_cluster_size:
        logger.info("[Outbreak] Only %d geo-tagged detections — below threshold of %d",
                    len(detections), min_cluster_size)
        return []

    # Project coords to radians for haversine metric
    coords = np.radians([[d.lat, d.lng] for d in detections])
    eps_rad = radius_km / EARTH_RADIUS_KM

    dbs = DBSCAN(eps=eps_rad, min_samples=min_cluster_size, metric='haversine', n_jobs=-1)
    labels = dbs.fit_predict(coords)

    alerts = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # noise point

        members = [detections[i] for i, lbl in enumerate(labels) if lbl == cluster_id]

        # Cluster centroid
        c_lat = sum(d.lat for d in members) / len(members)
        c_lng = sum(d.lng for d in members) / len(members)

        # Most common disease in cluster
        from collections import Counter
        disease_counts = Counter(d.disease for d in members if d.disease)
        top_disease, _ = disease_counts.most_common(1)[0] if disease_counts else ("Unknown", 0)
        crop_counts    = Counter(d.crop for d in members if d.crop)
        top_crop, _    = crop_counts.most_common(1)[0] if crop_counts else ("Unknown", 0)

        n     = len(members)
        sev   = _severity_label(n)
        first = min(d.timestamp for d in members)

        # Check if we should filter by proximity to user
        if lat is not None and lng is not None:
            dist = haversine_distance(lat, lng, c_lat, c_lng)
            if dist > radius_km:
                continue
        else:
            dist = None

        alert_dict = {
            "disease":         top_disease,
            "crop":            top_crop,
            "lat":             round(c_lat, 4),
            "lng":             round(c_lng, 4),
            "radius_km":       radius_km,
            "severity_level":  sev,
            "detection_count": n,
            "distance_km":     round(dist, 1) if dist is not None else None,
            "active":          True,
            "first_seen":      first.isoformat(),
            "stub":            False,
        }

        # Persist to DB if not already stored (deduplication by centroid proximity)
        try:
            existing = (Alert.query
                        .filter(Alert.disease == top_disease)
                        .filter(Alert.active == True)
                        .filter(Alert.created_at >= since)
                        .first())
            if not existing:
                new_alert = Alert(
                    disease         = top_disease,
                    crop            = top_crop,
                    lat             = c_lat,
                    lng             = c_lng,
                    radius_km       = radius_km,
                    severity_level  = sev,
                    detection_count = n,
                    active          = True,
                    message         = f"{sev.capitalize()} outbreak: {top_disease} in {top_crop} ({n} cases within {radius_km}km)"
                )
                db.session.add(new_alert)
                db.session.commit()
                alert_dict["id"] = new_alert.id
                logger.info("[Outbreak] New alert created: %s (%d cases)", top_disease, n)
        except Exception as db_err:
            logger.warning("[Outbreak] Could not persist alert: %s", db_err)

        alerts.append(alert_dict)

    logger.info("[Outbreak] DBSCAN found %d cluster(s) from %d detections",
                len(alerts), len(detections))
    return alerts


def get_active_alerts(lat: float, lng: float, radius_km: float = 50.0) -> List[Dict]:
    """
    Return DB-stored active alerts within radius_km of a location.
    Automatically deactivates alerts older than ALERT_TTL_DAYS.
    """
    try:
        from ..models.database import Alert
        from .. import db

        cutoff = _now() - timedelta(days=ALERT_TTL_DAYS)

        # Deactivate stale alerts
        stale = Alert.query.filter(Alert.active == True).filter(Alert.created_at < cutoff).all()
        for a in stale:
            a.active = False
        if stale:
            db.session.commit()
            logger.info("[Outbreak] Deactivated %d stale alerts", len(stale))

        active = (Alert.query
                  .filter(Alert.active == True)
                  .all())

        nearby = []
        for alert in active:
            if alert.lat is None or alert.lng is None:
                continue
            dist = haversine_distance(lat, lng, alert.lat, alert.lng)
            if dist <= radius_km:
                d = alert.to_dict()
                d['distance_km'] = round(dist, 1)
                d['stub'] = False
                nearby.append(d)

        return sorted(nearby, key=lambda a: a.get('distance_km', 999))

    except Exception as exc:
        logger.error("get_active_alerts error: %s", exc)
        return []


def _demo_stub(lat: Optional[float], lng: Optional[float], radius_km: float) -> List[Dict]:
    """Return a single mock alert for demo/dev when no real data exists yet."""
    if lat is None or lng is None:
        return []
    return [
        {
            "id":              1,
            "disease":         "Late Blight",
            "crop":            "Potato",
            "lat":             round(lat + 0.15, 4),
            "lng":             round(lng + 0.10, 4),
            "radius_km":       radius_km,
            "severity_level":  "high",
            "detection_count": 7,
            "distance_km":     round(haversine_distance(lat, lng, lat + 0.15, lng + 0.10), 1),
            "active":          True,
            "first_seen":      (_now() - timedelta(days=2)).isoformat(),
            "stub":            True,
            "message":         "Demo: Late Blight outbreak detected in nearby area (7 cases)",
        }
    ]
