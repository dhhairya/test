"""
CropGuard AI — Dashboard API Routes

Endpoints for fetching detection history, timeline aggregates,
and per-detection details for the Chart.js dashboard.
"""
import logging
from datetime import datetime, timedelta, timezone

_UTC = timezone.utc

def _now() -> datetime:
    return datetime.now(_UTC).replace(tzinfo=None)

from flask import Blueprint, jsonify, request, render_template

from ..models.database import Detection, db
from sqlalchemy import desc, func

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def index():
    """Serve the main SPA."""
    return render_template('index.html')


@dashboard_bp.route('/api/detections', methods=['GET'])
def get_detections():
    """
    GET /api/detections
    Query params:
        limit (int, default 50)
        days  (int, default 30)  — look-back window
        crop  (str, optional)    — filter by crop name
    """
    limit  = request.args.get('limit', 50, type=int)
    days   = request.args.get('days',  30, type=int)
    crop   = request.args.get('crop',  None)

    since  = _now() - timedelta(days=days)
    query  = (Detection.query
              .filter(Detection.timestamp >= since)
              .filter(Detection.is_low_confidence == False)
              .filter(Detection.crop.isnot(None)))

    if crop:
        query = query.filter(Detection.crop.ilike(f'%{crop}%'))

    detections = query.order_by(desc(Detection.timestamp)).limit(limit).all()

    return jsonify({
        'detections': [d.to_dict() for d in detections],
        'total':      len(detections),
    })


@dashboard_bp.route('/api/timeline', methods=['GET'])
def get_timeline():
    """
    GET /api/timeline
    Returns aggregated data for Chart.js charts.

    Query params:
        days (int, default 30)

    Response:
        {
            timeline:          [{ date, total, diseased, healthy }, ...],
            disease_breakdown: [{ disease, count }, ...],
            crop_breakdown:    [{ crop, count }, ...],
            severity_breakdown:[{ severity, count }, ...],
            summary:           { total, healthy, diseased, health_rate, avg_confidence }
        }
    """
    days  = request.args.get('days', 30, type=int)
    since = _now() - timedelta(days=days)

    detections = (Detection.query
                  .filter(Detection.timestamp >= since)
                  .filter(Detection.is_low_confidence == False)
                  .filter(Detection.crop.isnot(None))
                  .order_by(Detection.timestamp)
                  .all())

    # ── Aggregate by date ─────────────────────────────────────────────────────
    by_date: dict = {}
    disease_counts: dict = {}
    crop_counts: dict    = {}
    severity_counts: dict = {}
    conf_sum = 0.0

    for d in detections:
        date_str = d.timestamp.strftime('%Y-%m-%d')
        entry = by_date.setdefault(date_str, {'date': date_str, 'total': 0, 'diseased': 0, 'healthy': 0})
        entry['total'] += 1

        if d.is_healthy:
            entry['healthy'] += 1
        else:
            entry['diseased'] += 1
            if d.disease:
                disease_counts[d.disease] = disease_counts.get(d.disease, 0) + 1

        if d.crop:
            crop_counts[d.crop] = crop_counts.get(d.crop, 0) + 1

        if d.severity:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1

        conf_sum += d.confidence or 0.0

    total    = len(detections)
    healthy  = sum(1 for d in detections if d.is_healthy)
    diseased = total - healthy

    return jsonify({
        'timeline': list(by_date.values()),
        'disease_breakdown': [
            {'disease': k, 'count': v}
            for k, v in sorted(disease_counts.items(), key=lambda x: -x[1])[:10]
        ],
        'crop_breakdown': [
            {'crop': k, 'count': v}
            for k, v in sorted(crop_counts.items(), key=lambda x: -x[1])
        ],
        'severity_breakdown': [
            {'severity': k, 'count': v}
            for k, v in severity_counts.items()
        ],
        'summary': {
            'total':           total,
            'healthy':         healthy,
            'diseased':        diseased,
            'health_rate':     round(healthy / total * 100, 1) if total > 0 else 0,
            'avg_confidence':  round(conf_sum / total, 3) if total > 0 else 0,
        },
    })


@dashboard_bp.route('/api/detections/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """GET /api/detections/:id — single detection detail."""
    d = Detection.query.get_or_404(detection_id)
    return jsonify(d.to_dict())


@dashboard_bp.route('/api/detections/sync', methods=['POST'])
def sync_detections():
    """
    POST /api/detections/sync
    Accepts a batch of offline-queued detections for sync.
    Body: { detections: [ { image_b64, lat, lng, timestamp, ... }, ... ] }

    NOTE: Full offline sync with binary image upload is handled by the
    frontend's offline.js via the standard /predict endpoint.
    This endpoint handles lightweight metadata-only sync.
    """
    data       = request.get_json(force=True, silent=True) or {}
    batch      = data.get('detections', [])
    synced_ids = []

    for item in batch:
        # Minimal metadata sync — images must be re-submitted via /predict
        detection = Detection(
            crop            = item.get('crop'),
            disease         = item.get('disease'),
            confidence      = item.get('confidence', 0.0),
            is_healthy      = item.get('is_healthy', False),
            severity        = item.get('severity'),
            lat             = item.get('lat'),
            lng             = item.get('lng'),
            location_name   = item.get('location_name'),
            timestamp       = datetime.fromisoformat(item['timestamp']) if 'timestamp' in item else datetime.utcnow(),
            synced          = True,
        )
        db.session.add(detection)
        db.session.flush()
        synced_ids.append(detection.id)

    db.session.commit()
    return jsonify({'synced': len(synced_ids), 'ids': synced_ids}), 200
