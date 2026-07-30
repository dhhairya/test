"""
CropGuard AI — Alerts & Yield Routes (Phase 3)
"""
import logging
from flask import Blueprint, jsonify, request
from ..services.outbreak import detect_outbreaks, get_active_alerts
from ..services.yield_model import predict_yield_impact
from ..models.database import Detection

logger = logging.getLogger(__name__)

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    GET /api/alerts
    Query params:
        lat       (float, required)
        lng       (float, required)
        radius_km (float, default 50)
    """
    lat       = request.args.get('lat',       type=float)
    lng       = request.args.get('lng',       type=float)
    radius_km = request.args.get('radius_km', type=float, default=50.0)

    if lat is None or lng is None:
        return jsonify({'error': 'lat and lng are required query parameters.'}), 400

    # Combine stored alerts + real-time outbreak detection
    stored_alerts   = get_active_alerts(lat, lng, radius_km)
    detected_alerts = detect_outbreaks(lat, lng, radius_km)

    # Deduplicate (stored alerts take precedence)
    all_alerts = stored_alerts + [a for a in detected_alerts if a.get('stub')]

    return jsonify({
        'alerts':    all_alerts,
        'total':     len(all_alerts),
        'radius_km': radius_km,
    })


@alerts_bp.route('/api/yield/<int:detection_id>', methods=['GET'])
def get_yield_prediction(detection_id):
    """
    GET /api/yield/:detection_id
    Query params:
        field_size_ha (float, default 1.0)
    """
    detection     = Detection.query.get_or_404(detection_id)
    field_size_ha = request.args.get('field_size_ha', 1.0, type=float)

    if not detection.crop:
        return jsonify({'error': 'No valid crop detection found.'}), 400

    result = predict_yield_impact(
        crop          = detection.crop or '',
        disease       = detection.disease or '',
        severity      = detection.severity or 'moderate',
        field_size_ha = field_size_ha,
        is_healthy    = detection.is_healthy,
    )

    return jsonify({
        'detection_id':  detection_id,
        'crop':          detection.crop,
        'disease':       detection.disease,
        'severity':      detection.severity,
        'yield_forecast': result,
    })
