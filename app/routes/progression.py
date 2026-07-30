"""
CropGuard AI — Disease Progression Routes (Phase 2)
"""
import logging
from flask import Blueprint, jsonify, request
from ..services.progression import predict_progression
from ..services.weather import get_cached_forecast
from ..services.confidence import check_confidence
from ..models.database import Detection

logger = logging.getLogger(__name__)

progression_bp = Blueprint('progression', __name__)


@progression_bp.route('/api/progression/<int:detection_id>', methods=['GET'])
def get_progression(detection_id):
    """
    GET /api/progression/:detection_id
    Returns 7/14/30-day disease progression forecast for a detection.
    Uses confidence abstention — low-confidence detections are rejected.
    """
    detection = Detection.query.get_or_404(detection_id)

    if detection.is_low_confidence or not detection.disease:
        return jsonify({'error': 'Progression prediction requires a confident detection.'}), 400

    # Re-check confidence using same threshold
    passes, abstain = check_confidence(detection.confidence or 0.0, context='progression')
    if not passes:
        return jsonify(abstain), 200

    # Fetch weather if location is available
    weather = None
    if detection.lat and detection.lng:
        try:
            weather = get_cached_forecast(detection.lat, detection.lng)
        except Exception as exc:
            logger.warning(f"Weather fetch failed: {exc}")

    days_since = 0
    from datetime import datetime
    if detection.timestamp:
        days_since = max(0, (datetime.utcnow() - detection.timestamp).days)

    result = predict_progression(
        disease              = detection.disease,
        days_since_detection = days_since,
        weather              = weather,
        current_severity     = detection.severity or 'early',
    )

    return jsonify({
        'detection_id': detection_id,
        'crop':         detection.crop,
        'disease':      detection.disease,
        'progression':  result,
        'weather':      weather,
    })
