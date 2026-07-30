"""
CropGuard AI — Preventive Recommendations Routes (Phase 2)
"""
import logging
from flask import Blueprint, jsonify, request
from ..services.recommendations import get_recommendations
from ..services.weather import get_cached_forecast
from ..models.database import Detection

logger = logging.getLogger(__name__)

recommendations_bp = Blueprint('recommendations', __name__)


@recommendations_bp.route('/api/recommendations/<int:detection_id>', methods=['GET'])
def get_detection_recommendations(detection_id):
    """
    GET /api/recommendations/:detection_id
    Query params:
        soil_type (str, default 'loamy')

    Returns prioritized preventive action recommendations for a detection.
    """
    detection = Detection.query.get_or_404(detection_id)

    if not detection.crop:
        return jsonify({'error': 'No valid detection found for this ID.'}), 400

    soil_type = request.args.get('soil_type', 'loamy')

    weather = None
    if detection.lat and detection.lng:
        try:
            weather = get_cached_forecast(detection.lat, detection.lng)
        except Exception as exc:
            logger.warning(f"Weather fetch failed: {exc}")

    result = get_recommendations(
        disease    = detection.disease or '',
        crop       = detection.crop or '',
        soil_type  = soil_type,
        weather    = weather,
        is_healthy = detection.is_healthy,
    )

    return jsonify({
        'detection_id': detection_id,
        'crop':         detection.crop,
        'disease':      detection.disease,
        'soil_type':    soil_type,
        'recommendations': result,
    })


@recommendations_bp.route('/api/recommendations', methods=['POST'])
def get_quick_recommendations():
    """
    POST /api/recommendations
    Body: { disease, crop, soil_type, lat, lng }

    Quick recommendations without requiring a stored detection.
    Used by the frontend when the user wants to explore scenarios.
    """
    data = request.get_json(force=True, silent=True) or {}
    disease   = data.get('disease', '')
    crop      = data.get('crop', '')
    soil_type = data.get('soil_type', 'loamy')
    lat       = data.get('lat')
    lng       = data.get('lng')

    weather = None
    if lat and lng:
        try:
            weather = get_cached_forecast(lat, lng)
        except Exception:
            pass

    result = get_recommendations(
        disease    = disease,
        crop       = crop,
        soil_type  = soil_type,
        weather    = weather,
        is_healthy = disease.lower() in ('healthy', ''),
    )

    return jsonify({'recommendations': result})
