"""
CropGuard AI — /predict Endpoint

POST /predict
Accepts a crop leaf photo, runs the CNN classifier, applies the
confidence abstention check, persists the detection, and returns JSON.

All confidence-based abstention logic is delegated to:
    app.services.confidence.check_confidence()
Do NOT inline threshold comparisons here.
"""
import os
import uuid
import logging
from datetime import datetime, timedelta, timezone

_UTC = timezone.utc

def _now() -> datetime:
    """Return current UTC time as a naive datetime (for SQLite compatibility)."""
    return datetime.now(_UTC).replace(tzinfo=None)

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from ..services.classifier import get_classifier, validate_image
from ..services.confidence import check_confidence
from ..models.database import Detection, db

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _estimate_severity(confidence: float) -> str:
    """
    Rough initial severity estimate from confidence level.
    In production this should come from a dedicated severity-estimation head
    or from the disease progression service.
    """
    if confidence >= 0.93:
        return 'severe'
    elif confidence >= 0.88:
        return 'moderate'
    else:
        return 'early'


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict

    Form fields:
        image         (file, required)   — leaf/crop photo
        lat           (float, optional)  — user's latitude
        lng           (float, optional)  — user's longitude
        location_name (str,   optional)  — human-readable location

    Response (success, confidence ≥ threshold):
        { status, detection_id, crop, disease, is_healthy,
          confidence, severity, timestamp, demo_mode }

    Response (low confidence):
        { status: 'low_confidence', confidence, threshold, message, tips }

    Response (error):
        { error: <string> }
    """

    # ── 1. Validate file presence ─────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Send the photo as multipart/form-data with key "image".'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            'error': f'Unsupported file type. Accepted formats: {", ".join(ALLOWED_EXTENSIONS).upper()}.'
        }), 415

    # ── 2. Read and validate image bytes ─────────────────────────────────────
    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({'error': 'Uploaded file is empty.'}), 400

    if len(file_bytes) > current_app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024):
        return jsonify({'error': 'File too large. Maximum size is 16 MB.'}), 413

    is_valid, err_msg, pil_image = validate_image(file_bytes)
    if not is_valid:
        return jsonify({'error': err_msg}), 422

    # ── 3. Save upload ────────────────────────────────────────────────────────
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    ext      = secure_filename(file.filename).rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(upload_folder, filename)

    try:
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
    except OSError as exc:
        logger.error(f"Failed to save upload: {exc}")
        return jsonify({'error': 'Server error: could not save image. Try again.'}), 500

    # ── 4. Run classifier ─────────────────────────────────────────────────────
    try:
        classifier = get_classifier()
        result     = classifier.predict(pil_image)
    except Exception as exc:
        logger.error(f"Classifier error: {exc}")
        return jsonify({'error': 'Model inference failed. Please try again.'}), 500

    confidence   = result['confidence']
    lat          = request.form.get('lat',  type=float)
    lng          = request.form.get('lng',  type=float)
    loc_name     = request.form.get('location_name', '')
    demo_flag    = result.get('demo_mode', False)

    # ── 5. Confidence abstention check (THE authoritative check) ──────────────
    passes, abstain_payload = check_confidence(confidence, context='crop_classifier')

    if not passes:
        # Log the low-confidence attempt (no crop/disease stored)
        _save_detection(
            filename=filename, save_path=save_path,
            crop=None, disease=None, raw_class=None,
            confidence=confidence, is_healthy=False,
            is_low_confidence=True, severity=None,
            lat=lat, lng=lng, location_name=loc_name,
            demo_mode=demo_flag
        )
        return jsonify(abstain_payload), 200

    # ── 6. Build and save successful detection ────────────────────────────────
    severity = 'none' if result['is_healthy'] else _estimate_severity(confidence)

    detection = _save_detection(
        filename=filename, save_path=save_path,
        crop=result['crop'], disease=result['disease'],
        raw_class=result.get('raw_class'),
        confidence=confidence, is_healthy=result['is_healthy'],
        is_low_confidence=False, severity=severity,
        lat=lat, lng=lng, location_name=loc_name,
        demo_mode=demo_flag
    )

    return jsonify({
        'status':        'success',
        'detection_id':  detection.id,
        'crop':          result['crop'],
        'disease':       result['disease'],
        'description':   result.get('description', ''),
        'is_healthy':    result['is_healthy'],
        'confidence':    confidence,
        'severity':      severity,
        'lat':           lat,
        'lng':           lng,
        'location_name': loc_name,
        'timestamp':     detection.timestamp.isoformat(),
        'demo_mode':     demo_flag,
    }), 200


def _save_detection(**kwargs) -> Detection:
    """Helper: create and commit a Detection record."""
    detection = Detection(
        image_filename  = kwargs.get('filename'),
        image_path      = kwargs.get('save_path'),
        crop            = kwargs.get('crop'),
        disease         = kwargs.get('disease'),
        raw_class       = kwargs.get('raw_class'),
        confidence      = kwargs.get('confidence'),
        is_healthy      = kwargs.get('is_healthy', False),
        is_low_confidence = kwargs.get('is_low_confidence', False),
        severity        = kwargs.get('severity'),
        lat             = kwargs.get('lat'),
        lng             = kwargs.get('lng'),
        location_name   = kwargs.get('location_name'),
        demo_mode       = kwargs.get('demo_mode', False),
        timestamp       = _now(),
        synced          = True,
    )
    db.session.add(detection)
    db.session.commit()
    return detection
