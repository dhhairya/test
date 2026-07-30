"""
CropGuard AI — Confidence Abstention Utility

This is the SINGLE, authoritative implementation of the abstention rule.
Every prediction endpoint MUST use check_confidence() before returning a label.
Do NOT re-implement this logic inline anywhere else in the codebase.

Usage:
    from app.services.confidence import check_confidence

    passes, abstain_payload = check_confidence(confidence_score)
    if not passes:
        return jsonify(abstain_payload), 200
    # ... continue with normal response
"""
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def check_confidence(
    score: float,
    threshold: Optional[float] = None,
    context: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate whether model confidence clears the abstention threshold.

    Args:
        score:      Model confidence score in [0.0, 1.0].
        threshold:  Override threshold. If None, reads from Flask app config
                    (CONFIDENCE_THRESHOLD, default 0.85).
        context:    Optional label for logging (e.g. 'crop_classifier', 'progression').

    Returns:
        (passes, payload)
        - passes=True   → confidence is acceptable; proceed with prediction.
        - passes=False  → abstain; return payload directly to client.
                          payload contains 'status', 'confidence', 'threshold',
                          'message', and 'tips'.

    Example:
        passes, payload = check_confidence(0.72)
        # passes=False, payload={'status': 'low_confidence', ...}

        passes, payload = check_confidence(0.91)
        # passes=True, payload={}
    """
    # Resolve threshold — try Flask app config first, fall back to default
    if threshold is None:
        try:
            from flask import current_app
            threshold = current_app.config.get('CONFIDENCE_THRESHOLD', 0.85)
        except RuntimeError:
            # Outside application context (e.g. unit tests)
            threshold = 0.85

    # Clamp score to valid range
    score = max(0.0, min(1.0, float(score)))

    label = f"[{context}] " if context else ""

    if score >= threshold:
        logger.debug(f"{label}Confidence {score:.3f} ≥ threshold {threshold:.3f} → PASS")
        return True, {}

    logger.info(
        f"{label}Confidence {score:.3f} < threshold {threshold:.3f} → ABSTAIN"
    )

    return False, {
        'status':     'low_confidence',
        'confidence': round(score, 4),
        'threshold':  round(threshold, 4),
        'message': (
            'The model is not confident enough to make a reliable prediction. '
            'Please retake the photo following the tips below.'
        ),
        'tips': [
            'Use natural daylight — avoid harsh shadows or direct flash',
            'Hold the camera 15–30 cm from the leaf',
            'Focus on a single leaf that clearly shows symptoms',
            'Ensure the leaf fills most of the frame',
            'Keep the camera steady — blurry images reduce accuracy',
            'Clean the camera lens before capturing',
        ],
    }


def bulk_check_confidence(
    predictions: list,
    threshold: Optional[float] = None,
) -> list:
    """
    Filter a ranked list of predictions, returning only those above threshold.
    Used when the model returns top-k predictions.

    Args:
        predictions: list of dicts, each with a 'confidence' key.
        threshold:   abstention threshold.

    Returns:
        Filtered list — may be empty if all predictions are below threshold.
    """
    if threshold is None:
        try:
            from flask import current_app
            threshold = current_app.config.get('CONFIDENCE_THRESHOLD', 0.85)
        except RuntimeError:
            threshold = 0.85

    return [p for p in predictions if p.get('confidence', 0.0) >= threshold]
