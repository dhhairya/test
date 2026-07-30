"""
CropGuard AI — Yield & Financial Prediction Service (Phase 3)

Uses a scikit-learn Ridge regression model trained on synthetic yield data
derived from FAO crop production statistics and published disease impact studies.

Features used:
  - crop encoding
  - disease severity encoding
  - field_size_ha
  - disease aggressiveness score

To replace with real data:
    1. Collect historical yield records per crop/region/disease/severity
    2. Build features as below
    3. Retrain: python training/yield_model_train.py
    4. Load real checkpoint: joblib.load('models/yield_model.pkl')
"""
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Market prices (USD / metric ton) — update via live API in production ───────
MARKET_PRICES = {
    "Tomato":       250,
    "Potato":       180,
    "Corn":         200,
    "Apple":        800,
    "Grape":        600,
    "Pepper":      1200,
    "Pepper,_bell":1200,
    "Strawberry":  1500,
    "Peach":        700,
    "Cherry":      2000,
    "Soybean":      450,
    "Squash":       300,
    "Orange":       350,
    "Blueberry":   2500,
    "Raspberry":   3000,
    "Rice":         380,
    "Wheat":        290,
    "_default":     400,
}

# Healthy baseline yield (tons / hectare) — FAO global averages (synthetic approximation)
HEALTHY_YIELD_BASELINE = {
    "Tomato":       68.0,
    "Potato":       21.0,
    "Corn":          5.5,
    "Apple":        13.0,
    "Grape":        10.0,
    "Pepper":        8.0,
    "Pepper,_bell":  8.0,
    "Strawberry":   10.0,
    "Peach":         9.0,
    "Cherry":        4.5,
    "Soybean":       2.8,
    "Squash":       16.0,
    "Orange":       18.0,
    "Blueberry":     5.0,
    "Raspberry":     5.5,
    "Rice":          4.5,
    "Wheat":         3.4,
    "_default":     10.0,
}

CROP_CODES = {c: i for i, c in enumerate(HEALTHY_YIELD_BASELINE.keys())}
SEVERITY_CODES = {"none": 0, "early": 1, "moderate": 2, "severe": 3}
SEVERITY_YIELD_LOSS = {"none": 0.00, "early": 0.10, "moderate": 0.30, "severe": 0.55}

# Disease aggressiveness factor (multiplies yield loss)
DISEASE_AGGRESSION = {
    "late blight":    1.5,
    "early blight":   1.0,
    "common rust":    1.3,
    "yellow rust":    1.4,
    "black rot":      1.2,
    "leaf blight":    1.1,
    "bacterial":      0.9,
    "powdery mildew": 0.85,
    "viral":          1.6,
    "mosaic":         1.5,
    "_default":       1.0,
}

# ── Lazy model cache ───────────────────────────────────────────────────────────
_model_cache: Dict = {}


def _get_model():
    """
    Build a Ridge regression model trained on synthetic yield-loss data.
    Features: [crop_code, severity_code, disease_aggression, field_size_ha]
    Target:   yield_loss_fraction (0.0 – 0.95)
    """
    if "reg" in _model_cache:
        return _model_cache["reg"]

    from sklearn.linear_model import Ridge
    import random

    rng = random.Random(2024)
    np.random.seed(2024)
    n_crops = len(HEALTHY_YIELD_BASELINE)

    rows, targets = [], []
    for _ in range(2000):
        crop_code = rng.randint(0, n_crops - 1)
        sev_code  = rng.randint(0, 3)
        aggression = rng.uniform(0.7, 1.8)
        field_ha   = rng.uniform(0.1, 50.0)

        base_loss = [0.0, 0.10, 0.30, 0.55][sev_code]
        loss = base_loss * aggression + np.random.normal(0, 0.03)
        loss = float(np.clip(loss, 0.0, 0.95))

        rows.append([crop_code, sev_code, aggression, field_ha])
        targets.append(loss)

    X = np.array(rows)
    y = np.array(targets)

    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    logger.info("[YieldModel] Ridge regressor trained on %d synthetic samples", len(y))
    _model_cache["reg"] = reg
    return reg


def _disease_aggression(disease: str) -> float:
    d = disease.lower()
    for key, val in DISEASE_AGGRESSION.items():
        if key in d:
            return val
    return DISEASE_AGGRESSION["_default"]


def predict_yield_impact(
    crop: str,
    disease: str,
    severity: str = "moderate",
    field_size_ha: float = 1.0,
    is_healthy: bool = False,
) -> Dict[str, Any]:
    """
    Estimate yield and financial impact of detected disease.

    Args:
        crop:          Crop name
        disease:       Disease name
        severity:      'early' | 'moderate' | 'severe' | 'none'
        field_size_ha: Field area in hectares
        is_healthy:    True → zero disease loss

    Returns:
        {
            'healthy_yield_t_ha':    float,
            'predicted_yield_t_ha':  float,
            'yield_loss_t_ha':       float,
            'yield_loss_pct':        float,
            'revenue_healthy_usd':   float,
            'revenue_predicted_usd': float,
            'financial_loss_usd':    float,
            'price_per_ton_usd':     float,
            'field_size_ha':         float,
            'model':                 str,
            'disclaimer':            str,
        }
    """
    if is_healthy:
        severity = "none"

    baseline   = HEALTHY_YIELD_BASELINE.get(crop, HEALTHY_YIELD_BASELINE["_default"])
    price      = MARKET_PRICES.get(crop, MARKET_PRICES["_default"])
    sev_code   = SEVERITY_CODES.get(severity, 1)
    aggression = _disease_aggression(disease)
    crop_code  = CROP_CODES.get(crop, 0)

    # ML prediction of loss fraction
    try:
        reg = _get_model()
        x   = np.array([[crop_code, sev_code, aggression, field_size_ha]])
        loss_frac = float(np.clip(reg.predict(x)[0], 0.0, 0.95))
    except Exception as e:
        logger.warning("[YieldModel] Model predict failed: %s — using rule-based fallback", e)
        loss_frac = SEVERITY_YIELD_LOSS.get(severity, 0.30) * aggression
        loss_frac = float(np.clip(loss_frac, 0.0, 0.95))

    predicted_yield   = baseline * (1.0 - loss_frac)
    yield_loss        = baseline - predicted_yield
    revenue_healthy   = baseline * field_size_ha * price
    revenue_predicted = predicted_yield * field_size_ha * price
    financial_loss    = yield_loss * field_size_ha * price

    return {
        "healthy_yield_t_ha":    round(baseline, 2),
        "predicted_yield_t_ha":  round(predicted_yield, 2),
        "yield_loss_t_ha":       round(yield_loss, 2),
        "yield_loss_pct":        round(loss_frac * 100, 1),
        "revenue_healthy_usd":   round(revenue_healthy, 2),
        "revenue_predicted_usd": round(revenue_predicted, 2),
        "financial_loss_usd":    round(financial_loss, 2),
        "price_per_ton_usd":     price,
        "field_size_ha":         field_size_ha,
        "severity":              severity,
        "disease_aggression":    round(aggression, 2),
        "model":                 "Ridge regression (synthetic training data)",
        "data_source":           "synthetic_fao_approximation",
        "disclaimer": (
            "SAMPLE DATA: Yield estimates are based on synthetic FAO-approximated baselines "
            "and a Ridge regression model trained on synthetic data. "
            "Replace with real regional yield records and live commodity prices before production use."
        ),
    }
