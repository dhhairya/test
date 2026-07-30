"""
CropGuard AI — Preventive Measures Engine (Phase 2)

STATUS: STUB — rule-based engine with hardcoded rules.
Replace or augment with an ML model trained on agronomic intervention data.

Architecture:
    Input:  disease + crop + soil_type + weather_forecast + season
    Logic:  Rule lookup → modifiers from weather/soil → ranked recommendations
    Output: Ordered list of recommended actions + pesticide forecast

To upgrade:
    1. Collect labeled data: (disease, weather, soil) → intervention_outcome
    2. Train a ranking model (XGBoost / LightGBM works well for tabular rule data)
    3. Load it here and replace get_recommendations() body.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ── Recommendation rules ───────────────────────────────────────────────────────
# Each rule entry: { action, timing, priority, conditions }
# conditions = dict of optional filters (soil_type, weather_factor, season)
DISEASE_RULES: Dict[str, List[Dict]] = {
    "Late_blight": [
        {"action": "Apply copper-based fungicide (e.g., Bordeaux mixture)", "timing": "Immediately", "priority": "critical", "type": "fungicide"},
        {"action": "Remove and destroy infected plant material", "timing": "Within 24 hours", "priority": "critical", "type": "cultural"},
        {"action": "Avoid overhead irrigation — switch to drip irrigation", "timing": "Immediately", "priority": "high", "type": "irrigation"},
        {"action": "Apply mancozeb fungicide as preventive spray", "timing": "Every 7 days", "priority": "high", "type": "fungicide"},
        {"action": "Improve field drainage to reduce humidity", "timing": "Within 1 week", "priority": "medium", "type": "drainage"},
    ],
    "Early_blight": [
        {"action": "Apply chlorothalonil or mancozeb fungicide", "timing": "Within 3 days", "priority": "high", "type": "fungicide"},
        {"action": "Remove lower infected leaves to slow spread", "timing": "Immediately", "priority": "high", "type": "cultural"},
        {"action": "Increase plant spacing for better air circulation", "timing": "Next planting cycle", "priority": "medium", "type": "cultural"},
        {"action": "Apply organic mulch to prevent soil splash", "timing": "Within 1 week", "priority": "medium", "type": "cultural"},
    ],
    "Bacterial_spot": [
        {"action": "Apply copper hydroxide bactericide spray", "timing": "Within 48 hours", "priority": "high", "type": "bactericide"},
        {"action": "Avoid working in field when plants are wet", "timing": "Immediately", "priority": "high", "type": "cultural"},
        {"action": "Use certified disease-free seeds for next crop", "timing": "Next season", "priority": "medium", "type": "cultural"},
    ],
    "Powdery_mildew": [
        {"action": "Apply sulfur-based fungicide or neem oil", "timing": "Within 2 days", "priority": "high", "type": "fungicide"},
        {"action": "Improve air circulation by pruning dense foliage", "timing": "Within 1 week", "priority": "medium", "type": "cultural"},
        {"action": "Avoid excessive nitrogen fertilization", "timing": "Immediately", "priority": "medium", "type": "fertilizer"},
        {"action": "Apply potassium bicarbonate solution", "timing": "Weekly", "priority": "low", "type": "fungicide"},
    ],
    "Common_rust": [
        {"action": "Apply triazole-based fungicide (e.g., propiconazole)", "timing": "Within 48 hours", "priority": "critical", "type": "fungicide"},
        {"action": "Scout neighboring fields for spread monitoring", "timing": "Daily", "priority": "high", "type": "monitoring"},
        {"action": "Consider planting rust-resistant varieties next season", "timing": "Next season", "priority": "medium", "type": "cultural"},
    ],
    "healthy": [
        {"action": "Continue current crop management practices", "timing": "Ongoing", "priority": "low", "type": "monitoring"},
        {"action": "Apply preventive fungicide spray before monsoon season", "timing": "Seasonal", "priority": "low", "type": "fungicide"},
        {"action": "Monitor weekly for early disease signs", "timing": "Weekly", "priority": "low", "type": "monitoring"},
    ],
    "_default": [
        {"action": "Consult a local agronomist for diagnosis confirmation", "timing": "Within 48 hours", "priority": "high", "type": "consultation"},
        {"action": "Isolate affected plants to prevent spread", "timing": "Immediately", "priority": "high", "type": "cultural"},
        {"action": "Document symptoms with photos for expert review", "timing": "Immediately", "priority": "medium", "type": "monitoring"},
    ],
}

# Soil-type modifiers
SOIL_ADJUSTMENTS = {
    "clay":   {"drainage_note": "Clay soil retains water — prioritize drainage improvements."},
    "sandy":  {"drainage_note": "Sandy soil drains fast — increase irrigation frequency."},
    "loamy":  {"drainage_note": None},
    "silty":  {"drainage_note": "Silty soil is prone to compaction — avoid heavy machinery."},
    "peaty":  {"drainage_note": "Peaty soil is acidic — check pH before applying chemicals."},
}

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def get_recommendations(
    disease: str,
    crop: str,
    soil_type: str = "loamy",
    weather: Optional[Dict] = None,
    is_healthy: bool = False,
) -> Dict[str, Any]:
    """
    Generate prioritized preventive action recommendations.

    Args:
        disease:    Disease name from classifier
        crop:       Crop name
        soil_type:  User-selected soil type
        weather:    Weather forecast dict
        is_healthy: True if no disease detected

    Returns:
        {
            'recommendations': [ { action, timing, priority, type }, ... ],
            'pesticide_forecast': { type, timing_window, notes },
            'soil_note': str,
            'weather_note': str,
            'stub': True
        }
    """
    logger.info(f"[STUB] get_recommendations(disease={disease}, crop={crop}, soil={soil_type})")

    if is_healthy:
        disease_key = "healthy"
    else:
        disease_key = "_default"
        for key in DISEASE_RULES:
            if key not in ("_default", "healthy") and key.lower() in disease.lower().replace(" ", "_"):
                disease_key = key
                break

    rules = DISEASE_RULES.get(disease_key, DISEASE_RULES["_default"])

    # Weather adjustments
    weather_note = None
    if weather and weather.get('daily'):
        today = weather['daily'][0] if weather['daily'] else {}
        precip_prob = today.get('precip_prob', 0) or 0
        temp_max    = today.get('temp_max', 25) or 25

        if precip_prob > 70:
            weather_note = f"⚠️ High rain probability ({precip_prob}%) — apply fungicides before rain, not after."
        elif temp_max > 35:
            weather_note = f"🌡️ High temperature ({temp_max}°C) — avoid spraying during peak heat (11am–3pm)."

    # Sort by priority
    sorted_rules = sorted(rules, key=lambda r: PRIORITY_ORDER.get(r.get("priority", "low"), 3))

    # Pesticide forecast (stub)
    pesticide_forecast = _get_pesticide_forecast(disease_key, crop)

    # Soil note
    soil_adj = SOIL_ADJUSTMENTS.get(soil_type, SOIL_ADJUSTMENTS["loamy"])
    soil_note = soil_adj.get("drainage_note")

    return {
        "recommendations":   sorted_rules,
        "pesticide_forecast": pesticide_forecast,
        "soil_note":         soil_note,
        "weather_note":      weather_note,
        "disease_key":       disease_key,
        "stub":              True,
    }


def _get_pesticide_forecast(disease_key: str, crop: str) -> Dict[str, Any]:
    """Stub pesticide/insecticide need forecast."""
    forecasts = {
        "Late_blight":    {"type": "Fungicide", "active_ingredient": "Copper sulfate / Mancozeb", "timing_window": "Immediately, repeat every 5–7 days", "quantity_per_acre": "2–3 kg"},
        "Early_blight":   {"type": "Fungicide", "active_ingredient": "Chlorothalonil / Mancozeb",  "timing_window": "Within 3 days, repeat every 7–10 days", "quantity_per_acre": "1.5–2 kg"},
        "Powdery_mildew": {"type": "Fungicide", "active_ingredient": "Sulfur / Neem oil",          "timing_window": "Every 7–14 days until symptom-free",   "quantity_per_acre": "1 kg / 2L"},
        "Common_rust":    {"type": "Fungicide", "active_ingredient": "Propiconazole (triazole)",   "timing_window": "Immediately, repeat in 14 days",        "quantity_per_acre": "0.5L"},
        "Bacterial_spot": {"type": "Bactericide","active_ingredient": "Copper hydroxide",           "timing_window": "Every 5–7 days during wet weather",     "quantity_per_acre": "2 kg"},
        "_default":       {"type": "Consult agronomist", "active_ingredient": "To be determined",  "timing_window": "After expert diagnosis",                "quantity_per_acre": "N/A"},
    }
    return forecasts.get(disease_key, forecasts["_default"])
