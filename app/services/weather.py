"""
CropGuard AI — Weather Service (Open-Meteo API)

Fetches weather forecasts for a given lat/lng using the Open-Meteo API.
Open-Meteo is free and requires no API key.

Phase 2 dependency: Used by the disease progression predictor and
the preventive measures engine.
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Variables we request from the API
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
]


def get_forecast(
    lat: float,
    lng: float,
    days: int = 7,
    timeout: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a weather forecast for the given coordinates.

    Args:
        lat:     Latitude
        lng:     Longitude
        days:    Forecast horizon in days (1–16)
        timeout: Request timeout in seconds

    Returns:
        Parsed weather dict, or None if the request fails.
        Structure:
        {
            'current': { temperature, humidity, precipitation, wind_speed },
            'daily':   [ { date, temp_max, temp_min, precip, precip_prob, wind_max }, ... ],
            'hourly':  [ { time, temp, humidity, precip, wind }, ... ],
            'fetched_at': ISO timestamp
        }
    """
    params = {
        "latitude":        lat,
        "longitude":       lng,
        "forecast_days":   min(days, 16),
        "hourly":          ",".join(HOURLY_VARIABLES),
        "daily":           ",".join(DAILY_VARIABLES),
        "timezone":        "auto",
        "wind_speed_unit": "kmh",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
        return _parse_forecast(raw)
    except requests.exceptions.Timeout:
        logger.warning(f"Weather API timed out for ({lat}, {lng})")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error(f"Weather API error: {exc}")
        return None


def _parse_forecast(raw: Dict) -> Dict[str, Any]:
    """Transform raw Open-Meteo response into a cleaner structure."""
    daily_times = raw.get('daily', {}).get('time', [])
    daily_data  = raw.get('daily', {})
    hourly_times = raw.get('hourly', {}).get('time', [])
    hourly_data  = raw.get('hourly', {})

    # Build daily list
    daily_list = []
    for i, date in enumerate(daily_times):
        daily_list.append({
            'date':       date,
            'temp_max':   _get(daily_data, 'temperature_2m_max', i),
            'temp_min':   _get(daily_data, 'temperature_2m_min', i),
            'precip_mm':  _get(daily_data, 'precipitation_sum', i),
            'precip_prob': _get(daily_data, 'precipitation_probability_max', i),
            'wind_max':   _get(daily_data, 'wind_speed_10m_max', i),
        })

    # Build hourly list (first 24 hours only for brevity)
    hourly_list = []
    for i, ts in enumerate(hourly_times[:24]):
        hourly_list.append({
            'time':     ts,
            'temp':     _get(hourly_data, 'temperature_2m', i),
            'humidity': _get(hourly_data, 'relative_humidity_2m', i),
            'precip':   _get(hourly_data, 'precipitation', i),
            'wind':     _get(hourly_data, 'wind_speed_10m', i),
        })

    # Current conditions from first hourly entry
    current = hourly_list[0] if hourly_list else {}

    return {
        'current':    current,
        'daily':      daily_list,
        'hourly':     hourly_list,
        'fetched_at': datetime.utcnow().isoformat(),
        'lat':        raw.get('latitude'),
        'lng':        raw.get('longitude'),
        'timezone':   raw.get('timezone'),
    }


def _get(data: Dict, key: str, idx: int):
    """Safely get item at index from a data dict."""
    values = data.get(key, [])
    return values[idx] if idx < len(values) else None


def get_cached_forecast(lat: float, lng: float, max_age_minutes: int = 60) -> Optional[Dict]:
    """
    Return a cached weather forecast from the DB if recent enough.
    Falls back to a live API fetch if cache is stale or absent.
    """
    try:
        from flask import current_app
        from ..models.database import WeatherCache
        from .. import db

        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        # Round to 2 decimal places for cache key matching
        rlat, rlng = round(lat, 2), round(lng, 2)

        cached = (WeatherCache.query
                  .filter(WeatherCache.lat == rlat,
                          WeatherCache.lng == rlng,
                          WeatherCache.fetched_at >= cutoff)
                  .order_by(WeatherCache.fetched_at.desc())
                  .first())

        if cached:
            logger.debug(f"Weather cache hit for ({rlat}, {rlng})")
            return json.loads(cached.data_json)

        # Cache miss — fetch live
        data = get_forecast(lat, lng)
        if data:
            entry = WeatherCache(lat=rlat, lng=rlng, data_json=json.dumps(data))
            db.session.add(entry)
            db.session.commit()
        return data

    except Exception as exc:
        logger.warning(f"Weather cache error: {exc}. Fetching directly.")
        return get_forecast(lat, lng)
