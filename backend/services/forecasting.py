"""Forecasting, inventory, and market-intelligence orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    CONFIDENCE_DECAY_PER_DAY,
    DEFAULT_LEAD_TIME_DAYS,
    MAX_HORIZON_DAYS,
    SERVICE_LEVEL_Z,
)
from ..model_registry import DataStore, ModelBundle
from .explain import generate_explanation
from .news import compute_risk, get_scored_news

FEATURE_NAMES = [
    "min_price",
    "max_price",
    "temp_max",
    "temp_min",
    "rainfall",
    "wind_speed",
    "modal_price_lag_1",
    "modal_price_lag_7",
    "modal_price_lag_14",
    "district_id",
    "commodity_id",
]


def recursive_forecast(
    subset: pd.DataFrame,
    models: ModelBundle,
    district_enc: int,
    commodity_enc: int,
    days_ahead: int,
) -> list[dict[str, Any]]:
    latest = subset.iloc[-1]
    last_date = pd.Timestamp(latest["date"])

    current_min = float(latest["min_price"])
    current_max = float(latest["max_price"])
    temp_max = float(latest["temp_max"])
    temp_min = float(latest["temp_min"])
    rainfall = float(latest["rainfall"])
    wind_speed = float(latest["wind_speed"])

    lag_1 = float(subset.iloc[-1]["modal_price"])
    lag_7 = float(subset.iloc[-7]["modal_price"])
    lag_14 = float(subset.iloc[-14]["modal_price"])

    path: list[dict[str, Any]] = []
    for step in range(1, days_ahead + 1):
        features = np.array(
            [[
                current_min,
                current_max,
                temp_max,
                temp_min,
                rainfall,
                wind_speed,
                lag_1,
                lag_7,
                lag_14,
                district_enc,
                commodity_enc,
            ]]
        )

        xgb_prediction = float(models.xgb_model.predict(features)[0])
        residual_bias = (lag_1 - lag_7) * 0.05
        fusion_prediction = xgb_prediction + residual_bias

        lag_14 = lag_7
        lag_7 = lag_1
        lag_1 = fusion_prediction

        path.append(
            {
                "date": (last_date + pd.Timedelta(days=step)).date().isoformat(),
                "price": round(fusion_prediction, 2),
            }
        )

    return path


def safety_stock(demand_std: float, lead_time_days: int, risk: float, horizon: int) -> float:
    base = SERVICE_LEVEL_Z * demand_std * np.sqrt(lead_time_days)
    horizon_multiplier = 1 + (horizon / 60)
    risk_multiplier = 1 + min(risk / 20, 0.5)
    return float(max(0.0, base * horizon_multiplier * risk_multiplier))


def classify_trend(change_percent: float) -> str:
    if change_percent > 1.5:
        return "bullish"
    if change_percent < -1.5:
        return "bearish"
    return "stable"


def build_advisory(risk: float, trend: str, change_percent: float) -> str:
    if risk > 8:
        return "High volatility detected. Increase procurement buffers and monitor news closely."
    if risk > 4:
        return "Moderate market instability. Stage purchases and reassess within a week."
    if trend == "bullish":
        return "Signals are stable but upward-leaning. Lock in partial supply early."
    if trend == "bearish":
        return "Signals are stable but softening. Avoid overstocking and negotiate on delivery."
    return "Market conditions appear stable. Maintain normal procurement cadence."


def history_payload(subset: pd.DataFrame, limit: int = 90) -> dict[str, Any]:
    window = subset.tail(limit)
    series = []
    for _, row in window.iterrows():
        series.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "modal_price": round(float(row["modal_price"]), 2),
                "min_price": round(float(row["min_price"]), 2),
                "max_price": round(float(row["max_price"]), 2),
            }
        )

    prices = subset["modal_price"].astype(float)
    stats = {
        "average": round(float(prices.mean()), 2),
        "minimum": round(float(prices.min()), 2),
        "maximum": round(float(prices.max()), 2),
        "volatility": round(float(prices.std(ddof=0)) if len(prices) > 1 else 0.0, 2),
        "latest": round(float(prices.iloc[-1]), 2),
        "observations": int(len(prices)),
    }

    return {"series": series, "stats": stats}


def build_forecast(
    store: DataStore,
    models: ModelBundle,
    district: str,
    commodity: str,
    forecast_date: pd.Timestamp,
) -> dict[str, Any]:
    subset = store.subset(district, commodity)
    if len(subset) < 15:
        raise ValueError("Not enough historical data for this district and commodity.")

    days_ahead = int((forecast_date.normalize() - store.last_date.normalize()).days)
    if days_ahead <= 0:
        raise ValueError("Forecast date must be after the latest available data.")
    if days_ahead > MAX_HORIZON_DAYS:
        raise ValueError(f"Maximum forecast horizon is {MAX_HORIZON_DAYS} days.")

    district_enc, commodity_enc = models.encode(district, commodity)

    path = recursive_forecast(subset, models, district_enc, commodity_enc, days_ahead)
    final_price = float(path[-1]["price"])
    last_actual = float(subset.iloc[-1]["modal_price"])
    change_percent = ((final_price - last_actual) / last_actual * 100) if last_actual else 0.0
    trend = classify_trend(change_percent)

    volatility = float(subset["modal_price"].std(ddof=0))
    volatility_band = SERVICE_LEVEL_Z * volatility * np.sqrt(max(days_ahead, 1) / 14)
    horizon_cap = final_price * (0.04 + 0.003 * days_ahead)
    band = max(final_price * 0.02, min(volatility_band, horizon_cap))
    low = round(max(0.0, final_price - band), 2)
    high = round(final_price + band, 2)

    confidence = float(max(30.0, 90.0 - days_ahead * CONFIDENCE_DECAY_PER_DAY))

    scored_news, sentiment, news_count = get_scored_news(commodity, limit=8)
    risk = compute_risk(sentiment, news_count)

    stock = safety_stock(volatility, DEFAULT_LEAD_TIME_DAYS, risk, days_ahead)
    advisory = build_advisory(risk, trend, change_percent)

    explanation = generate_explanation(
        district=district,
        commodity=commodity,
        forecast_date=forecast_date.date().isoformat(),
        base_price=final_price,
        low=low,
        high=high,
        change_percent=change_percent,
        trend=trend,
        sentiment=sentiment,
        risk=risk,
        confidence=confidence,
        stock=stock,
        headlines=[item.get("title", "") for item in scored_news],
        horizon=days_ahead,
    )

    history = history_payload(subset, limit=90)

    return {
        "district": district,
        "commodity": commodity,
        "forecast_date": forecast_date.date().isoformat(),
        "forecast_horizon_days": days_ahead,
        "fusion_forecast": round(final_price, 2),
        "forecast_low": low,
        "forecast_high": high,
        "expected_change_percent": round(change_percent, 2),
        "trend": trend,
        "confidence_percent": round(confidence, 1),
        "risk_index": round(float(risk), 2),
        "sentiment_score": round(float(sentiment), 4),
        "news_count": news_count,
        "recommended_safety_stock": round(stock, 2),
        "advisory": advisory,
        "explanation": explanation,
        "model": {
            "name": "XGBoost + residual drift correction",
            "xgb": "xgb_price_model.pkl",
            "base_price": f"{last_actual:,.2f}",
            "features": ", ".join(FEATURE_NAMES),
        },
        "news": scored_news,
        "history": history["series"],
        "projection": path,
    }
