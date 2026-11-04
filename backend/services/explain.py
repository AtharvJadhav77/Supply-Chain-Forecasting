"""LLM-powered market narrative with a deterministic offline fallback."""

from __future__ import annotations

import threading

from ..config import GROQ_API_KEY, GROQ_MODEL

_client = None
_client_lock = threading.Lock()
_client_disabled = False


def _get_client():
    global _client
    if not GROQ_API_KEY or _client_disabled:
        return None
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None and not _client_disabled:
            try:
                from groq import Groq

                _client = Groq(api_key=GROQ_API_KEY, timeout=10.0, max_retries=0)
            except Exception:
                _client = None
    return _client


def llm_status() -> str:
    return "fallback" if not GROQ_API_KEY or _client_disabled else "configured"


def generate_explanation(
    district: str,
    commodity: str,
    forecast_date: str,
    base_price: float,
    low: float,
    high: float,
    change_percent: float,
    trend: str,
    sentiment: float,
    risk: float,
    confidence: float,
    stock: float,
    headlines: list[str],
    horizon: int,
) -> str:
    global _client_disabled

    client = _get_client()
    if client is None:
        return _fallback(
            commodity,
            base_price,
            low,
            high,
            change_percent,
            trend,
            risk,
            confidence,
            stock,
            horizon,
        )

    news_text = "\n".join(headlines[:3]) or "No recent headlines available."
    prompt = f"""
You are a senior agricultural economist writing a concise market brief.

District: {district}
Commodity: {commodity}
Forecast date: {forecast_date}
Forecast horizon: {horizon} days
Expected price: {round(base_price, 2)}
Expected range: {round(low, 2)} - {round(high, 2)}
Expected change: {round(change_percent, 2)}%
Direction: {trend}
News sentiment: {round(sentiment, 2)}
Risk index: {round(risk, 2)}
Confidence: {confidence}%
Recommended safety stock: {round(stock, 2)}

Recent news:
{news_text}

Write a professional brief with these short sections:
1. Price outlook
2. Reliability and horizon
3. Market stability
4. Inventory recommendation

Use plain text, no markdown, maximum 160 words.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an agricultural market expert. Be precise and concise.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "invalid api key" in message or "authentication" in message:
            _client_disabled = True
        return _fallback(
            commodity,
            base_price,
            low,
            high,
            change_percent,
            trend,
            risk,
            confidence,
            stock,
            horizon,
        )


def _fallback(
    commodity: str,
    base_price: float,
    low: float,
    high: float,
    change_percent: float,
    trend: str,
    risk: float,
    confidence: float,
    stock: float,
    horizon: int,
) -> str:
    if risk > 8:
        stability = "elevated volatility and unstable market conditions"
    elif risk > 4:
        stability = "moderate instability that warrants close monitoring"
    else:
        stability = "relatively stable market conditions"

    direction_word = {
        "bullish": "upward",
        "bearish": "downward",
        "stable": "sideways",
    }.get(trend, "sideways")

    return (
        f"Price outlook: {commodity} is projected near {base_price:,.2f} over the next "
        f"{horizon} days, with an expected {direction_word} move of {abs(change_percent):.2f}% "
        f"and a plausible range of {low:,.2f} to {high:,.2f}.\n"
        f"Reliability: confidence is {confidence:.1f}% and decays with the forecast horizon, "
        f"so shorter horizons should be treated as more actionable than longer ones.\n"
        f"Stability: the news and price signals point to {stability}.\n"
        f"Inventory: maintain a safety stock of roughly {stock:,.0f} units and revisit "
        f"procurement if risk signals rise."
    )
