from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import feedparser
import urllib.parse
from transformers import pipeline

# =====================================================
# LOAD MODELS
# =====================================================

xgb_model = joblib.load("models/xgb_price_model.pkl")
le_district = joblib.load("models/district_encoder.pkl")
le_commodity = joblib.load("models/commodity_encoder.pkl")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

historical_df = pd.read_csv("data/processed/phase2_feature_complete.csv")
historical_df["date"] = pd.to_datetime(historical_df["date"])

app = FastAPI(title="Agri Price Intelligence API")


# =====================================================
# REQUEST SCHEMA
# =====================================================

class ForecastRequest(BaseModel):
    district: str
    commodity: str


# =====================================================
# NEWS INTELLIGENCE
# =====================================================

def fetch_news(commodity):
    query = f"{commodity} India agriculture price mandi"
    encoded = urllib.parse.quote(query)

    url = (
        f"https://news.google.com/rss/search?"
        f"q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)
    titles = [entry.title for entry in feed.entries[:10]]
    return titles


def compute_sentiment(titles):
    if len(titles) == 0:
        return 0, 0

    scores = []
    for text in titles:
        result = sentiment_model(text)[0]
        score = result["score"]
        if result["label"] == "NEGATIVE":
            score = -score
        scores.append(score)

    return float(np.mean(scores)), len(scores)


def compute_risk(sentiment, news_count):
    if news_count == 0:
        return 0
    return abs(sentiment) * news_count


# =====================================================
# CONFIDENCE LAYER
# =====================================================

def compute_confidence(risk_index):
    """
    Higher risk → lower confidence
    Confidence bounded between 50% and 95%
    """
    confidence = 0.95 - min(risk_index / 20, 0.45)
    return round(confidence * 100, 2)


# =====================================================
# SAFETY STOCK
# =====================================================

def safety_stock(demand_std, lead_time_days, risk_index):

    z = 1.65  # 95% service level
    base = z * demand_std * np.sqrt(lead_time_days)

    # Increase buffer moderately
    multiplier = 1 + min(risk_index / 25, 0.4)

    return float(base * multiplier)


# =====================================================
# ADVISORY LOGIC
# =====================================================

def generate_advisory(sentiment, risk):

    if risk > 8:
        return "High volatility risk detected. Increase procurement buffer."
    elif risk > 4:
        return "Moderate market instability. Monitor supply closely."
    else:
        return "Market conditions appear stable."


# =====================================================
# MAIN ENDPOINT
# =====================================================

@app.post("/forecast")
def forecast(request: ForecastRequest):

    # Validate encoders
    try:
        district_enc = le_district.transform([request.district])[0]
        commodity_enc = le_commodity.transform([request.commodity])[0]
    except:
        raise HTTPException(status_code=400, detail="Invalid district or commodity")

    # Filter historical data
    subset = historical_df[
        (historical_df["district"] == request.district) &
        (historical_df["commodity"] == request.commodity)
    ].sort_values("date")

    if len(subset) < 15:
        raise HTTPException(status_code=400, detail="Not enough historical data")

    latest = subset.iloc[-1]

    # Auto-generate lag features
    modal_lag_1 = latest["modal_price"]
    modal_lag_7 = subset.iloc[-7]["modal_price"]
    modal_lag_14 = subset.iloc[-14]["modal_price"]

    features = np.array([[ 
        latest["min_price"],
        latest["max_price"],
        latest["temp_max"],
        latest["temp_min"],
        latest["rainfall"],
        latest["wind_speed"],
        modal_lag_1,
        modal_lag_7,
        modal_lag_14,
        district_enc,
        commodity_enc
    ]])

    # Base Forecast (T+1 simulation)
    base_price = float(xgb_model.predict(features)[0])

    # NEWS INTELLIGENCE
    titles = fetch_news(request.commodity)
    sentiment, news_count = compute_sentiment(titles)
    risk = compute_risk(sentiment, news_count)

    # Confidence (instead of heavy price shifting)
    confidence = compute_confidence(risk)

    # Safety stock using historical volatility
    demand_std = subset["modal_price"].std()
    stock = safety_stock(demand_std, 5, risk)

    advisory = generate_advisory(sentiment, risk)

    return {
        "district": request.district,
        "commodity": request.commodity,
        "predicted_for": "Next Trading Day",
        "base_forecast": base_price,
        "risk_index": risk,
        "confidence_percent": confidence,
        "sentiment_score": sentiment,
        "news_count": news_count,
        "recommended_safety_stock": stock,
        "advisory": advisory
    }
