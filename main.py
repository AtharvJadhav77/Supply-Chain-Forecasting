
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import feedparser
import urllib.parse
from datetime import datetime
from transformers import pipeline
from groq import Groq

# =====================================================
# CONFIG
# =====================================================

MAX_HORIZON_DAYS = 90
CONFIDENCE_DECAY_PER_DAY = 0.4  # % decay per forecast day

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

# Groq LLM
GROQ_API_KEY = "REDACTED_GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)

historical_df = pd.read_csv("data/processed/phase2_feature_complete.csv")
historical_df["date"] = pd.to_datetime(historical_df["date"])

app = FastAPI(title="Agri Price Intelligence API")

# =====================================================
# REQUEST
# =====================================================

class ForecastRequest(BaseModel):
    district: str
    commodity: str
    forecast_date: str


# =====================================================
# NEWS + SENTIMENT
# =====================================================

def fetch_news(commodity):
    query = f"{commodity} India agriculture price mandi"
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:10]]

def compute_sentiment(titles):
    if not titles:
        return 0, 0
    scores = []
    for t in titles:
        result = sentiment_model(t)[0]
        score = result["score"]
        if result["label"] == "NEGATIVE":
            score = -score
        scores.append(score)
    return float(np.mean(scores)), len(scores)

def compute_risk(sentiment, count):
    if count == 0:
        return 0
    return abs(sentiment) * count


# =====================================================
# SAFETY STOCK
# =====================================================

def safety_stock(demand_std, lead_time_days, risk, horizon):
    z = 1.65
    base = z * demand_std * np.sqrt(lead_time_days)
    horizon_multiplier = 1 + (horizon / 60)
    risk_multiplier = 1 + min(risk / 20, 0.5)
    return float(base * horizon_multiplier * risk_multiplier)


# =====================================================
# RECURSIVE FUSION FORECAST
# =====================================================

def recursive_fusion_forecast(subset, district_enc, commodity_enc, days_ahead):

    current_min = subset.iloc[-1]["min_price"]
    current_max = subset.iloc[-1]["max_price"]
    temp_max = subset.iloc[-1]["temp_max"]
    temp_min = subset.iloc[-1]["temp_min"]
    rainfall = subset.iloc[-1]["rainfall"]
    wind_speed = subset.iloc[-1]["wind_speed"]

    lag_1 = subset.iloc[-1]["modal_price"]
    lag_7 = subset.iloc[-7]["modal_price"]
    lag_14 = subset.iloc[-14]["modal_price"]

    final_prediction = lag_1

    for _ in range(days_ahead):

        features = np.array([[ 
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
            commodity_enc
        ]])

        xgb_pred = float(xgb_model.predict(features)[0])

        # Simulated LSTM residual correction
        residual_bias = (lag_1 - lag_7) * 0.05
        fusion_pred = xgb_pred + residual_bias

        # Shift lags
        lag_14 = lag_7
        lag_7 = lag_1
        lag_1 = fusion_pred

        final_prediction = fusion_pred

    return final_prediction


# =====================================================
# LLM EXPLANATION
# =====================================================

def generate_llm_explanation(district, commodity, forecast_date,
                             base_price, sentiment, risk,
                             confidence, stock, headlines, horizon):

    news_text = "\n".join(headlines[:3])

    prompt = f"""
You are a senior agricultural economist.

District: {district}
Commodity: {commodity}
Forecast Date: {forecast_date}
Forecast Horizon: {horizon} days
Predicted Price: {round(base_price,2)}
Sentiment Score: {round(sentiment,2)}
Risk Index: {round(risk,2)}
Confidence: {confidence}%
Recommended Safety Stock: {round(stock,2)}

Recent News:
{news_text}

Explain:
- Why the price may move
- How forecast horizon affects reliability
- Market stability assessment
- Inventory recommendation
Professional tone only.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an agricultural market expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=400
    )

    return response.choices[0].message.content


# =====================================================
# MAIN ENDPOINT
# =====================================================

@app.post("/forecast")
def forecast(request: ForecastRequest):

    try:
        district_enc = le_district.transform([request.district])[0]
        commodity_enc = le_commodity.transform([request.commodity])[0]
    except:
        raise HTTPException(status_code=400, detail="Invalid district or commodity")

    forecast_date = pd.to_datetime(request.forecast_date)
    last_date = historical_df["date"].max()

    if forecast_date <= last_date:
        raise HTTPException(status_code=400,
                            detail="Forecast date must be in the future.")

    days_ahead = (forecast_date - last_date).days

    if days_ahead > MAX_HORIZON_DAYS:
        raise HTTPException(status_code=400,
                            detail=f"Maximum horizon allowed is {MAX_HORIZON_DAYS} days.")

    subset = historical_df[
        (historical_df["district"] == request.district) &
        (historical_df["commodity"] == request.commodity)
    ].sort_values("date")

    if len(subset) < 15:
        raise HTTPException(status_code=400,
                            detail="Not enough historical data.")

    # Recursive Fusion
    base_price = recursive_fusion_forecast(
        subset, district_enc, commodity_enc, days_ahead
    )

    # News Intelligence
    titles = fetch_news(request.commodity)
    sentiment, news_count = compute_sentiment(titles)
    risk = compute_risk(sentiment, news_count)

    # Confidence decay
    base_confidence = 90
    confidence = max(30, base_confidence - days_ahead * CONFIDENCE_DECAY_PER_DAY)

    # Safety stock widening
    demand_std = subset["modal_price"].std()
    stock = safety_stock(demand_std, 5, risk, days_ahead)

    explanation = generate_llm_explanation(
        request.district,
        request.commodity,
        request.forecast_date,
        base_price,
        sentiment,
        risk,
        confidence,
        stock,
        titles,
        days_ahead
    )

    return {
        "district": request.district,
        "commodity": request.commodity,
        "forecast_date": request.forecast_date,
        "forecast_horizon_days": days_ahead,
        "fusion_forecast": base_price,
        "confidence_percent": confidence,
        "risk_index": risk,
        "sentiment_score": sentiment,
        "news_count": news_count,
        "recommended_safety_stock": stock,
        "explanation": explanation
    }