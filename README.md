# AgriCast — AI Agri Price Intelligence

**A hybrid forecasting and decision-support system for volatile agricultural supply chains.**

AgriCast combines XGBoost price forecasting, a residual drift correction layer, live news sentiment, and prescriptive inventory logic behind a polished web dashboard and a documented FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-EC6C00)
![Transformers](https://img.shields.io/badge/Transformers-5.0-FFD21E)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why this project exists

Agri commodity prices are volatile because they are driven by weather, arrivals, perishability, and news shocks. A single model rarely captures all of that. AgriCast treats forecasting as a **multi-layer decision problem**:

1. **Price model** — XGBoost predicts the base modal price from weather, lag, and market features.
2. **Residual correction** — a drift heuristic keeps recursive forecasts from wandering.
3. **News intelligence** — live headlines are scored with a transformer sentiment model and converted into a risk index.
4. **Prescriptive layer** — confidence decay, uncertainty bands, and safety-stock recommendations turn a prediction into an actionable decision.

The result is not a "future price oracle" — it is a decision-support system that separates **what the price may do** from **how much to trust it** and **how much buffer to hold**.

---

## Dashboard preview

The frontend is a zero-build, dependency-light SPA served directly by FastAPI:

- Dark, responsive "forecast studio" UI with glass panels and a live model-stack visual
- District + commodity selector with pair-aware availability
- 7 / 15 / 30 / 60-day quick horizons and a custom forecast date
- KPI cards for price, confidence, risk, and safety stock
- Historical + projected price chart (Chart.js, vendored for offline use)
- AI market brief and scored live news feed
- Model architecture section and one-click OpenAPI docs

Open **http://127.0.0.1:8000** after starting the server.

---

## Architecture

```text
                    ┌─────────────────────────────┐
                    │  APMC price history         │
                    │  Weather · lags · calendar  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
      ┌────────▼────────┐                       ┌────────▼────────┐
      │ XGBoost price   │                       │ Residual drift  │
      │ engine          │                       │ correction      │
      └────────┬────────┘                       └────────┬────────┘
              └────────────────────┬────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Fusion forecast │
                          └────────┬────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐       ┌─────────▼─────────┐      ┌─────────▼─────────┐
│ News retrieval │       │ Sentiment + risk  │      │ Uncertainty band  │
│ Google News    │       │ DistilBERT        │      │ Confidence decay  │
└───────┬────────┘       └─────────┬─────────┘      └─────────┬─────────┘
        └──────────────────────────┼──────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │ Decision output │
                          │ price · range   │
                          │ risk · stock    │
                          │ AI brief        │
                          └─────────────────┘
```

---

## Project structure

```text
COMPLETE-SUPPY-CHAIN/
├── backend/
│   ├── main.py                 # FastAPI app, routes, static frontend mount
│   ├── main1.py                # Original single-step API (kept for reference)
│   ├── config.py               # Paths, limits, .env loading
│   ├── model_registry.py       # Lazy dataset + model + encoder loading
│   ├── schemas.py              # Pydantic request/response models
│   └── services/
│       ├── forecasting.py      # Recursive fusion, confidence, safety stock
│       ├── news.py             # Google News RSS + sentiment scoring
│       └── explain.py          # Groq LLM brief with offline fallback
├── frontend/
│   ├── index.html              # Dashboard shell
│   ├── styles.css              # Dark responsive design system
│   ├── app.js                  # API client, chart, interactions
│   └── vendor/chart.umd.min.js # Vendored Chart.js
├── notebooks/                  # EDA, feature engineering, LSTM, XGBoost, SHAP
├── models/                     # Trained model artifacts (unchanged)
├── data/processed/             # Feature-complete dataset (unchanged)
├── run.py                      # One-command launcher
├── requirements.txt
└── README.md
```

Notebooks, model artifacts, and processed data are intentionally left untouched.

---

## Quickstart

### 1. Activate the environment

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure the LLM (optional)

The app works without an LLM key; it returns a deterministic market brief.

```bash
cp .env.example .env
# then set GROQ_API_KEY in .env
```

### 3. Run

```powershell
python run.py
```

or with reload during development:

```powershell
uvicorn backend.main:app --reload
```

Open:

- **Dashboard:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service, model, and dataset status |
| `GET` | `/api/meta` | Districts, commodities, availability, defaults, dataset stats |
| `GET` | `/api/history` | Historical price series + volatility stats for a pair |
| `GET` | `/api/news` | Live headlines with sentiment scores |
| `POST` | `/api/forecast` | Full fusion forecast, risk, safety stock, and AI brief |
| `POST` | `/forecast` | Backward-compatible alias for the original endpoint |

### Forecast request

```bash
curl -X POST http://127.0.0.1:8000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"district":"Pune","commodity":"Onion","forecast_date":"2025-01-15"}'
```

### Forecast response (abridged)

```json
{
  "district": "Pune",
  "commodity": "Onion",
  "forecast_horizon_days": 15,
  "fusion_forecast": 2353.05,
  "forecast_low": 2153.04,
  "forecast_high": 2553.06,
  "expected_change_percent": -10.83,
  "trend": "bearish",
  "confidence_percent": 84.0,
  "risk_index": 0.58,
  "sentiment_score": -0.0722,
  "news_count": 8,
  "recommended_safety_stock": 4914.72,
  "advisory": "Signals are stable but softening. Avoid overstocking and negotiate on delivery.",
  "explanation": "Price outlook: Onion is projected near 2,353.05 over the next 15 days...",
  "news": [
    {
      "title": "Onion prices ease as arrivals rise",
      "source": "AgroSpectrum India",
      "sentiment": "neutral",
      "score": -0.07
    }
  ],
  "history": [{"date": "2024-12-31", "modal_price": 2638.89}],
  "projection": [{"date": "2025-01-01", "price": 2412.94}]
}
```

---

## Modeling details

| Component | Implementation |
| --- | --- |
| Base price model | `models/xgb_price_model.pkl` |
| Encoders | District and commodity `LabelEncoder`s |
| Temporal correction | `(lag_1 − lag_7) × 0.05` residual bias per recursive step |
| Horizon | Up to 90 days, recursive multi-step |
| Confidence | `max(30, 90 − 0.4 × days_ahead)` |
| Uncertainty band | `1.65 × σ × sqrt(days / 14)` with a 2% floor |
| Safety stock | `Z × σ_d × sqrt(L) × (1 + horizon/60) × risk multiplier` |
| Risk index | `abs(mean sentiment) × headline count` |
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| Narrative | Groq `llama-3.1-8b-instant` with deterministic fallback |

**Dataset:** 164,171 rows · 2024-01-15 → 2024-12-31 · 34 districts · 1,027 district-commodity pairs.

---

## Design notes

- **Lazy loading:** models and the 16 MB dataset load once on first API use, keeping startup fast.
- **Graceful degradation:** news failures return an empty feed; LLM failures return a structured fallback brief; the dashboard remains usable offline.
- **No build step:** the frontend is plain HTML/CSS/JS with vendored Chart.js, so a single `python run.py` is enough.
- **Backward compatibility:** the original `/forecast` route and the original `backend/main1.py` implementation are preserved.

---

## Roadmap

- [ ] Wire the actual LSTM `.keras` residual model into inference instead of the residual heuristic
- [ ] Add SHAP explainability endpoint and waterfall chart in the dashboard
- [ ] Add model versioning and scheduled retraining pipeline
- [ ] Add prediction-interval calibration against held-out 2024 data
- [ ] Add authentication and rate limiting for public deployment
- [ ] Add Dockerfile and CI checks

---

## Disclaimer

This project is a decision-support prototype. Forecasts are not financial or procurement advice. Validate model outputs against current mandi data before making commercial decisions.
