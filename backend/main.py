"""AgriCast FastAPI application."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, MAX_HORIZON_DAYS
from .model_registry import get_models, get_store
from .schemas import ForecastRequest
from .services.explain import llm_status
from .services.forecasting import build_forecast, history_payload
from .services.news import get_scored_news

FORECAST_CACHE_TTL_SECONDS = 300
_forecast_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}


def _warm_up() -> None:
    try:
        store = get_store()
        models = get_models()

        from .services.news import _load_sentiment_pipeline

        _load_sentiment_pipeline()

        defaults = store.default_pair
        forecast_date = store.last_date + pd.Timedelta(days=7)
        key = (
            defaults["district"],
            defaults["commodity"],
            forecast_date.date().isoformat(),
        )
        _forecast_cache[key] = (
            time.time(),
            build_forecast(
                store=store,
                models=models,
                district=defaults["district"],
                commodity=defaults["commodity"],
                forecast_date=forecast_date,
            ),
        )
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(
    title="AgriCast API",
    version="2.0.0",
    description=(
        "Hybrid agri price forecasting, live news intelligence, and "
        "prescriptive supply-chain recommendations."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    store = get_store()
    models = get_models()
    return {
        "status": "ok",
        "service": "AgriCast API",
        "version": "2.0.0",
        "models": {
            "xgb": type(models.xgb_model).__name__,
            "encoders": "loaded",
        },
        "data": {
            "rows": store.rows,
            "first_date": store.first_date.date().isoformat(),
            "last_date": store.last_date.date().isoformat(),
            "districts": len(store.districts),
            "commodities": len(store.commodities),
            "pairs": len(store.pair_counts),
        },
        "llm": llm_status(),
        "news": "enabled",
    }


@app.get("/api/meta")
def meta() -> dict:
    store = get_store()
    default_date = (store.last_date + pd.Timedelta(days=7)).date().isoformat()

    return {
        "districts": store.districts,
        "commodities": store.commodities,
        "availability": store.availability,
        "defaults": {
            **store.default_pair,
            "forecast_date": default_date,
        },
        "dataset": {
            "rows": store.rows,
            "districts": len(store.districts),
            "commodities": len(store.commodities),
            "pairs": int(len(store.pair_counts)),
            "first_date": store.first_date.date().isoformat(),
            "last_date": store.last_date.date().isoformat(),
        },
        "limits": {
            "max_horizon_days": MAX_HORIZON_DAYS,
        },
    }


@app.get("/api/history")
def history(
    district: str = Query(...),
    commodity: str = Query(...),
    limit: int = Query(120, ge=30, le=365),
) -> dict:
    store = get_store()
    if not store.has_pair(district, commodity):
        raise HTTPException(status_code=404, detail="No data for this district and commodity.")

    subset = store.subset(district, commodity)
    payload = history_payload(subset, limit=limit)
    return {
        "district": district,
        "commodity": commodity,
        "series": payload["series"],
        "stats": payload["stats"],
    }


@app.get("/api/news")
def news(commodity: str = Query(...)) -> dict:
    store = get_store()
    if commodity not in store.commodities:
        raise HTTPException(status_code=404, detail="Unknown commodity.")

    items, sentiment, count = get_scored_news(commodity, limit=10)
    return {
        "commodity": commodity,
        "sentiment_score": round(sentiment, 4),
        "news_count": count,
        "items": items,
    }


@app.post("/api/forecast")
@app.post("/forecast", include_in_schema=False)
def forecast(request: ForecastRequest) -> dict:
    store = get_store()
    models = get_models()

    if request.district not in store.districts:
        raise HTTPException(status_code=400, detail="Unknown district.")
    if request.commodity not in store.commodities:
        raise HTTPException(status_code=400, detail="Unknown commodity.")
    if not store.has_pair(request.district, request.commodity):
        raise HTTPException(
            status_code=400,
            detail="No historical data for this district and commodity pair.",
        )

    forecast_date = pd.to_datetime(request.forecast_date, errors="coerce")
    if pd.isna(forecast_date):
        raise HTTPException(status_code=400, detail="Invalid forecast date format.")

    cache_key = (request.district, request.commodity, forecast_date.date().isoformat())
    cached = _forecast_cache.get(cache_key)
    if cached and time.time() - cached[0] < FORECAST_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        result = build_forecast(
            store=store,
            models=models,
            district=request.district,
            commodity=request.commodity,
            forecast_date=forecast_date,
        )
        _forecast_cache[cache_key] = (time.time(), result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc
