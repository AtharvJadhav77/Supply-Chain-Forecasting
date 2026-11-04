"""Request and response schemas for the AgriCast API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    district: str = Field(..., examples=["Pune"])
    commodity: str = Field(..., examples=["Onion"])
    forecast_date: str = Field(..., examples=["2025-01-15"])


class NewsItem(BaseModel):
    title: str
    url: str | None = None
    source: str | None = None
    published: str | None = None
    sentiment: str = "neutral"
    score: float = 0.0


class PricePoint(BaseModel):
    date: str
    modal_price: float
    min_price: float | None = None
    max_price: float | None = None


class ProjectionPoint(BaseModel):
    date: str
    price: float


class HistoryStats(BaseModel):
    average: float
    minimum: float
    maximum: float
    volatility: float
    latest: float
    observations: int


class HistoryResponse(BaseModel):
    district: str
    commodity: str
    series: list[PricePoint]
    stats: HistoryStats


class ForecastResponse(BaseModel):
    district: str
    commodity: str
    forecast_date: str
    forecast_horizon_days: int
    fusion_forecast: float
    forecast_low: float
    forecast_high: float
    expected_change_percent: float
    trend: str
    confidence_percent: float
    risk_index: float
    sentiment_score: float
    news_count: int
    recommended_safety_stock: float
    advisory: str
    explanation: str
    model: dict[str, str]
    news: list[NewsItem] = Field(default_factory=list)
    history: list[PricePoint] = Field(default_factory=list)
    projection: list[ProjectionPoint] = Field(default_factory=list)
