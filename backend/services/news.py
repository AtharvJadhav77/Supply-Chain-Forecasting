"""Live news retrieval and transformer-based sentiment scoring."""

from __future__ import annotations

import threading
import time
import urllib.parse
from typing import Any

import feedparser
import numpy as np
import requests

from ..config import NEWS_CACHE_SECONDS

NEWS_URL = (
    "https://news.google.com/rss/search?"
    "q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)

_news_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_scored_cache: dict[str, tuple[float, tuple[list[dict[str, Any]], float, int]]] = {}
_cache_lock = threading.Lock()
_sentiment_pipeline = None
_sentiment_lock = threading.Lock()


def _load_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    with _sentiment_lock:
        if _sentiment_pipeline is None:
            try:
                import torch

                torch.set_num_threads(1)
            except Exception:
                pass

            from transformers import pipeline

            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
    return _sentiment_pipeline


def fetch_news(commodity: str, limit: int = 10) -> list[dict[str, Any]]:
    now = time.time()
    with _cache_lock:
        cached = _news_cache.get(commodity)
        if cached and now - cached[0] < NEWS_CACHE_SECONDS:
            return cached[1][:limit]

    query = urllib.parse.quote(f"{commodity} India agriculture price mandi")
    url = NEWS_URL.format(query=query)

    items: list[dict[str, Any]] = []
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (AgriCast news service)"},
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        for entry in feed.entries[:limit]:
            source = ""
            if getattr(entry, "source", None) is not None:
                source = getattr(entry.source, "title", "") or ""

            items.append(
                {
                    "title": str(getattr(entry, "title", "")).strip(),
                    "url": getattr(entry, "link", None),
                    "source": source.strip(),
                    "published": getattr(entry, "published", None),
                }
            )
    except Exception:
        items = []

    with _cache_lock:
        _news_cache[commodity] = (now, items)

    return items


def score_news(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, int]:
    if not items:
        return [], 0.0, 0

    pipeline = _load_sentiment_pipeline()
    titles = [item.get("title", "")[:512] for item in items]

    try:
        with _sentiment_lock:
            results = pipeline(titles)
    except Exception:
        results = [{"label": "neutral", "score": 0.0} for _ in items]

    scored: list[dict[str, Any]] = []
    signed_scores: list[float] = []

    for item, result in zip(items, results):
        label = str(result.get("label", "neutral")).lower()
        confidence = float(result.get("score", 0.0))
        signed = confidence if label == "positive" else -confidence if label == "negative" else 0.0

        signed_scores.append(signed)
        scored.append(
            {
                **item,
                "sentiment": label if label in {"positive", "negative"} else "neutral",
                "score": round(signed, 4),
            }
        )

    aggregate = float(np.mean(signed_scores)) if signed_scores else 0.0
    return scored, aggregate, len(scored)


def get_scored_news(commodity: str, limit: int = 10) -> tuple[list[dict[str, Any]], float, int]:
    now = time.time()
    with _cache_lock:
        cached = _scored_cache.get(commodity)
        if cached and now - cached[0] < NEWS_CACHE_SECONDS:
            return cached[1]

    items = fetch_news(commodity, limit=limit)
    result = score_news(items)

    with _cache_lock:
        _scored_cache[commodity] = (now, result)

    return result


def compute_risk(sentiment: float, news_count: int) -> float:
    if news_count == 0:
        return 0.0
    return round(abs(sentiment) * news_count, 4)
