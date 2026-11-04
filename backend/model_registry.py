"""Lazy, process-wide access to datasets, encoders, and trained models."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from .config import DATA_PATH, MODELS_DIR

REQUIRED_COLUMNS = [
    "date",
    "district",
    "commodity",
    "modal_price",
    "min_price",
    "max_price",
    "temp_max",
    "temp_min",
    "rainfall",
    "wind_speed",
]


class DataStore:
    def __init__(self) -> None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {DATA_PATH}. "
                "Run the feature engineering notebook first."
            )

        frame = pd.read_csv(DATA_PATH)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["district", "commodity", "date"]).reset_index(drop=True)

        self.frame = frame
        self.rows = int(len(frame))
        self.first_date: pd.Timestamp = frame["date"].min()
        self.last_date: pd.Timestamp = frame["date"].max()
        self.districts: list[str] = sorted(frame["district"].unique().tolist())
        self.commodities: list[str] = sorted(frame["commodity"].unique().tolist())

        availability = (
            frame.groupby("district")["commodity"].unique().apply(lambda values: sorted(values.tolist()))
        )
        self.availability: dict[str, list[str]] = availability.to_dict()

        counts = frame.groupby(["district", "commodity"]).size().reset_index(name="observations")
        self.pair_counts = counts
        self._default_pair = self._pick_default_pair(counts)

    @staticmethod
    def _pick_default_pair(counts: pd.DataFrame) -> dict[str, str]:
        preferred = counts[counts["commodity"].str.lower() == "onion"]
        if preferred.empty:
            preferred = counts
        row = preferred.sort_values("observations", ascending=False).iloc[0]
        return {"district": str(row["district"]), "commodity": str(row["commodity"])}

    def subset(self, district: str, commodity: str) -> pd.DataFrame:
        mask = (self.frame["district"] == district) & (self.frame["commodity"] == commodity)
        return self.frame.loc[mask].sort_values("date")

    def has_pair(self, district: str, commodity: str) -> bool:
        return commodity in self.availability.get(district, [])

    def observations(self, district: str, commodity: str) -> int:
        subset = self.subset(district, commodity)
        return int(len(subset))

    @property
    def default_pair(self) -> dict[str, str]:
        return self._default_pair


class ModelBundle:
    def __init__(self) -> None:
        self.xgb_model = joblib.load(MODELS_DIR / "xgb_price_model.pkl")
        self.district_encoder = joblib.load(MODELS_DIR / "district_encoder.pkl")
        self.commodity_encoder = joblib.load(MODELS_DIR / "commodity_encoder.pkl")

    def encode(self, district: str, commodity: str) -> tuple[int, int]:
        district_encoded = int(self.district_encoder.transform([district])[0])
        commodity_encoded = int(self.commodity_encoder.transform([commodity])[0])
        return district_encoded, commodity_encoded


@lru_cache(maxsize=1)
def get_store() -> DataStore:
    return DataStore()


@lru_cache(maxsize=1)
def get_models() -> ModelBundle:
    return ModelBundle()


def health_snapshot() -> dict[str, Any]:
    store = get_store()
    models = get_models()
    return {
        "dataset_rows": store.rows,
        "last_date": store.last_date.date().isoformat(),
        "xgb_model": type(models.xgb_model).__name__,
        "encoders": "loaded",
    }
