"""Application configuration and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_PATH = ROOT_DIR / "data" / "processed" / "phase2_feature_complete.csv"
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_HORIZON_DAYS = 90
CONFIDENCE_DECAY_PER_DAY = 0.4
SERVICE_LEVEL_Z = 1.65
DEFAULT_LEAD_TIME_DAYS = 5
NEWS_CACHE_SECONDS = 600


def load_env(path: Path | str = ROOT_DIR / ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
