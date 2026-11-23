"""Saving and loading the trained models with everything needed to use them safely."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from .data import ROOT

MODEL_DIR = ROOT / "models"
BUNDLE_PATH = MODEL_DIR / "aml_bundle.pkl"
META_PATH = MODEL_DIR / "model_meta.json"


def save(models: dict, metadata: dict, bundle_path: Path = BUNDLE_PATH, meta_path: Path = META_PATH) -> None:
    import lightgbm
    import sklearn
    import torch

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bundle_path, "wb") as f:
        pickle.dump(models, f)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__, "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__},
        "models": list(models),
        **metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def load(path: Path = BUNDLE_PATH) -> dict:
    """Only load bundles you produced yourself: pickle executes code on load."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_meta(path: Path = META_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
