"""Small helpers for running the project in Google Colab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

KAGGLE_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _kaggle_credentials_from_secrets() -> None:
    """Use Colab secrets KAGGLE_USERNAME / KAGGLE_KEY if the user added them (key icon in the left bar)."""
    if not in_colab() or os.environ.get("KAGGLE_KEY"):
        return
    try:
        from google.colab import userdata

        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    except Exception:  # noqa: BLE001 - secrets are optional; kagglehub will ask or fail with its own message
        pass


def download_ibm(size: str = "HI-Small") -> tuple[Path, Path | None]:
    """Download `<size>_Trans.csv` and `<size>_Patterns.txt` from Kaggle with kagglehub (cached)."""
    import kagglehub

    _kaggle_credentials_from_secrets()
    try:   # newer kagglehub can fetch single files instead of the whole 7 GB dataset
        trans = Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=f"{size}_Trans.csv"))
        try:
            pat = Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=f"{size}_Patterns.txt"))
        except Exception:  # noqa: BLE001
            pat = None
        return trans, pat
    except TypeError:
        from .data import find_ibm_files

        return find_ibm_files(kagglehub.dataset_download(KAGGLE_HANDLE), size)


def gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU (CPU only)"
