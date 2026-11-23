"""Daily batch scoring with a trained bundle — the code path the API uses.

Input: a batch of transactions in IBM format that contains the history to build the graph from
plus the day(s) to score. Output: a risk score per scored transaction, the daily alert list and
the reasons behind each alert. Features are computed by the same functions as in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifact, explain, features, metrics
from .features import FeatureSet, Snapshots
from .pipeline import predict_models


@dataclass
class Scored:
    df: pd.DataFrame            # prepared rows (history + scored)
    feats: FeatureSet
    snaps: Snapshots
    scores: dict[str, np.ndarray]
    rows: np.ndarray            # indices of the scored rows
    risk: np.ndarray            # served model's score (NaN on history rows)
    alert: np.ndarray           # bool, top-k per day among scored rows


class Scorer:
    def __init__(self, bundle_path: Path = artifact.BUNDLE_PATH, meta_path: Path = artifact.META_PATH):
        self.models = artifact.load(bundle_path)
        self.meta = artifact.load_meta(meta_path)
        self.served = self.meta["served_model"]
        self.k = int(self.meta["alerts_per_day"])

    def score(self, std: pd.DataFrame, score_from: pd.Timestamp | None = None, k: int | None = None) -> Scored:
        df = features.prepare(std)
        snaps = Snapshots(df)
        feats = features.build(df, snaps)
        if score_from is None:
            score_from = df["ts"].max().floor("D")          # default: the last day in the file
        first_day = int((pd.Timestamp(score_from).floor("D") - df.attrs["t0"]) // pd.Timedelta(days=1))
        days = [d for d in snaps.days if d >= first_day]
        scores = predict_models(self.models, feats, snaps, days)
        rows = np.flatnonzero(np.isin(df["day"].to_numpy(), days))
        risk = scores[self.served]
        alert = np.zeros(len(df), bool)
        alert[rows] = metrics.alerts(risk[rows], df["day"].to_numpy()[rows], k or self.k)
        return Scored(df, feats, snaps, scores, rows, risk, alert)

    def reasons(self, s: Scored, rows: np.ndarray, n: int = 4) -> list[list[dict]]:
        """Top contributing features per row (from the LightGBM-graph model, which has exact TreeSHAP)."""
        lg = self.models.get("lgbm_graph")
        if lg is None or len(rows) == 0:
            return [[] for _ in rows]
        X = s.feats.full.iloc[rows]
        C = lg.contributions(X)
        return [explain.top_reasons(C.iloc[i], X.iloc[i], n) for i in range(len(rows))]

    def rule_flags(self, s: Scored, rows: np.ndarray) -> list[list[str]]:
        r = self.models.get("rules")
        if r is None or len(rows) == 0:
            return [[] for _ in rows]
        F = r.flags(s.feats.full.iloc[rows])
        return [[c for c in F.columns if F.iloc[i][c]] for i in range(len(rows))]
