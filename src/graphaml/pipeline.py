"""End to end: standardized transactions -> features -> four models -> evaluation.

The notebooks call these functions step by step; `scripts/train.py` runs them in one go;
the API reuses `predict_models`, so training and serving compute features with the same code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features, metrics
from .data import time_split
from .features import FeatureSet, Snapshots
from .models import GNNScorer, LGBMScorer, RuleScorer, rank_average

MODEL_ORDER = ["rules", "lgbm_tabular", "lgbm_graph", "gnn", "ensemble"]   # simplest first
MODEL_LABELS = {
    "rules": "Red-flag rules (what banks run today)",
    "lgbm_tabular": "LightGBM, transaction + account history",
    "lgbm_graph": "LightGBM + graph features",
    "gnn": "Graph neural network",
    "ensemble": "GNN + LightGBM-graph (rank average)",
}


@dataclass
class Experiment:
    df: pd.DataFrame            # prepared (src, dst, t, day, cluster, ...)
    split: np.ndarray           # train / valid / test per row
    feats: FeatureSet
    snaps: Snapshots

    @property
    def y(self) -> np.ndarray:
        return self.df["is_laundering"].to_numpy()

    def rows(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.split == name)

    def days(self, name: str) -> list[int]:
        return sorted(np.unique(self.df["day"].to_numpy()[self.split == name]).tolist())

    def default_k(self, alert_rate: float = 0.005) -> int:
        """Daily alert budget: `alert_rate` of a typical training day's volume (at least 5)."""
        per_day = pd.Series(self.df["day"].to_numpy()[self.split == "train"]).value_counts()
        return max(5, int(round(alert_rate * per_day.median())))


def build_experiment(std: pd.DataFrame, shares=(0.6, 0.2), verbose: bool = True) -> Experiment:
    t0 = time.time()
    df = features.prepare(std)
    split = time_split(df, shares).to_numpy()
    snaps = Snapshots(df)
    feats = features.build(df, snaps)
    if verbose:
        print(f"{len(df):,} transactions, {df.attrs['n_accounts']:,} accounts, "
              f"{feats.full.shape[1]} features in {time.time() - t0:.1f}s")
    return Experiment(df, split, feats, snaps)


def train_models(exp: Experiment, gnn_kwargs: dict | None = None, verbose: bool = True,
                 which=("rules", "lgbm_tabular", "lgbm_graph", "gnn")) -> dict:
    tr, va = exp.rows("train"), exp.rows("valid")
    X, y = exp.feats.full, exp.y
    models = {}
    if "rules" in which:
        models["rules"] = RuleScorer().fit(X.iloc[tr])
    for name, cols in (("lgbm_tabular", exp.feats.tabular_cols), ("lgbm_graph", list(X.columns))):
        if name in which:
            t0 = time.time()
            models[name] = LGBMScorer(name, cols).fit(X.iloc[tr], y[tr], X.iloc[va], y[va])
            if verbose:
                print(f"{name}: {models[name].model.best_iteration_} trees, {time.time() - t0:.0f}s")
    if "gnn" in which:
        models["gnn"] = GNNScorer(exp.feats.tabular_cols, **(gnn_kwargs or {})).fit(
            exp.snaps, X, y, exp.days("train"), exp.days("valid"), verbose=verbose)
    return models


def predict_models(models: dict, feats: FeatureSet, snaps: Snapshots, days,
                   reuse: dict | None = None) -> dict[str, np.ndarray]:
    """Scores of every model for the rows of `days` (NaN elsewhere).

    `reuse`: scores already computed for the same days (e.g. the baselines before the GNN was
    trained); those models are not scored again.
    """
    day_of = snaps.day_of
    rows = np.flatnonzero(np.isin(day_of, list(days)))
    X = feats.full.iloc[rows]
    out = {}
    for name, m in models.items():
        s = np.full(len(day_of), np.nan)
        if reuse is not None and name in reuse:
            s = reuse[name]
        elif name == "gnn":
            s = m.predict(snaps, feats.full, days)
        else:
            s[rows] = m.predict(X)
        out[name] = s
    if "gnn" in out and "lgbm_graph" in out:
        s = np.full(len(day_of), np.nan)
        s[rows] = rank_average(out["gnn"][rows], out["lgbm_graph"][rows])
        out["ensemble"] = s
    return {k: out[k] for k in MODEL_ORDER if k in out}


def evaluate_split(exp: Experiment, scores: dict, split: str, k: int, thresholds: dict | None = None,
                   n_boot: int = 200, reference: str = "rules", seed: int = 0) -> metrics.Report:
    rows = exp.rows(split)
    d = exp.df.iloc[rows]
    return metrics.evaluate({m: s[rows] for m, s in scores.items()}, d["is_laundering"], d["day"], d["cluster"],
                            d["amount_usd"], thresholds=thresholds, k=k, n_boot=n_boot,
                            reference=reference if reference in scores else None, seed=seed)


def f1_thresholds(exp: Experiment, scores: dict) -> dict[str, float]:
    """Per-model threshold that maximises F1 on the validation split (then frozen for test)."""
    va = exp.rows("valid")
    return {m: metrics.best_f1_threshold(exp.y[va], s[va]) for m, s in scores.items()}


def make_metadata(exp: Experiment, models: dict, source: str, served: str, why: str, k: int, thresholds: dict,
                  val: metrics.Report, test: metrics.Report) -> dict:
    """Everything the API and the model card need besides the weights."""
    from .monitoring import reference_profile

    tr = exp.rows("train")
    sample = exp.feats.tabular.iloc[tr].sample(min(len(tr), 200_000), random_state=0)
    t0 = exp.df.attrs["t0"]
    split_days = {s: [str((t0 + pd.Timedelta(days=x)).date()) for x in (exp.days(s)[0], exp.days(s)[-1])]
                  for s in ("train", "valid", "test")}
    return {
        "trained_on": source,
        "rows": len(exp.df), "accounts": int(exp.df.attrs["n_accounts"]),
        "laundering_rate": float(exp.y.mean()),
        "split_days": split_days,
        "served_model": served, "served_reason": why,
        "alerts_per_day": int(k),
        "f1_thresholds": thresholds,
        "valid_metrics": val.point.round(4).to_dict(orient="index"),
        "test_metrics": test.point.round(4).to_dict(orient="index"),
        "test_ci_low": test.ci_low.round(4).to_dict(orient="index"),
        "test_ci_high": test.ci_high.round(4).to_dict(orient="index"),
        "gnn_history": models["gnn"].history if "gnn" in models else [],
        "reference_profile": reference_profile(sample, list(sample.columns), []),
    }
