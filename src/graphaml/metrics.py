"""Evaluation in the units an AML team works in.

Laundering is ~0.1% of transactions, so accuracy and ROC-AUC say little. We report:
    ap                  average precision (area under precision-recall)
    f1                  minority-class F1 at a threshold chosen on validation (the IBM paper's metric)
    precision@k, recall@k   with k = alerts per day the analysts can review
    schemes@k           share of laundering *attempts* with at least one alerted transaction:
                        one alert is enough for an investigator to unravel the whole scheme
    amount@k            share of laundered money in alerted transactions

Intervals: cluster bootstrap. Transactions of one scheme are correlated, so whole schemes are
resampled together; all models share the same resamples, which gives paired differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rank_in_day(score, day) -> np.ndarray:
    """1 = highest score of its day."""
    return pd.Series(np.asarray(score, float)).groupby(np.asarray(day)).rank(ascending=False, method="first").to_numpy()


def alerts(score, day, k: int) -> np.ndarray:
    return rank_in_day(score, day) <= k


def best_f1_threshold(y, score) -> float:
    from sklearn.metrics import precision_recall_curve

    p, r, thr = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def _weighted_ap(order, y, w) -> float:
    ys, ws = y[order], w[order]
    cw, cp = np.cumsum(ws), np.cumsum(ws * ys)
    tot = cp[-1]
    if tot <= 0:
        return float("nan")
    return float(np.sum(ws * ys * cp / np.maximum(cw, 1e-12)) / tot)


@dataclass
class _Prepared:
    order: np.ndarray
    alert: np.ndarray
    pred: np.ndarray


def _metrics(p: _Prepared, y, amount, cluster_of_row, w_row, w_cluster, k) -> dict:
    yw = y * w_row
    tp_k = np.sum(yw * p.alert)
    out = {
        "ap": _weighted_ap(p.order, y, w_row),
        f"precision@{k}": tp_k / max(np.sum(w_row * p.alert), 1e-12),
        f"recall@{k}": tp_k / max(np.sum(yw), 1e-12),
        f"amount@{k}": np.sum(yw * amount * p.alert) / max(np.sum(yw * amount), 1e-12),
    }
    tp, fp, fn = np.sum(yw * p.pred), np.sum((1 - y) * w_row * p.pred), np.sum(yw * (1 - p.pred))
    out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    # schemes: positive clusters, caught if any of their rows is alerted
    pos = y == 1
    caught = pd.Series(p.alert[pos]).groupby(cluster_of_row[pos]).max()
    wc = w_cluster[caught.index.to_numpy()]
    out[f"schemes@{k}"] = float(np.sum(wc * caught.to_numpy()) / max(np.sum(wc), 1e-12))
    return out


@dataclass
class Report:
    point: pd.DataFrame
    ci_low: pd.DataFrame
    ci_high: pd.DataFrame
    vs_reference: pd.DataFrame
    boots: dict | None = None   # model -> DataFrame of bootstrap replicates (same resamples for all)

    def table(self, digits: int = 3) -> pd.DataFrame:
        """'point [low, high]' strings, convenient for README tables."""
        f = lambda a, b, c: f"{a:.{digits}f} [{b:.{digits}f}, {c:.{digits}f}]"  # noqa: E731
        return pd.DataFrame({c: [f(self.point.loc[m, c], self.ci_low.loc[m, c], self.ci_high.loc[m, c])
                                 for m in self.point.index] for c in self.point.columns}, index=self.point.index)


def evaluate(scores: dict[str, np.ndarray], y, day, cluster, amount, thresholds: dict[str, float] | None = None,
             k: int = 100, n_boot: int = 200, reference: str | None = None, seed: int = 0) -> Report:
    y = np.asarray(y, float)
    amount = np.asarray(amount, float)
    codes, uniq = pd.factorize(np.asarray(cluster))
    n_clusters = len(uniq)
    thresholds = thresholds or {}
    prep = {m: _Prepared(order=np.argsort(-np.asarray(s, float), kind="stable"),
                         alert=alerts(s, day, k).astype(float),
                         pred=(np.asarray(s, float) >= thresholds.get(m, np.inf)).astype(float))
            for m, s in scores.items()}

    ones_r, ones_c = np.ones(len(y)), np.ones(n_clusters)
    point = pd.DataFrame({m: _metrics(p, y, amount, codes, ones_r, ones_c, k) for m, p in prep.items()}).T
    if n_boot == 0:
        return Report(point, point * np.nan, point * np.nan, pd.DataFrame())

    rng = np.random.default_rng(seed)
    boots = {m: [] for m in prep}
    for _ in range(n_boot):
        wc = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(float)
        wr = wc[codes]
        for m, p in prep.items():
            boots[m].append(_metrics(p, y, amount, codes, wr, wc, k))
    boots = {m: pd.DataFrame(v) for m, v in boots.items()}
    lo = pd.DataFrame({m: b.quantile(0.025) for m, b in boots.items()}).T
    hi = pd.DataFrame({m: b.quantile(0.975) for m, b in boots.items()}).T
    diffs = {}
    if reference is not None:
        for m in prep:
            d = boots[m] - boots[reference]
            diffs[m] = {c: f"{point.loc[m, c] - point.loc[reference, c]:+.3f} "
                           f"[{d[c].quantile(0.025):+.3f}, {d[c].quantile(0.975):+.3f}]" for c in d.columns}
    return Report(point, lo, hi, pd.DataFrame(diffs).T, boots)


def simplest_not_worse(report: Report, order: list[str], metric: str = "ap") -> tuple[str, str]:
    """The simplest model whose gap to the best one is within noise (paired 95% CI includes 0).

    `order` lists models from simplest to most complex. Returns (name, explanation).
    """
    best = report.point[metric].idxmax()
    if report.boots is None:
        return best, f"{best} has the highest {metric} (no bootstrap available)"
    for cand in order:
        if cand == best:
            return best, f"{best} has the highest {metric} and simpler models are significantly worse"
        d = report.boots[best][metric] - report.boots[cand][metric]
        if d.quantile(0.025) <= 0:
            return cand, (f"{cand}: {metric} {report.point.loc[cand, metric]:.3f} vs best {best} "
                          f"{report.point.loc[best, metric]:.3f}, difference within noise "
                          f"[{d.quantile(0.025):+.3f}, {d.quantile(0.975):+.3f}]")
    return best, f"{best} has the highest {metric}"


def schemes_caught_curve(score, day, cluster, y, ks) -> np.ndarray:
    """Share of laundering schemes with ≥1 alert, for each alert budget k (per day)."""
    r = rank_in_day(score, day)
    pos = np.asarray(y) == 1
    best = pd.Series(r[pos]).groupby(np.asarray(cluster)[pos]).min().to_numpy()
    return np.array([(best <= k).mean() for k in ks])


def recall_by_pattern(alert_mask, y, cluster, pattern) -> pd.Series:
    """Share of schemes caught per scheme type (FAN-OUT, CYCLE, ...)."""
    pos = np.asarray(y) == 1
    df = pd.DataFrame({"cluster": np.asarray(cluster)[pos], "pattern": np.asarray(pattern)[pos],
                       "alert": np.asarray(alert_mask)[pos]})
    df["pattern"] = df["pattern"].replace("", "UNLABELLED")
    per = df.groupby("cluster").agg(pattern=("pattern", "first"), caught=("alert", "max"))
    return per.groupby("pattern")["caught"].mean()
