"""From scores to an alert budget: how many alerts per day should analysts review?

The data has no fines or recovery amounts, so the economics are explicit scenarios:
    cost_per_alert     analyst time for one alert (default 15 ≈ 20 minutes of a loaded salary)
    value_per_scheme   what detecting one laundering scheme is worth (avoided fines, reporting duty met)
    share_of_amount    optional extra value as a share of the scheme's laundered money
One alert on any transaction of a scheme counts as detecting the scheme: the investigator then
follows the money through the graph (which is what the demo page shows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import rank_in_day


@dataclass
class Economics:
    cost_per_alert: float = 15.0
    value_per_scheme: float = 2_000.0
    share_of_amount: float = 0.0


def _scheme_table(score, day, cluster, y, amount) -> pd.DataFrame:
    r = rank_in_day(score, day)
    pos = np.asarray(y) == 1
    return pd.DataFrame({"cluster": np.asarray(cluster)[pos], "rank": r[pos],
                         "amount": np.asarray(amount, float)[pos]}).groupby("cluster").agg(
        best_rank=("rank", "min"), amount=("amount", "sum"))


def value_curve(score, day, cluster, y, amount, ks, econ: Economics = Economics()) -> pd.DataFrame:
    """Net value of reviewing the top-k transactions of every day, for each k in `ks`."""
    schemes = _scheme_table(score, day, cluster, y, amount)
    n_days = len(np.unique(day))
    day_sizes = pd.Series(np.asarray(day)).value_counts()
    rows = []
    for k in ks:
        caught = schemes["best_rank"] <= k
        n_alerts = float(np.minimum(day_sizes, k).sum())
        value = caught.sum() * econ.value_per_scheme + econ.share_of_amount * schemes.loc[caught, "amount"].sum()
        cost = n_alerts * econ.cost_per_alert
        rows.append({"alerts_per_day": k, "alerts": n_alerts, "schemes_caught": int(caught.sum()),
                     "scheme_recall": float(caught.mean()) if len(schemes) else np.nan,
                     "amount_caught_usd": float(schemes.loc[caught, "amount"].sum()),
                     "cost": cost, "value": value, "net": value - cost, "net_per_day": (value - cost) / n_days})
    return pd.DataFrame(rows)


def alerts_needed(score, day, cluster, y, target: float) -> int | None:
    """Smallest daily alert budget that detects at least `target` share of schemes (None if never)."""
    schemes = _scheme_table(score, day, cluster, y, np.zeros(len(y)))
    if len(schemes) == 0:
        return None
    need = np.sort(schemes["best_rank"].to_numpy())
    idx = int(np.ceil(target * len(need))) - 1
    return int(need[max(idx, 0)])
