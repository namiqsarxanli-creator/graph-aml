"""Features at three levels of information, kept strictly apart so every comparison is fair.

1. `edge_features`     what the transaction itself says (amount, format, currency, hour).
2. `history_features`  what a *real-time* engine knows when the payment arrives: the two accounts'
                       activity strictly before this minute (counts, sums, 24h bursts, new counterparty,
                       money recently received and now sent on). Rule engines live here.
3. `graph_features`    what a *daily batch* monitor knows the next morning: the transaction graph up
                       to the end of the transaction's day, summarised per account and per neighbour.
                       The GNN sees exactly this information too (through message passing), so
                       "LightGBM + graph features" vs "GNN" is an apples-to-apples comparison.

Leakage rule: nothing ever uses labels, and nothing uses data from after the end of the
transaction's day. `tests/test_features.py` checks this by deleting the future and recomputing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CURRENCIES, FORMATS

DAY = 86_400
_BIG = np.int64(1) << np.int64(36)   # > any time offset in seconds we will see (~2000 years)


# --------------------------------------------------------------------------- preparation

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer account codes, seconds since start and day index to a standardized table."""
    df = df.copy()
    n = len(df)
    codes, uniques = pd.factorize(pd.concat([df["src_key"], df["dst_key"]], ignore_index=True))
    df["src"] = codes[:n].astype(np.int64)
    df["dst"] = codes[n:].astype(np.int64)
    t0 = df["ts"].min().floor("D")
    df["t"] = ((df["ts"] - t0).dt.total_seconds()).astype(np.int64)
    df["day"] = (df["t"] // DAY).astype(np.int64)
    df.attrs["n_accounts"] = len(uniques)
    df.attrs["t0"] = t0
    # unit used by bootstrap and by "schemes caught": a laundering attempt, or a single row
    df["cluster"] = np.where(df["attempt_id"] >= 0, df["attempt_id"],
                             df["attempt_id"].max() + 1 + np.arange(n)).astype(np.int64)
    return df


# --------------------------------------------------------------------------- 1. edge

def edge_features(df: pd.DataFrame) -> pd.DataFrame:
    paid, recv = df["amount_paid"].to_numpy(float), df["amount_received"].to_numpy(float)
    cents = np.round(paid * 100).astype(np.int64)
    out = {
        "log_amount_usd": np.log1p(df["amount_usd"].to_numpy(float)),
        "log_fx_ratio": np.clip(np.log((recv + 1e-9) / (paid + 1e-9)), -15, 15),
        "cross_currency": (df["pay_currency"] != df["recv_currency"]).to_numpy(),
        "same_bank": (df["from_bank"] == df["to_bank"]).to_numpy(),
        "self_transfer": (df["src_key"] == df["dst_key"]).to_numpy(),
        "round_100": (cents % 10_000 == 0),
        "whole_units": (cents % 100 == 0),
        "hour": df["ts"].dt.hour.to_numpy(),
    }
    for f in FORMATS:
        out[f"fmt_{f}"] = (df["payment_format"] == f).to_numpy()
    for c in CURRENCIES:
        out[f"cur_{c}"] = (df["pay_currency"] == c).to_numpy()
    return pd.DataFrame(out, index=df.index).astype(np.float32)


# --------------------------------------------------------------------------- 2. history (strict past)

@dataclass
class _Prior:
    count: np.ndarray
    total: np.ndarray
    gap: np.ndarray     # seconds since the last event of the group, inf if none


def prior_window(ev_group, ev_time, ev_value, q_group, q_time, window: int | None = None) -> _Prior:
    """For each query (group, time): count/sum of events of the same group with time in [t-window, t).

    Strictly before t, so the query's own transaction (and anything in the same minute) is excluded.
    Fully vectorised: one sort of the events, two binary searches per query.
    """
    ev_group = np.asarray(ev_group, np.int64)
    ev_time = np.asarray(ev_time, np.int64)
    order = np.lexsort((ev_time, ev_group))
    keys = ev_group[order] * _BIG + ev_time[order]
    times = ev_time[order]
    # running sum restarted for every account: a cumulative sum over all 5M rows reaches ~1e15 USD on IBM
    # data, and differences of such numbers lose small amounts (and change when rows are added later)
    cg = pd.Series(np.asarray(ev_value, float)[order]).groupby(ev_group[order], sort=False).cumsum().to_numpy()
    q_group = np.asarray(q_group, np.int64)
    q_time = np.asarray(q_time, np.int64)
    gstart = np.searchsorted(keys, q_group * _BIG, side="left")
    hi = np.searchsorted(keys, q_group * _BIG + q_time, side="left")
    lo = gstart if window is None else np.searchsorted(keys, q_group * _BIG + np.maximum(q_time - window, 0), "left")

    def before(p):   # sum of the group's events at positions [gstart, p)
        out = np.zeros(len(p))
        ok = p > gstart
        out[ok] = cg[p[ok] - 1]
        return out

    has_prev = hi > gstart
    gap = np.full(len(q_time), np.inf)
    gap[has_prev] = q_time[has_prev] - times[hi[has_prev] - 1]
    return _Prior(count=(hi - lo).astype(float), total=np.maximum(before(hi) - before(lo), 0.0), gap=gap)


def _pair_codes(src, dst, n_accounts):
    pair = src.astype(np.int64) * n_accounts + dst.astype(np.int64)
    codes, uniq = pd.factorize(pair)
    rev = pd.Index(uniq).get_indexer(dst.astype(np.int64) * n_accounts + src.astype(np.int64))
    rev = np.where(rev < 0, len(uniq), rev)   # pair never seen -> an empty group
    return codes.astype(np.int64), rev.astype(np.int64)


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Account activity strictly before each transaction (what a real-time engine can know)."""
    src, dst, t = df["src"].to_numpy(), df["dst"].to_numpy(), df["t"].to_numpy()
    amt = df["amount_usd"].to_numpy(float)
    n_acc = df.attrs.get("n_accounts", int(max(src.max(), dst.max()) + 1))
    out = {}
    # (feature prefix, which account's events, queried for which account)
    for name, ev_g, q_g in (("s_out", src, src), ("s_in", dst, src), ("d_in", dst, dst), ("d_out", src, dst)):
        allw = prior_window(ev_g, t, amt, q_g, t)
        day = prior_window(ev_g, t, amt, q_g, t, window=DAY)
        out[f"{name}_n_all"] = np.log1p(allw.count)
        out[f"{name}_amt_all"] = np.log1p(allw.total)
        out[f"{name}_n_24h"] = np.log1p(day.count)
        out[f"{name}_amt_24h"] = np.log1p(day.total)
        out[f"{name}_gap_h"] = np.log1p(np.minimum(allw.gap, 30 * DAY) / 3600)
        if name == "s_in":
            # money that came in during the last day and is now being sent on (pass-through / mule)
            out["s_passthrough_24h"] = np.log1p(day.total) - np.log1p(amt)

    pair, rev = _pair_codes(src, dst, n_acc)
    out["pair_n_all"] = np.log1p(prior_window(pair, t, np.ones(len(t)), pair, t).count)
    out["rev_pair_n_all"] = np.log1p(prior_window(pair, t, np.ones(len(t)), rev, t).count)
    first = ~pd.Series(pair).duplicated().to_numpy()
    ones = np.ones(first.sum())
    out["s_uniq_dst_all"] = np.log1p(prior_window(src[first], t[first], ones, src, t).count)
    out["d_uniq_src_all"] = np.log1p(prior_window(dst[first], t[first], ones, dst, t).count)
    return pd.DataFrame(out, index=df.index).astype(np.float32)


# --------------------------------------------------------------------------- 3. graph (daily batch)

NODE_BASE = ["log_in_deg", "log_out_deg", "log_in_amt", "log_out_amt", "log_uniq_in", "log_uniq_out",
             "ach_share_in", "ach_share_out", "flow_through", "cross_cur_share_out"]
NEIGHBOR_OF = ["log_in_deg", "log_out_deg", "flow_through", "ach_share_in"]


class Snapshots:
    """Day-by-day views of the growing transaction graph (rows must be sorted by time).

    `prefix(d)` = number of transactions with time < end of day d, so the graph of day d is the
    first `prefix(d)` rows. Everything per day is O(rows) with bincount.
    """

    def __init__(self, df: pd.DataFrame):
        self.src = np.array(df["src"], dtype=np.int64)   # writable copies (torch shares their memory)
        self.dst = np.array(df["dst"], dtype=np.int64)
        self.t = np.array(df["t"], dtype=np.int64)
        assert np.all(np.diff(self.t) >= 0), "rows must be sorted by time"
        self.amt = df["amount_usd"].to_numpy(float)
        self.ach = (df["payment_format"] == "ACH").to_numpy(float)
        self.cross = (df["pay_currency"] != df["recv_currency"]).to_numpy(float)
        self.n = df.attrs.get("n_accounts", int(max(self.src.max(), self.dst.max()) + 1))
        pair, rev = _pair_codes(self.src, self.dst, self.n)
        first_idx = pd.Series(np.arange(len(pair))).groupby(pair).min()
        self.first_pos = np.full(len(first_idx) + 1, np.iinfo(np.int64).max)  # last slot: never-seen pair
        self.first_pos[first_idx.index.to_numpy()] = first_idx.to_numpy()
        self.pair, self.rev = pair, rev
        self.is_first = self.first_pos[pair] == np.arange(len(pair))
        self.days = np.unique(df["day"].to_numpy())
        self.day_of = df["day"].to_numpy()

    def prefix(self, day: int) -> int:
        return int(np.searchsorted(self.t, (day + 1) * DAY, side="left"))

    def rows_of(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.day_of == day)

    def node_base(self, m: int) -> np.ndarray:
        """NODE_BASE features of every account in the graph made of the first m rows."""
        s, d, a = self.src[:m], self.dst[:m], self.amt[:m]
        n = self.n
        out_deg = np.bincount(s, minlength=n).astype(float)
        in_deg = np.bincount(d, minlength=n).astype(float)
        out_amt = np.bincount(s, weights=a, minlength=n)
        in_amt = np.bincount(d, weights=a, minlength=n)
        first = self.is_first[:m]
        uniq_out = np.bincount(s[first], minlength=n)
        uniq_in = np.bincount(d[first], minlength=n)
        ach_out = np.bincount(s, weights=self.ach[:m], minlength=n) / np.maximum(out_deg, 1)
        ach_in = np.bincount(d, weights=self.ach[:m], minlength=n) / np.maximum(in_deg, 1)
        cross_out = np.bincount(s, weights=self.cross[:m], minlength=n) / np.maximum(out_deg, 1)
        flow = np.minimum(in_amt, out_amt) / np.maximum(np.maximum(in_amt, out_amt), 1e-9)
        return np.column_stack([np.log1p(in_deg), np.log1p(out_deg), np.log1p(in_amt), np.log1p(out_amt),
                                np.log1p(uniq_in), np.log1p(uniq_out), ach_in, ach_out, flow, cross_out])

    def neighbor_means(self, m: int, base: np.ndarray) -> np.ndarray:
        """Mean of selected node features over out-neighbours and over in-neighbours (2-hop view)."""
        s, d = self.src[:m], self.dst[:m]
        cols = [NODE_BASE.index(c) for c in NEIGHBOR_OF]
        out_deg = np.maximum(np.bincount(s, minlength=self.n), 1)
        in_deg = np.maximum(np.bincount(d, minlength=self.n), 1)
        res = []
        for c in cols:
            res.append(np.bincount(s, weights=base[d, c], minlength=self.n) / out_deg)   # where my money goes
        for c in cols:
            res.append(np.bincount(d, weights=base[s, c], minlength=self.n) / in_deg)    # where my money comes from
        return np.column_stack(res)


NODE_ALL = NODE_BASE + [f"out_nb_{c}" for c in NEIGHBOR_OF] + [f"in_nb_{c}" for c in NEIGHBOR_OF]


def graph_features(df: pd.DataFrame, snaps: Snapshots | None = None) -> pd.DataFrame:
    """Per transaction: both accounts' node features in the graph at the end of its day."""
    snaps = snaps or Snapshots(df)
    k = len(NODE_ALL)
    X = np.zeros((len(df), 2 * k + 1), dtype=np.float32)
    for day in snaps.days:
        m = snaps.prefix(day)
        rows = snaps.rows_of(day)
        base = snaps.node_base(m)
        node = np.hstack([base, snaps.neighbor_means(m, base)])
        X[rows, :k] = node[snaps.src[rows]]
        X[rows, k:2 * k] = node[snaps.dst[rows]]
        # money also flows back from receiver to sender somewhere in the graph so far (2-cycle)
        X[rows, -1] = snaps.first_pos[snaps.rev[rows]] < m
    cols = [f"gs_{c}" for c in NODE_ALL] + [f"gd_{c}" for c in NODE_ALL] + ["reciprocal_pair"]
    return pd.DataFrame(X, columns=cols, index=df.index)


# --------------------------------------------------------------------------- all together

class FeatureSet:
    """All features in one float32 frame; the three levels are column subsets (no copies kept)."""

    def __init__(self, edge: pd.DataFrame, history: pd.DataFrame, graph: pd.DataFrame):
        self.edge_cols, self.history_cols, self.graph_cols = list(edge.columns), list(history.columns), list(graph.columns)
        self.full = pd.concat([edge, history, graph], axis=1)

    @property
    def tabular_cols(self) -> list[str]:
        return self.edge_cols + self.history_cols

    @property
    def tabular(self) -> pd.DataFrame:
        return self.full[self.tabular_cols]

    @property
    def edge(self) -> pd.DataFrame:
        return self.full[self.edge_cols]

    @property
    def history(self) -> pd.DataFrame:
        return self.full[self.history_cols]

    @property
    def graph(self) -> pd.DataFrame:
        return self.full[self.graph_cols]


def build(df: pd.DataFrame, snaps: Snapshots | None = None) -> FeatureSet:
    return FeatureSet(edge_features(df), history_features(df), graph_features(df, snaps))
