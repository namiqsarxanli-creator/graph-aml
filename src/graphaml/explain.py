"""What an investigator sees for one alert: the money trail around it and why the model flagged it."""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "log_amount_usd": "transaction amount",
    "fmt_ACH": "paid by ACH",
    "cross_currency": "currency conversion",
    "s_out_n_24h": "sender made many payments in the last 24h (fan-out)",
    "d_in_n_24h": "receiver got many payments in the last 24h (fan-in)",
    "s_passthrough_24h": "sender passes on money it just received",
    "s_in_n_24h": "sender received money in the last 24h",
    "pair_n_all": "first payment between these accounts",
    "rev_pair_n_all": "money has flowed back before",
    "reciprocal_pair": "money flows both ways between these accounts",
    "s_uniq_dst_all": "sender's number of distinct receivers",
    "d_uniq_src_all": "receiver's number of distinct senders",
    "gs_flow_through": "sender forwards most of what it receives",
    "gd_flow_through": "receiver forwards most of what it receives",
    "gs_log_uniq_out": "sender pays many different accounts",
    "gd_log_uniq_in": "receiver is paid by many different accounts",
    "gs_ach_share_out": "sender's payments are mostly ACH",
    "gd_ach_share_in": "receiver's incoming money is mostly ACH",
    "gs_out_nb_flow_through": "sender's receivers forward money on",
    "gd_out_nb_flow_through": "receiver's receivers forward money on",
    "gs_in_nb_flow_through": "sender's payers forward money on",
    "gd_in_nb_flow_through": "receiver's payers forward money on",
}


_NODE_WORDS = {"log_in_deg": "number of incoming payments", "log_out_deg": "number of outgoing payments",
               "log_in_amt": "money received", "log_out_amt": "money sent", "log_uniq_in": "distinct payers",
               "log_uniq_out": "distinct receivers", "ach_share_in": "share of ACH money received",
               "ach_share_out": "share of ACH payments sent", "flow_through": "forwards what it receives",
               "cross_cur_share_out": "share of currency conversions"}
_HIST_WORDS = {"s_out": "sender's past payments", "s_in": "sender's past receipts",
               "d_in": "receiver's past receipts", "d_out": "receiver's past payments"}


def label(feature: str) -> str:
    """Plain-language name of a feature, for alert explanations."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for pre, who in (("gs_", "sender"), ("gd_", "receiver")):
        if feature.startswith(pre):
            rest = feature[len(pre):]
            for nb, whose in (("out_nb_", "'s receivers"), ("in_nb_", "'s payers")):
                if rest.startswith(nb):
                    return f"{who}{whose}: {_NODE_WORDS.get(rest[len(nb):], rest[len(nb):])}"
            return f"{who}: {_NODE_WORDS.get(rest, rest)}"
    for pre, words in _HIST_WORDS.items():
        if feature.startswith(pre + "_"):
            rest = feature[len(pre) + 1:].replace("_", " ").replace("n ", "count, ").replace("amt ", "amount, ")
            return f"{words} ({rest})"
    if feature.startswith("fmt_"):
        return f"paid by {feature[4:]}"
    if feature.startswith("cur_"):
        return f"paid in {feature[4:]}"
    return feature.replace("_", " ")


def top_reasons(contrib_row: pd.Series, values_row: pd.Series, n: int = 4) -> list[dict]:
    """Features that pushed this score up the most (LightGBM TreeSHAP, log-odds units)."""
    pos = contrib_row[contrib_row > 0].sort_values(ascending=False).head(n)
    return [{"feature": f, "label": label(f), "value": float(values_row[f]), "contribution": round(float(c), 3)}
            for f, c in pos.items()]


def ego_network(df: pd.DataFrame, scores: np.ndarray, account: int, hops: int = 2, max_edges: int = 200,
                upto_t: int | None = None, per_node: int = 60) -> pd.DataFrame:
    """Transactions within `hops` steps of `account` (both directions), highest-risk first when capped.

    `df` is a prepared table (src, dst, t). Only rows with t < upto_t are used, so the view
    matches what was known when the alert was raised.
    """
    mask = np.ones(len(df), bool) if upto_t is None else df["t"].to_numpy() < upto_t
    idx = np.flatnonzero(mask)
    src, dst = df["src"].to_numpy()[idx], df["dst"].to_numpy()[idx]
    sc = np.nan_to_num(np.asarray(scores, float)[idx], nan=0.0)
    # incidence lists in CSR form: rows touching each account
    keep = src != dst
    ends = np.concatenate([src, dst[keep]])
    rows = np.concatenate([np.arange(len(src)), np.flatnonzero(keep)])
    order = np.argsort(ends, kind="stable")
    ends, rows = ends[order], rows[order]

    def incident(node):
        a, b = np.searchsorted(ends, node, "left"), np.searchsorted(ends, node, "right")
        r = rows[a:b]
        return r[np.argsort(-sc[r], kind="stable")[:per_node]]   # hubs: keep their riskiest links

    seen_nodes, picked = {int(account)}, set()
    frontier = deque([(int(account), 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        edges = incident(node)
        for j in edges:
            picked.add(j)
            for nb in (int(src[j]), int(dst[j])):
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    frontier.append((nb, depth + 1))
    picked = sorted(picked, key=lambda j: -sc[j])[:max_edges]
    out = df.iloc[idx[picked]].copy()
    out["score"] = sc[picked]
    return out.sort_values("t")


def cycles_through(edges: pd.DataFrame, account: int, max_len: int = 8, limit: int = 10) -> list[list[int]]:
    """Directed cycles in the ego network that pass through `account` (money coming back)."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_edges_from(zip(edges["src"], edges["dst"]))
    g.remove_edges_from(nx.selfloop_edges(g))
    if account not in g:
        return []
    out = []
    for cyc in nx.simple_cycles(g, length_bound=max_len):
        if account in cyc:
            i = cyc.index(account)
            out.append([int(v) for v in cyc[i:] + cyc[:i]])
            if len(out) >= limit:
                break
    return sorted(out, key=len)
