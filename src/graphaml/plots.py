"""Figures for the notebook and README (matplotlib only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RISK, OK, NODE, ACCENT, CYCLE = "#d6336c", "#1f9d6b", "#9aa3b2", "#2f5bea", "#7b3fe4"
MODEL_COLORS = {"rules": "#9aa3b2", "lgbm_tabular": "#e8890c", "lgbm_graph": "#2f5bea", "gnn": "#d6336c",
                "ensemble": "#7b3fe4"}


def _plt():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    return plt


def scheme_examples(df: pd.DataFrame, patterns=None, max_per_row: int = 4, seed: int = 0):
    """One drawn example of each laundering scheme type (accounts = dots, payments = arrows)."""
    import networkx as nx

    plt = _plt()
    pats = [p for p in (patterns or sorted(df.loc[df["pattern"] != "", "pattern"].unique()))]
    n = len(pats)
    rows = int(np.ceil(n / max_per_row))
    fig, axes = plt.subplots(rows, max_per_row, figsize=(3.2 * max_per_row, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)
    for ax, p in zip(axes, pats):
        attempts = df.loc[df["pattern"] == p, "attempt_id"].unique()
        sizes = df[df["attempt_id"].isin(attempts)].groupby("attempt_id").size()
        pick = sizes[(sizes >= 4) & (sizes <= 14)]
        a = rng.choice(pick.index if len(pick) else sizes.index)
        e = df[df["attempt_id"] == a]
        g = nx.MultiDiGraph()
        g.add_edges_from(zip(e["src_key"], e["dst_key"]))
        pos = nx.circular_layout(g) if p == "CYCLE" else nx.kamada_kawai_layout(nx.DiGraph(g))
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=90, node_color=RISK)
        nx.draw_networkx_edges(g, pos, ax=ax, arrowsize=10, edge_color="#555", width=1.0,
                               connectionstyle="arc3,rad=0.08", node_size=90)
        ax.set_title(f"{p}\n{len(e)} payments, ${e['amount_usd'].sum():,.0f}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def daily_volume(df: pd.DataFrame, split: np.ndarray):
    plt = _plt()
    d = pd.DataFrame({"day": df["ts"].dt.floor("D"), "y": df["is_laundering"], "split": split})
    g = d.groupby("day").agg(n=("y", "size"), rate=("y", "mean"), split=("split", "first"))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    colors = g["split"].map({"train": "#9aa3b2", "valid": "#e8890c", "test": "#2f5bea"})
    ax.bar(g.index, g["n"], color=colors, width=0.8)
    ax.set_ylabel("transactions per day")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["rate"] * 100, color=RISK, marker="o", lw=1.5)
    ax2.set_ylabel("laundering, % of transactions", color=RISK)
    ax2.grid(False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=s) for s, c in (("train", "#9aa3b2"), ("valid", "#e8890c"),
                                                            ("test", "#2f5bea"))], loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def training_curve(history: list[dict]):
    plt = _plt()
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(h["epoch"], h["train_loss"], color="#555", label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax2 = ax.twinx()
    ax2.plot(h["epoch"], h["val_ap"], color=RISK, marker="o", ms=3, label="validation AP")
    ax2.set_ylabel("validation average precision", color=RISK)
    ax2.grid(False)
    fig.tight_layout()
    return fig


def metric_bars(report, metric: str, labels: dict | None = None):
    plt = _plt()
    p, lo, hi = report.point[metric], report.ci_low[metric], report.ci_high[metric]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(p) + 0.8))
    y = np.arange(len(p))[::-1]
    ax.barh(y, p, color=[MODEL_COLORS.get(m, "#888") for m in p.index], alpha=0.85)
    ax.errorbar(p, y, xerr=[p - lo, hi - p], fmt="none", ecolor="#222", capsize=3, lw=1)
    ax.set_yticks(y, [(labels or {}).get(m, m) for m in p.index])
    ax.set_xlabel(f"{metric} (test, 95% cluster-bootstrap CI)")
    fig.tight_layout()
    return fig


def pattern_recall(table: pd.DataFrame, k: int):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    w = 0.8 / table.shape[1]
    x = np.arange(len(table))
    for i, m in enumerate(table.columns):
        ax.bar(x + i * w - 0.4 + w / 2, table[m], width=w, label=m, color=MODEL_COLORS.get(m, "#888"))
    ax.set_xticks(x, table.index, rotation=20)
    ax.set_ylabel(f"schemes caught at {k} alerts/day")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    return fig


def value_curves(curves: dict[str, pd.DataFrame], title: str = ""):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for m, c in curves.items():
        axes[0].plot(c["alerts_per_day"], c["scheme_recall"], label=m, color=MODEL_COLORS.get(m, "#888"))
        axes[1].plot(c["alerts_per_day"], c["net_per_day"], label=m, color=MODEL_COLORS.get(m, "#888"))
    axes[0].set_ylabel("share of schemes detected")
    axes[1].set_ylabel("net value per day")
    for ax in axes:
        ax.set_xlabel("alerts reviewed per day")
    axes[1].axhline(0, color="#444", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def ego(edges: pd.DataFrame, focal_txn: int, keys: pd.Series, cycles=None):
    """Money trail around one alert: arrow colour = model risk, dashed = true laundering."""
    import networkx as nx

    plt = _plt()
    g = nx.MultiDiGraph()
    for i, e in edges.iterrows():
        g.add_edge(int(e.src), int(e.dst), key=int(i), risk=float(e.score), bad=bool(e.is_laundering), focal=int(i) == focal_txn)
    in_cycle = {v for c in (cycles or []) for v in c}
    pos = nx.spring_layout(nx.Graph(g), seed=1, k=0.6)
    fig, ax = plt.subplots(figsize=(8, 6))
    for style, sel in (("solid", lambda d: not d["bad"]), ("dashed", lambda d: d["bad"])):
        es = [(u, v) for u, v, d in g.edges(data=True) if sel(d)]
        cs = [ACCENT if d["focal"] else (RISK if d["risk"] > 0.5 else "#bbb") for u, v, d in g.edges(data=True) if sel(d)]
        nx.draw_networkx_edges(g, pos, edgelist=es, edge_color=cs, style=style, ax=ax, arrowsize=9, width=1.4,
                               connectionstyle="arc3,rad=0.1", node_size=60)
    focal = edges.loc[focal_txn] if focal_txn in edges.index else None
    colors = [ACCENT if focal is not None and v in (focal.src, focal.dst) else (CYCLE if v in in_cycle else NODE)
              for v in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=60, ax=ax)
    nx.draw_networkx_labels(g, pos, {v: keys[v].split(":")[1][-5:] for v in g.nodes if v in in_cycle or
                                     (focal is not None and v in (focal.src, focal.dst))}, font_size=7, ax=ax)
    ax.set_title("Money trail: blue = alert, red = high model risk, dashed = true laundering, purple = cycle")
    ax.axis("off")
    fig.tight_layout()
    return fig
