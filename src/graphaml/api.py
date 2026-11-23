"""HTTP service around the trained AML models.

    GET  /health          model, training data, test metrics (no auth)
    POST /alerts          CSV of transactions (history + day to score) -> today's alert queue with reasons
    POST /score/batch     same input -> CSV with a risk score for every scored transaction
    POST /drift           CSV -> PSI of every feature against the training period
    GET  /demo/case       synthetic day, scored by the served model and by the rules (demo page)
    GET  /demo/graph      money trail around one transaction of the demo day
    GET  /                demo page

Run locally:  uvicorn graphaml.api:app --app-dir src --port 8000
Set AML_API_KEY to require the `X-API-Key` header on every endpoint except /health and the demo.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import data, explain, metrics, synthetic
from .monitoring import drift_report
from .serving import Scored, Scorer

MAX_ROWS_CSV = 2_000_000
DEMO_SIZE = dict(n_accounts=1500, n_days=8, attempts_per_day=7.0)


@lru_cache
def get_scorer() -> Scorer:
    return Scorer()


def require_key(x_api_key: str | None = Header(None)) -> None:
    expected = os.environ.get("AML_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


app = FastAPI(
    title="Graph AML monitoring API",
    version="1.0.0",
    description="Ranks a day's transactions for anti-money-laundering review using the transaction graph, "
                "and shows the money trail behind every alert.",
)


async def _read_transactions(file: UploadFile) -> pd.DataFrame:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), names=data.RAW_COLS, header=0,
                         dtype={"Account": str, "Account.1": str})
        if len(df) > MAX_ROWS_CSV:
            raise HTTPException(status_code=413, detail=f"at most {MAX_ROWS_CSV:,} rows per file")
        if "Is Laundering" in df and df["Is Laundering"].isna().all():
            df["Is Laundering"] = 0
        df["Is Laundering"] = df["Is Laundering"].fillna(0)
        std = data.standardize(df)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - any parse failure is the caller's input problem
        raise HTTPException(status_code=400, detail=f"could not read transactions (IBM AML column layout "
                                                    f"expected: {', '.join(data.RAW_COLS)}): {e}") from e
    if len(std) == 0:
        raise HTTPException(status_code=400, detail="no transactions in file")
    if (std["amount_paid"] < 0).any():
        raise HTTPException(status_code=422, detail="negative amounts")
    return std


def _alert_items(scorer: Scorer, s: Scored, rows: np.ndarray, with_labels: bool = False) -> list[dict]:
    rows = rows[np.argsort(-s.risk[rows], kind="stable")]
    reasons = scorer.reasons(s, rows)
    flags = scorer.rule_flags(s, rows)
    d = s.df
    items = []
    for i, r in enumerate(rows):
        it = {
            "txn": int(r), "time": d["ts"].iat[r].strftime("%Y-%m-%d %H:%M"),
            "from_account": d["src_key"].iat[r], "to_account": d["dst_key"].iat[r],
            "amount": round(float(d["amount_paid"].iat[r]), 2), "currency": d["pay_currency"].iat[r],
            "amount_usd": round(float(d["amount_usd"].iat[r]), 2), "format": d["payment_format"].iat[r],
            "risk": round(float(s.risk[r]), 4), "reasons": reasons[i], "rule_flags": flags[i],
        }
        if with_labels:
            it["is_laundering"] = bool(d["is_laundering"].iat[r])
            it["pattern"] = d["pattern"].iat[r]
        items.append(it)
    return items


@app.get("/health")
def health(scorer: Scorer = Depends(get_scorer)) -> dict:
    m = scorer.meta
    return {"status": "ok", "served_model": m["served_model"], "created_utc": m["created_utc"],
            "trained_on": m["trained_on"], "alerts_per_day": m["alerts_per_day"],
            "test_metrics": m["test_metrics"].get(m["served_model"], {})}


@app.post("/alerts", dependencies=[Depends(require_key)])
async def alerts(file: UploadFile = File(..., description="transactions in IBM AML layout: history + day(s) to score"),
                 score_from: str | None = Query(None, description="first day to score (YYYY-MM-DD); default last day"),
                 k: int | None = Query(None, ge=1, le=10_000, description="alerts per day"),
                 scorer: Scorer = Depends(get_scorer)) -> dict:
    std = await _read_transactions(file)
    s = scorer.score(std, pd.Timestamp(score_from) if score_from else None, k)
    return {"model": scorer.served, "scored_transactions": int(len(s.rows)), "history_transactions": int(len(s.df) - len(s.rows)),
            "alerts_per_day": k or scorer.k, "alerts": _alert_items(scorer, s, np.flatnonzero(s.alert))}


@app.post("/score/batch", dependencies=[Depends(require_key)])
async def score_batch(file: UploadFile = File(...), score_from: str | None = None, k: int | None = Query(None, ge=1),
                      scorer: Scorer = Depends(get_scorer)):
    std = await _read_transactions(file)
    s = scorer.score(std, pd.Timestamp(score_from) if score_from else None, k)
    d = s.df.iloc[s.rows]
    out = pd.DataFrame({"timestamp": d["ts"].dt.strftime(data.TS_FORMAT), "from_account": d["src_key"],
                        "to_account": d["dst_key"], "amount_paid": d["amount_paid"], "payment_currency": d["pay_currency"],
                        "payment_format": d["payment_format"], "risk": s.risk[s.rows].round(5),
                        "alert": s.alert[s.rows]})
    buf = io.StringIO()
    out.sort_values("risk", ascending=False).to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=aml_scores.csv"})


@app.post("/drift", dependencies=[Depends(require_key)])
async def drift(file: UploadFile = File(...), scorer: Scorer = Depends(get_scorer)) -> dict:
    """Population Stability Index of each transaction/history feature against the training period."""
    from . import features

    std = await _read_transactions(file)
    df = features.prepare(std)
    X = pd.concat([features.edge_features(df), features.history_features(df)], axis=1)
    rep = drift_report(X, scorer.meta["reference_profile"])
    worst = rep["status"].map({"drift": 2, "watch": 1, "stable": 0}).max()
    return {"transactions": len(df), "overall": {2: "drift", 1: "watch", 0: "stable"}.get(worst, "unknown"),
            "features": rep.replace({np.nan: None}).to_dict(orient="records")}


# --------------------------------------------------------------------------- demo (synthetic data only)

@lru_cache(maxsize=8)
def _demo(seed: int) -> Scored:
    std = synthetic.make_dataset(seed=1000 + seed, **DEMO_SIZE)
    return get_scorer().score(std)       # history = first days, scored = the last day


def _caught(s: Scored, score: np.ndarray, k: int) -> dict:
    r = s.rows
    d = s.df.iloc[r]
    alert = metrics.alerts(score[r], d["day"], k)
    y = d["is_laundering"].to_numpy() == 1
    schemes = d.loc[y, "cluster"].nunique()
    caught = d.loc[y & alert, "cluster"].nunique()
    return {"alerts": int(alert.sum()), "true_alerts": int((alert & y).sum()), "schemes": int(schemes),
            "schemes_caught": int(caught),
            "laundered_usd": round(float(d.loc[y, "amount_usd"].sum()), 2),
            "laundered_usd_caught": round(float(d.loc[y, "amount_usd"].where(alert[y]).sum()), 2)}


@app.get("/demo/case")
def demo_case(seed: int = Query(0, ge=0, le=999), k: int = Query(20, ge=1, le=200),
              scorer: Scorer = Depends(get_scorer)) -> dict:
    """A synthetic bank day (never real data): alert queue of the served model vs the rules baseline."""
    s = _demo(seed)
    r = s.rows
    alert = np.zeros(len(s.df), bool)
    alert[r] = metrics.alerts(s.risk[r], s.df["day"].to_numpy()[r], k)
    comparison = {scorer.served: _caught(s, s.risk, k)}
    if "rules" in s.scores:
        comparison["rules"] = _caught(s, s.scores["rules"], k)
    return {"seed": seed, "day": s.df["ts"].iloc[r].min().strftime("%Y-%m-%d"), "transactions": int(len(r)),
            "history_transactions": int(len(s.df) - len(r)), "accounts": int(s.df.attrs["n_accounts"]),
            "model": scorer.served, "comparison": comparison,
            "alerts": _alert_items(scorer, s, np.flatnonzero(alert), with_labels=True)}


@app.get("/demo/graph")
def demo_graph(seed: int = Query(0, ge=0, le=999), txn: int = Query(..., ge=0), hops: int = Query(2, ge=1, le=3)) -> dict:
    s = _demo(seed)
    if txn >= len(s.df):
        raise HTTPException(status_code=404, detail="unknown transaction")
    d = s.df
    focal_src, focal_dst = int(d["src"].iat[txn]), int(d["dst"].iat[txn])
    risk = np.where(np.isnan(s.risk), 0.0, s.risk)
    edges = pd.concat([explain.ego_network(d, risk, focal_src, hops=hops, max_edges=90, per_node=8),
                       explain.ego_network(d, risk, focal_dst, hops=hops, max_edges=50, per_node=8)])
    edges = edges[~edges.index.duplicated()]
    if txn not in edges.index:
        edges = pd.concat([edges, d.loc[[txn]].assign(score=risk[txn])])
    cycles = explain.cycles_through(edges, focal_src)
    keys = pd.concat([d[["src", "src_key"]].rename(columns={"src": "id", "src_key": "key"}),
                      d[["dst", "dst_key"]].rename(columns={"dst": "id", "dst_key": "key"})]).drop_duplicates("id").set_index("id")["key"]
    ids = pd.unique(np.concatenate([edges["src"].to_numpy(), edges["dst"].to_numpy()]))
    node_risk = pd.concat([edges.groupby("src")["score"].max(), edges.groupby("dst")["score"].max()]).groupby(level=0).max()
    in_cycle = {v for c in cycles for v in c}
    nodes = [{"id": int(v), "label": keys[v], "risk": round(float(node_risk.get(v, 0.0)), 4),
              "focal": int(v) in (focal_src, focal_dst), "in_cycle": int(v) in in_cycle} for v in ids]
    scored_rows = set(s.rows.tolist())
    edge_items = [{"id": int(i), "source": int(e.src), "target": int(e.dst), "amount_usd": round(float(e.amount_usd), 2),
                   "time": e.ts.strftime("%m-%d %H:%M"), "format": e.payment_format, "risk": round(float(e.score), 4),
                   "scored_today": int(i) in scored_rows, "is_laundering": bool(e.is_laundering), "pattern": e.pattern,
                   "focal": int(i) == txn}
                  for i, e in edges.iterrows()]
    return {"txn": txn, "nodes": nodes, "edges": edge_items, "cycles": [[keys[v] for v in c] for c in cycles]}


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
