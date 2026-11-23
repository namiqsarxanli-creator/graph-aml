"""Train every model, pick the served one on validation, evaluate ONCE on test, save the bundle.

    python scripts/train.py --data synthetic                       # 2 minutes on a laptop CPU
    python scripts/train.py --data ibm --ibm-dir data/raw          # IBM HI-Small, use a GPU (Colab)

Writes models/aml_bundle.pkl, models/model_meta.json and reports/results_<data>.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphaml import artifact, business, data, metrics, pipeline, synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "ibm"], default="synthetic")
    ap.add_argument("--ibm-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--size", default="HI-Small")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-dir", default=str(artifact.MODEL_DIR))
    ap.add_argument("--reports", default=str(ROOT / "reports"))
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.data == "synthetic":
        std = synthetic.make_dataset(seed=a.seed)
        source = f"synthetic (seed {a.seed})"
    else:
        trans, pat = data.find_ibm_files(a.ibm_dir, a.size)
        std = data.load_ibm(trans, pat, nrows=a.nrows)
        source = f"IBM AML {a.size}" + (f", first {a.nrows:,} rows" if a.nrows else "")
    std, tail = data.drop_simulation_tail(std)
    print(f"loaded {len(std):,} rows from {source} in {time.time() - t0:.0f}s; dropped simulation tail: {tail}")

    exp = pipeline.build_experiment(std)
    models = pipeline.train_models(exp, gnn_kwargs={"epochs": a.epochs, "seed": a.seed})
    scores = pipeline.predict_models(models, exp.feats, exp.snaps, exp.days("valid") + exp.days("test"))
    thr = pipeline.f1_thresholds(exp, scores)
    k = exp.default_k()

    val = pipeline.evaluate_split(exp, scores, "valid", k, thr, n_boot=a.n_boot)
    served, why = metrics.simplest_not_worse(val, pipeline.MODEL_ORDER, "ap")
    print(f"served model: {why}")
    test = pipeline.evaluate_split(exp, scores, "test", k, thr, n_boot=a.n_boot)   # the one and only look
    pd.set_option("display.width", 200)
    print(test.table())

    te = exp.rows("test")
    d = exp.df.iloc[te]
    by_pattern = pd.DataFrame({m: metrics.recall_by_pattern(metrics.alerts(s[te], d["day"], k), d["is_laundering"],
                                                            d["cluster"], d["pattern"]) for m, s in scores.items()})
    need = {m: {f"{int(q * 100)}%": business.alerts_needed(s[te], d["day"], d["cluster"], d["is_laundering"], q)
                for q in (0.5, 0.8)} for m, s in scores.items()}

    meta = pipeline.make_metadata(exp, models, source, served, why, k, thr, val, test)
    split_days = meta["split_days"]
    artifact.save(models, meta, Path(a.model_dir) / artifact.BUNDLE_PATH.name, Path(a.model_dir) / artifact.META_PATH.name)

    rep = Path(a.reports)
    rep.mkdir(parents=True, exist_ok=True)
    lines = [f"# Results on {source}", "",
             f"Test days {split_days['test'][0]} to {split_days['test'][1]}, {len(te):,} transactions, "
             f"{int(d['is_laundering'].sum()):,} laundering, alert budget {k} per day. "
             "95% cluster-bootstrap intervals (whole schemes resampled).", "",
             test.table().rename(index=pipeline.MODEL_LABELS).to_markdown(), "",
             "## Paired difference to the rules baseline", "",
             test.vs_reference.rename(index=pipeline.MODEL_LABELS).to_markdown(), "",
             f"## Schemes caught at {k} alerts/day, by type", "",
             by_pattern.round(2).to_markdown(), "",
             "## Alerts per day needed to catch 50% / 80% of schemes", "",
             pd.DataFrame(need).T.to_markdown(), "",
             f"Served model: {why}.", ""]
    (rep / f"results_{a.data}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {rep / f'results_{a.data}.md'}")
    return meta


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
