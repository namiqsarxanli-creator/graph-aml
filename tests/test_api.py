import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphaml import artifact, data

pytestmark = pytest.mark.skipif(not artifact.BUNDLE_PATH.exists(), reason="run scripts/train.py first")

from fastapi.testclient import TestClient  # noqa: E402

from graphaml.api import app, get_scorer  # noqa: E402

client = TestClient(app)
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "transactions_example.csv"


def _post(path, key=None, params=None):
    with open(EXAMPLE, "rb") as f:
        return client.post(path, files={"file": ("tx.csv", f, "text/csv")}, params=params or {},
                           headers={"X-API-Key": key} if key else {})


def test_health_is_open():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_alerts_require_key_when_configured(monkeypatch):
    monkeypatch.setenv("AML_API_KEY", "secret")
    assert _post("/alerts").status_code == 401
    r = _post("/alerts", key="secret")
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < len(body["alerts"]) <= body["alerts_per_day"]
    risks = [a["risk"] for a in body["alerts"]]
    assert risks == sorted(risks, reverse=True)
    assert all("reasons" in a for a in body["alerts"])


def test_training_serving_skew():
    """CSV -> API gives the same scores as the training pipeline on the same transactions."""
    from graphaml import pipeline

    scorer = get_scorer()
    exp = pipeline.build_experiment(data.load_ibm(EXAMPLE), verbose=False)
    last_day = exp.snaps.days[-1]
    ref = pipeline.predict_models(scorer.models, exp.feats, exp.snaps, [last_day])[scorer.served]
    rows = np.flatnonzero(exp.df["day"].to_numpy() == last_day)
    r = _post("/score/batch")
    assert r.status_code == 200
    out = pd.read_csv(io.StringIO(r.text))
    assert len(out) == len(rows)
    np.testing.assert_allclose(np.sort(out["risk"].to_numpy()), np.sort(np.round(ref[rows], 5)), atol=1e-5)


def test_bad_file_is_rejected():
    r = client.post("/alerts", files={"file": ("x.csv", b"not,a,transaction,file\n1,2,3,4", "text/csv")})
    assert r.status_code in (400, 422)


def test_demo_case_and_graph():
    c = client.get("/demo/case", params={"seed": 2, "k": 10})
    assert c.status_code == 200
    body = c.json()
    assert len(body["alerts"]) == 10 and "rules" in body["comparison"]
    g = client.get("/demo/graph", params={"seed": 2, "txn": body["alerts"][0]["txn"]})
    assert g.status_code == 200
    gj = g.json()
    assert any(e["focal"] for e in gj["edges"]) and any(n["focal"] for n in gj["nodes"])


def test_drift_runs():
    r = _post("/drift")
    assert r.status_code == 200
    assert r.json()["overall"] in {"stable", "watch", "drift"}
