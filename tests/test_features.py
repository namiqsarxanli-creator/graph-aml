import numpy as np
import pandas as pd

from graphaml import features


def test_prior_window_counts_strictly_earlier_events_in_window():
    g = np.array([0, 0, 0, 1])
    t = np.array([0, 100, 200, 150])
    v = np.array([1.0, 2.0, 4.0, 8.0])
    r = features.prior_window(g, t, v, q_group=np.array([0, 0, 1, 1]), q_time=np.array([200, 201, 150, 151]),
                              window=150)
    # [50,200) -> {100}; [51,201) -> {100,200}; group 1 at its own time -> none; [1,151) -> {150}
    assert r.count.tolist() == [1, 2, 0, 1]
    assert r.total.tolist() == [2.0, 6.0, 0.0, 8.0]
    assert r.gap[0] == 100 and np.isinf(r.gap[2])


def test_no_feature_uses_the_future(std, exp):
    """Delete everything from the validation period on and recompute: earlier rows must not change."""
    cut_day = exp.days("valid")[0]
    cut = int(np.searchsorted(exp.df["t"].to_numpy(), cut_day * features.DAY))
    past = features.prepare(std.iloc[:cut])
    np.testing.assert_allclose(features.history_features(past).to_numpy(), exp.feats.history.iloc[:cut].to_numpy())
    np.testing.assert_allclose(features.graph_features(past).to_numpy(), exp.feats.graph.iloc[:cut].to_numpy())
    np.testing.assert_allclose(features.edge_features(past).to_numpy(), exp.feats.edge.iloc[:cut].to_numpy())


def test_labels_are_not_features(exp):
    shuffled = exp.df.copy()
    shuffled["is_laundering"] = np.random.default_rng(0).permutation(shuffled["is_laundering"].to_numpy())
    shuffled["pattern"] = ""
    np.testing.assert_allclose(features.history_features(shuffled).to_numpy(), exp.feats.history.to_numpy())
    np.testing.assert_allclose(features.graph_features(shuffled).to_numpy(), exp.feats.graph.to_numpy())


def test_reciprocal_pair_and_pair_counts():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2022-09-01 00:00", "2022-09-01 01:00", "2022-09-01 02:00", "2022-09-02 00:00"]),
        "src_key": ["a", "b", "a", "a"], "dst_key": ["b", "a", "b", "c"],
        "from_bank": 1, "to_bank": 1, "amount_paid": 10.0, "amount_received": 10.0, "amount_usd": 10.0,
        "pay_currency": "US Dollar", "recv_currency": "US Dollar", "payment_format": "ACH",
        "is_laundering": 0, "attempt_id": -1, "pattern": "",
    })
    p = features.prepare(df)
    h = features.history_features(p)
    assert np.expm1(h["pair_n_all"]).round().tolist() == [0, 0, 1, 0]        # a->b seen once before row 2
    assert np.expm1(h["rev_pair_n_all"]).round().tolist() == [0, 1, 1, 0]    # b->a after a->b, and vice versa
    g = features.graph_features(p)
    assert g["reciprocal_pair"].tolist() == [1, 1, 1, 0]                     # same-day graph has both directions
