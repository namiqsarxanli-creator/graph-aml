import numpy as np
from sklearn.metrics import average_precision_score

from graphaml import business, metrics


def test_weighted_ap_matches_sklearn_without_weights():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(float)
    s = rng.random(2000) + y * 0.5
    order = np.argsort(-s)
    assert np.isclose(metrics._weighted_ap(order, y, np.ones_like(y)), average_precision_score(y, s))


def test_alerts_are_top_k_per_day():
    s = np.array([0.9, 0.1, 0.8, 0.7, 0.6, 0.5])
    day = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.alerts(s, day, 2).tolist() == [True, False, True, True, True, False]


def test_scheme_recall_counts_a_scheme_once_if_any_row_alerted():
    y = np.array([1, 1, 1, 0, 1])
    cluster = np.array([7, 7, 8, 9, 10])
    day = np.zeros(5)
    s = np.array([0.9, 0.0, 0.1, 0.5, 0.2])        # top-1: row 0 -> scheme 7 caught; 8 and 10 missed
    rep = metrics.evaluate({"m": s}, y, day, cluster, np.ones(5), k=1, n_boot=0)
    assert np.isclose(rep.point.loc["m", "schemes@1"], 1 / 3)
    assert np.isclose(rep.point.loc["m", "recall@1"], 1 / 4)
    assert business.alerts_needed(s, day, cluster, y, 1.0) == 4


def test_bootstrap_interval_contains_point_and_model_choice():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.03).astype(int)
    cluster = np.where(y == 1, rng.integers(0, 30, n), 1000 + np.arange(n))
    day = rng.integers(0, 5, n)
    a, b = rng.random(n) + y, rng.random(n)
    rep = metrics.evaluate({"a": a, "b": b}, y, day, cluster, np.ones(n), k=20, n_boot=100, reference="b")
    assert rep.ci_low.loc["a", "ap"] <= rep.point.loc["a", "ap"] <= rep.ci_high.loc["a", "ap"]
    assert rep.vs_reference.loc["b", "ap"].startswith("+0.000")
    name, _ = metrics.simplest_not_worse(rep, ["b", "a"], "ap")
    assert name == "a"                                # a is clearly better than the random b


def test_value_curve_net_is_value_minus_cost():
    y = np.array([1, 0, 1, 0])
    c = business.value_curve(np.array([0.9, 0.8, 0.1, 0.2]), np.zeros(4), np.array([0, 1, 2, 3]), y, np.ones(4),
                             [1, 4], business.Economics(cost_per_alert=10, value_per_scheme=100))
    assert c.net.tolist() == [100 - 10, 200 - 40]
