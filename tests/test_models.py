import pickle

import numpy as np
from sklearn.metrics import average_precision_score

from graphaml import pipeline
from graphaml.models import RuleScorer


def test_rule_score_is_number_of_flags_plus_tiebreak(exp):
    r = RuleScorer().fit(exp.feats.full)
    f = r.flags(exp.feats.full)
    s = r.predict(exp.feats.full)
    assert np.all(np.floor(s) == f.sum(axis=1).to_numpy())


def test_gnn_learns_and_survives_pickle(exp):
    models = pipeline.train_models(exp, which=("gnn",), gnn_kwargs={"epochs": 10, "hidden": 32, "patience": 10},
                                   verbose=False)
    gnn = models["gnn"]
    days = exp.days("valid")
    va = exp.rows("valid")
    s = gnn.predict(exp.snaps, exp.feats.full, days)
    assert np.isnan(s[exp.rows("train")]).all() and not np.isnan(s[va]).any()
    assert average_precision_score(exp.y[va], s[va]) > 2 * exp.y[va].mean()      # clearly better than random
    again = pickle.loads(pickle.dumps(gnn))
    np.testing.assert_allclose(again.predict(exp.snaps, exp.feats.full, days)[va], s[va], rtol=1e-5, atol=1e-6)
