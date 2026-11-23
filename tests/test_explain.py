import numpy as np
import pandas as pd

from graphaml import explain


def test_cycle_through_account_is_found():
    edges = pd.DataFrame({"src": [1, 2, 3, 1, 5], "dst": [2, 3, 1, 4, 1], "t": [0, 1, 2, 3, 4]})
    assert explain.cycles_through(edges, 1) == [[1, 2, 3]]
    assert explain.cycles_through(edges, 4) == []


def test_ego_network_respects_hops_and_time():
    df = pd.DataFrame({"src": [1, 2, 3, 9], "dst": [2, 3, 4, 1], "t": [0, 10, 20, 30]})
    assert set(explain.ego_network(df, np.zeros(4), 1, hops=1).index) == {0, 3}
    # row 3 is in the future, row 2 is three steps away
    assert set(explain.ego_network(df, np.zeros(4), 1, hops=2, upto_t=25).index) == {0, 1}


def test_feature_labels_are_readable():
    assert explain.label("gd_in_nb_flow_through") == "receiver's payers forward money on"
    assert explain.label("gs_log_uniq_out") == "sender pays many different accounts"
    assert "_" not in explain.label("gs_out_nb_log_in_deg")
