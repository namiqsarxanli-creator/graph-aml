import numpy as np

from graphaml import data, synthetic


def test_synthetic_has_ibm_layout_and_every_laundering_row_has_a_scheme():
    raw, patterns = synthetic.make_raw(n_accounts=300, n_days=4, seed=1)
    assert list(raw.columns) == data.RAW_COLS
    std = data.standardize(raw, patterns)
    lau = std[std.is_laundering == 1]
    assert len(lau) > 0
    assert (lau.attempt_id >= 0).all() and (lau.pattern != "").all()
    assert (std.loc[std.is_laundering == 0, "attempt_id"] == -1).all()
    assert std.ts.is_monotonic_increasing


def test_parse_patterns_reads_ibm_format():
    text = "\n".join([
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops",
        "2022/09/01 00:10,011,800AA,022,800BB,100.00,US Dollar,100.00,US Dollar,ACH,1",
        "2022/09/01 01:10,022,800BB,011,800AA,99.00,US Dollar,99.00,US Dollar,ACH,1",
        "END LAUNDERING ATTEMPT - CYCLE",
        "",
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out",
        "2022/09/02 00:10,011,800AA,033,800CC,50.00,Euro,50.00,Euro,Cheque,1",
        "END LAUNDERING ATTEMPT - FAN-OUT",
    ])
    p = data.parse_patterns(text)
    assert list(p.pattern) == ["CYCLE", "CYCLE", "FAN-OUT"]
    assert list(p.attempt_id) == [0, 0, 1]
    assert p.from_bank.iloc[0] == 11          # zero padding removed, matches the CSV's integer banks


def test_time_split_keeps_days_whole_and_in_order(std):
    split = data.time_split(std)
    day = std.ts.dt.floor("D")
    assert split.groupby(day).nunique().max() == 1
    codes = split.map({"train": 0, "valid": 1, "test": 2}).to_numpy()
    assert np.all(np.diff(codes) >= 0)
    assert set(split) == {"train", "valid", "test"}


def test_amounts_converted_to_usd():
    raw, pat = synthetic.make_raw(n_accounts=200, n_days=3, seed=2)
    std = data.standardize(raw, pat)
    eur = std[std.pay_currency == "Euro"].iloc[0]
    assert np.isclose(eur.amount_usd, eur.amount_paid * data.USD_RATE["Euro"])


def test_simulation_tail_is_dropped():
    import pandas as pd

    days = pd.date_range("2022-09-01", periods=6, freq="D")
    n = [1000, 1000, 1000, 1000, 10, 5]
    df = pd.DataFrame({"ts": [d for d, k in zip(days, n) for _ in range(k)],
                       "is_laundering": [int(i >= 4) for i, k in enumerate(n) for _ in range(k)]})
    out, info = data.drop_simulation_tail(df)
    assert len(out) == 4000 and info["days_dropped"] == 2 and info["laundering_share_in_dropped"] == 1.0
    same, info2 = data.drop_simulation_tail(out)
    assert len(same) == 4000 and info2["days_dropped"] == 0
