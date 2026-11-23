"""Loading IBM AML transaction files (or synthetic ones in the same format) into one standard table.

Raw format (IBM "Transactions for Anti Money Laundering", Altman et al. 2023, Kaggle):
    Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency,
    Amount Paid, Payment Currency, Payment Format, Is Laundering
The companion *_Patterns.txt file lists every laundering attempt (fan-out, cycle, ...) with its
transactions. Joining it lets us measure which *schemes* a model catches, not only which rows.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

RAW_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
            "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
TS_FORMAT = "%Y/%m/%d %H:%M"

FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

# Approximate value of one unit in US dollars (September 2022). Used only to put amounts paid in
# different currencies on one scale before summing them per account; not a modelling input on its own.
USD_RATE = {
    "US Dollar": 1.0, "Euro": 1.0, "UK Pound": 1.15, "Swiss Franc": 1.03, "Canadian Dollar": 0.75,
    "Australian Dollar": 0.67, "Yuan": 0.14, "Yen": 0.0070, "Rupee": 0.0123, "Ruble": 0.0165,
    "Shekel": 0.29, "Mexican Peso": 0.050, "Saudi Riyal": 0.266, "Brazil Real": 0.19, "Bitcoin": 19_500.0,
}
CURRENCIES = list(USD_RATE)

PATTERN_KEY = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_paid_r"]


def read_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read an IBM-format CSV. Account columns stay strings (they are hex ids)."""
    return pd.read_csv(path, names=RAW_COLS, header=0, nrows=nrows,
                       dtype={"Account": str, "Account.1": str, "From Bank": "int64", "To Bank": "int64"})


def parse_patterns(text: str) -> pd.DataFrame:
    """One row per transaction listed in a *_Patterns.txt file, with its attempt id and scheme name."""
    rows, attempt, pattern = [], -1, None
    for line in io.StringIO(text):
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            pattern = re.split(r"[:]", line.split(" - ", 1)[1])[0].strip().upper()
        elif line.startswith("END LAUNDERING ATTEMPT"):
            pattern = None
        elif line and pattern is not None:
            f = line.split(",")
            if len(f) < 11:
                continue
            rows.append((f[0], int(f[1]), f[2], int(f[3]), f[4], float(f[7]), attempt, pattern))
    out = pd.DataFrame(rows, columns=["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_paid",
                                      "attempt_id", "pattern"])
    out["ts"] = pd.to_datetime(out["ts"], format=TS_FORMAT)
    out["amount_paid_r"] = out.pop("amount_paid").round(2)
    return out


def standardize(raw: pd.DataFrame, patterns: str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw IBM columns -> one tidy, time-sorted table with account keys and (optionally) scheme labels."""
    df = pd.DataFrame({
        "ts": pd.to_datetime(raw["Timestamp"], format=TS_FORMAT),
        "from_bank": raw["From Bank"].astype("int64"),
        "from_acct": raw["Account"].astype(str),
        "to_bank": raw["To Bank"].astype("int64"),
        "to_acct": raw["Account.1"].astype(str),
        "amount_received": raw["Amount Received"].astype(float),
        "recv_currency": raw["Receiving Currency"].astype(str),
        "amount_paid": raw["Amount Paid"].astype(float),
        "pay_currency": raw["Payment Currency"].astype(str),
        "payment_format": raw["Payment Format"].astype(str),
        "is_laundering": raw["Is Laundering"].astype("int8") if "Is Laundering" in raw else np.int8(0),
    })
    df["src_key"] = df["from_bank"].astype(str) + ":" + df["from_acct"]
    df["dst_key"] = df["to_bank"].astype(str) + ":" + df["to_acct"]
    df["amount_usd"] = df["amount_paid"] * df["pay_currency"].map(USD_RATE).fillna(1.0)

    df["attempt_id"] = -1
    df["pattern"] = ""
    if patterns is not None:
        pat = parse_patterns(patterns) if isinstance(patterns, str) else patterns
        pat = pat.drop_duplicates(PATTERN_KEY)
        df["amount_paid_r"] = df["amount_paid"].round(2)
        df = df.drop(columns=["attempt_id", "pattern"]).merge(
            pat[PATTERN_KEY + ["attempt_id", "pattern"]], on=PATTERN_KEY, how="left")
        df["attempt_id"] = df["attempt_id"].fillna(-1).astype("int64")
        df["pattern"] = df["pattern"].fillna("")
        df = df.drop(columns="amount_paid_r")
        # a scheme label only makes sense on a laundering row
        df.loc[df["is_laundering"] == 0, ["attempt_id", "pattern"]] = (-1, "")

    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df), dtype=np.int64)
    return df


def load_ibm(trans_csv: str | Path, patterns_txt: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    patterns = Path(patterns_txt).read_text() if patterns_txt else None
    return standardize(read_raw(trans_csv, nrows=nrows), patterns)


def find_ibm_files(folder: str | Path, size: str = "HI-Small") -> tuple[Path, Path | None]:
    """Locate `<size>_Trans.csv` and `<size>_Patterns.txt` anywhere under `folder` (Kaggle layouts vary)."""
    folder = Path(folder)
    trans = next(iter(sorted(folder.rglob(f"{size}_Trans.csv"))), None)
    if trans is None:
        raise FileNotFoundError(f"{size}_Trans.csv not found under {folder}")
    pat = next(iter(sorted(folder.rglob(f"{size}_Patterns.txt"))), None)
    return trans, pat


def drop_simulation_tail(df: pd.DataFrame, min_share: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Remove the trailing days on which the simulator's normal activity has stopped.

    IBM files end with a week or so of almost empty days (HI-Small: 2022-09-11 to 09-18) that contain
    little more than the last hops of laundering schemes, so ~60% of their rows are laundering. No bank
    has such days; leaving them in the test split rewards any model that notices "quiet day = laundering".
    A trailing day is dropped when its volume is below `min_share` of the median day.
    """
    day = df["ts"].dt.floor("D")
    counts = day.value_counts().sort_index()
    normal = counts[counts >= min_share * counts.median()]
    last_normal = normal.index.max()
    tail = day > last_normal
    info = {"days_dropped": int(counts.index.to_series().gt(last_normal).sum()), "rows_dropped": int(tail.sum()),
            "laundering_dropped": int(df.loc[tail, "is_laundering"].sum()),
            "laundering_share_in_dropped": float(df.loc[tail, "is_laundering"].mean()) if tail.any() else 0.0,
            "last_day_kept": str(last_normal.date())}
    return df[~tail].reset_index(drop=True), info


def time_split(df: pd.DataFrame, shares: tuple[float, float] = (0.6, 0.2)) -> pd.Series:
    """Whole calendar days go to train / valid / test in time order (≈60/20/20 of transactions).

    A day is never split: models score transactions in daily batches, and the graph of a day
    must belong to one side of the cut. Returns the split name per row.
    """
    day = df["ts"].dt.floor("D")
    counts = day.value_counts().sort_index()
    mid = (counts.cumsum() - counts / 2) / counts.sum()
    name = pd.Series(np.where(mid < shares[0], "train", np.where(mid < shares[0] + shares[1], "valid", "test")),
                     index=counts.index)
    if name.nunique() < 3:   # very few days: force one day to each split from the end
        name.iloc[:] = "train"
        name.iloc[-2], name.iloc[-1] = "valid", "test"
    return day.map(name).rename("split")
