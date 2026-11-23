"""Synthetic transactions in the IBM AML file format, with injected laundering schemes.

Used for tests, CI, the demo page and a quick end-to-end run in Colab. It is NOT a benchmark:
real results come from the IBM data (notebooks). The generator is deliberately honest:
launderers are ordinary accounts with ordinary background activity, laundering amounts overlap
with normal ones, and most schemes use common payment formats, so no single column gives them away.

Schemes (as in the IBM generator): FAN-OUT, FAN-IN, CYCLE, SCATTER-GATHER, GATHER-SCATTER,
STACK, BIPARTITE, RANDOM (a chain with no repeated account).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FORMATS, RAW_COLS, TS_FORMAT, USD_RATE

BACKGROUND_FORMAT_P = {"ACH": 0.12, "Bitcoin": 0.02, "Cash": 0.11, "Cheque": 0.33, "Credit Card": 0.27,
                       "Reinvestment": 0.08, "Wire": 0.07}
LAUNDERING_ACH_P = 0.55   # ACH is over-represented in laundering, but not exclusive
CURRENCY_P = {"US Dollar": 0.5, "Euro": 0.2, "Yuan": 0.05, "UK Pound": 0.05, "Rupee": 0.04, "Yen": 0.04,
              "Canadian Dollar": 0.03, "Swiss Franc": 0.02, "Australian Dollar": 0.02, "Ruble": 0.02,
              "Shekel": 0.01, "Mexican Peso": 0.01, "Saudi Riyal": 0.005, "Brazil Real": 0.005}
PATTERN_P = {"FAN-OUT": 0.17, "FAN-IN": 0.17, "CYCLE": 0.14, "SCATTER-GATHER": 0.12, "GATHER-SCATTER": 0.12,
             "STACK": 0.1, "BIPARTITE": 0.06, "RANDOM": 0.12}


class _Gen:
    def __init__(self, n_accounts, n_banks, seed):
        self.rng = rng = np.random.default_rng(seed)
        self.n = n_accounts
        self.bank = rng.integers(1, n_banks + 1, n_accounts)
        self.acct = np.array([format(0x80000000 + i * 104_729, "X") for i in range(n_accounts)])
        self.merchant = rng.random(n_accounts) < 0.03
        self.activity = rng.lognormal(0.2, 0.8, n_accounts) * np.where(self.merchant, 0.3, 1.0)
        self.scale = rng.lognormal(5.6, 1.1, n_accounts)
        cur = list(CURRENCY_P)
        self.currency = rng.choice(cur, n_accounts, p=np.array(list(CURRENCY_P.values())) / sum(CURRENCY_P.values()))
        base = np.array([BACKGROUND_FORMAT_P[f] for f in FORMATS])
        self.fmt_p = rng.dirichlet(base * 30, n_accounts)
        # regular counterparties: a few merchants plus a few people, chosen by popularity
        pop = self.activity * np.where(self.merchant, 25.0, 1.0)
        pop = pop / pop.sum()
        self.partners = rng.choice(n_accounts, size=(n_accounts, 6), p=pop)
        self.n_partners = rng.integers(2, 7, n_accounts)
        self.pop = pop
        self.rows: list[tuple] = []
        self.pattern_lines: list[str] = []

    def _amount_recv(self, amount_paid, pay_cur, recv_cur):
        return amount_paid * USD_RATE[pay_cur] / USD_RATE[recv_cur]

    def add(self, ts, src, dst, amount, fmt, laundering, store=True):
        rng = self.rng
        pay_cur = "Bitcoin" if fmt == "Bitcoin" else self.currency[src]
        if fmt == "Bitcoin":
            amount = amount / USD_RATE["Bitcoin"] * USD_RATE[self.currency[src]]
        # most transfers are received in the payer's currency; cross-border ones are converted
        recv_cur = pay_cur if (fmt == "Bitcoin" or rng.random() < 0.85) else self.currency[dst]
        amount = round(float(max(amount, 0.01)), 2 if fmt != "Bitcoin" else 6)
        recv = round(self._amount_recv(amount, pay_cur, recv_cur), 2 if recv_cur != "Bitcoin" else 6)
        row = (ts.strftime(TS_FORMAT), int(self.bank[src]), self.acct[src], int(self.bank[dst]), self.acct[dst],
               recv, recv_cur, amount, pay_cur, fmt, int(laundering))
        if store:
            self.rows.append(row)
        return row

    # ------------------------------------------------------------------ background

    def background(self, start, n_days):
        rng = self.rng
        for d in range(n_days):
            day0 = start + pd.Timedelta(days=d)
            counts = rng.poisson(self.activity)
            srcs = np.repeat(np.arange(self.n), counts)
            minutes = rng.integers(0, 24 * 60, len(srcs))
            for src, m in zip(srcs, minutes):
                fmt = FORMATS[rng.choice(len(FORMATS), p=self.fmt_p[src])]
                if fmt == "Reinvestment":
                    dst = src
                elif rng.random() < 0.75:
                    dst = self.partners[src, rng.integers(0, self.n_partners[src])]
                else:
                    dst = rng.choice(self.n, p=self.pop)
                if dst == src and fmt != "Reinvestment":
                    dst = (src + 1 + rng.integers(0, self.n - 1)) % self.n
                amount = self.scale[src] * rng.lognormal(0, 0.7)
                self.add(day0 + pd.Timedelta(minutes=int(m)), src, dst, amount, fmt, 0)

    # ------------------------------------------------------------------ laundering schemes

    def _fmt(self, src):
        if self.rng.random() < LAUNDERING_ACH_P:
            return "ACH"
        f = FORMATS[self.rng.choice(len(FORMATS), p=self.fmt_p[src])]
        return "ACH" if f == "Reinvestment" else f

    def scheme(self, name, t0):
        rng = self.rng
        txns = []   # (time, src, dst, amount)
        hop = lambda: pd.Timedelta(minutes=int(rng.exponential(240)) + 1)   # noqa: E731
        base = rng.lognormal(7.3, 0.9)

        def pick(k):
            # mules: ordinary, non-merchant accounts
            cand = np.flatnonzero(~self.merchant)
            return list(rng.choice(cand, size=k, replace=False))

        if name == "FAN-OUT":
            k = rng.integers(4, 13)
            a, *outs = pick(k + 1)
            t = t0
            for o in outs:
                t = t + hop() / 3
                txns.append((t, a, o, base / k * rng.uniform(0.8, 1.2)))
        elif name == "FAN-IN":
            k = rng.integers(4, 13)
            a, *ins = pick(k + 1)
            t = t0
            for i in ins:
                t = t + hop() / 3
                txns.append((t, i, a, base / k * rng.uniform(0.8, 1.2)))
        elif name == "CYCLE":
            k = rng.integers(3, 9)
            ring = pick(k)
            t, amt = t0, base
            for i in range(k):
                t = t + hop()
                amt *= rng.uniform(0.97, 0.995)
                txns.append((t, ring[i], ring[(i + 1) % k], amt))
        elif name in ("SCATTER-GATHER", "GATHER-SCATTER"):
            k = rng.integers(3, 8)
            nodes = pick(k + 2) if name == "SCATTER-GATHER" else pick(2 * k + 1)
            if name == "SCATTER-GATHER":
                a, b, *mid = nodes
                for m in mid:
                    t1 = t0 + hop()
                    amt = base / k * rng.uniform(0.85, 1.15)
                    txns.append((t1, a, m, amt))
                    txns.append((t1 + hop(), m, b, amt * rng.uniform(0.97, 0.995)))
            else:
                hub, rest = nodes[0], nodes[1:]
                ins, outs = rest[:k], rest[k:]
                t = t0
                for i in ins:
                    t = t + hop() / 3
                    txns.append((t, i, hub, base / k * rng.uniform(0.85, 1.15)))
                for o in outs:
                    t = t + hop() / 3
                    txns.append((t, hub, o, base / k * rng.uniform(0.85, 1.15)))
        elif name == "STACK":
            l1, l2, l3 = (rng.integers(2, 4) for _ in range(3))
            nodes = pick(l1 + l2 + l3)
            A, B, C = nodes[:l1], nodes[l1:l1 + l2], nodes[l1 + l2:]
            for x, y_layer, dt in ((A, B, 0), (B, C, 1)):
                for u in x:
                    for v in y_layer:
                        txns.append((t0 + hop() * (1 + 2 * dt), u, v, base / (l1 * l2) * rng.uniform(0.85, 1.15)))
        elif name == "BIPARTITE":
            l1, l2 = rng.integers(2, 5), rng.integers(2, 5)
            nodes = pick(l1 + l2)
            for u in nodes[:l1]:
                for v in nodes[l1:]:
                    if rng.random() < 0.7:
                        txns.append((t0 + hop(), u, v, base / (l1 * l2) * rng.uniform(0.85, 1.15)))
        elif name == "RANDOM":
            k = rng.integers(4, 10)
            chain = pick(k + 1)
            t, amt = t0, base
            for i in range(k):
                t = t + hop()
                amt *= rng.uniform(0.97, 0.995)
                txns.append((t, chain[i], chain[i + 1], amt))
        else:
            raise ValueError(name)

        detail = {"FAN-OUT": "Max {n}-degree Fan-Out", "FAN-IN": "Max {n}-degree Fan-In",
                  "CYCLE": "Max {n} hops"}.get(name, "Max {n} transactions").format(n=len(txns))
        self.pattern_lines.append(f"BEGIN LAUNDERING ATTEMPT - {name}:  {detail}")
        for t, s, d, a in sorted(txns, key=lambda x: x[0]):
            row = self.add(t, s, d, a, self._fmt(s), 1)
            self.pattern_lines.append(",".join(
                [row[0], f"{row[1]:03d}", row[2], f"{row[3]:03d}", row[4], f"{row[5]:.2f}", row[6],
                 f"{row[7]:.2f}", row[8], row[9], "1"]))
        self.pattern_lines.append(f"END LAUNDERING ATTEMPT - {name}")
        self.pattern_lines.append("")


def make_raw(n_accounts: int = 2500, n_days: int = 12, n_banks: int = 25, attempts_per_day: float = 6.0,
             seed: int = 0, start: str = "2022-09-01") -> tuple[pd.DataFrame, str]:
    """Return (transactions in IBM CSV layout, patterns text in IBM layout)."""
    g = _Gen(n_accounts, n_banks, seed)
    start_ts = pd.Timestamp(start)
    g.background(start_ts, n_days)
    names, p = list(PATTERN_P), np.array(list(PATTERN_P.values()))
    n_attempts = g.rng.poisson(attempts_per_day * n_days)
    for _ in range(n_attempts):
        # start early enough that most of the scheme finishes inside the generated period
        t0 = start_ts + pd.Timedelta(minutes=int(g.rng.integers(0, (n_days - 0.5) * 24 * 60)))
        g.scheme(names[g.rng.choice(len(names), p=p / p.sum())], t0)
    raw = pd.DataFrame(g.rows, columns=RAW_COLS)
    end = (start_ts + pd.Timedelta(days=n_days)).strftime(TS_FORMAT)
    raw = raw[raw["Timestamp"] < end]   # cut schemes that run past the period, as a real extract would
    raw = raw.assign(_t=pd.to_datetime(raw["Timestamp"], format=TS_FORMAT)).sort_values("_t", kind="stable")
    return raw.drop(columns="_t").reset_index(drop=True), "\n".join(g.pattern_lines)


def make_dataset(**kwargs) -> pd.DataFrame:
    """Synthetic data already standardized (see `data.standardize`)."""
    from .data import standardize

    raw, patterns = make_raw(**kwargs)
    return standardize(raw, patterns)
