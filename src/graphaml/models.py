"""Four ways to rank transactions for AML review, from what banks run today to a graph neural network.

    RuleScorer      red-flag rules on real-time information (what most monitoring systems are)
    LGBMScorer      gradient boosting on tabular features, or tabular + graph features
    EdgeGNN         directed GraphSAGE-style network on daily graph snapshots, scoring edges

The GNN is written in plain PyTorch (sparse matrix products, no PyTorch Geometric), so it runs
the same in Colab, in CI and in the Docker image without compiled extensions.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from .features import NODE_BASE, Snapshots

# --------------------------------------------------------------------------- rules


class RuleScorer:
    """Score = number of red flags raised; transaction amount breaks ties.

    Flags are typical AML scenarios. Thresholds come from training-period quantiles,
    never from labels, so this is the "untrained" baseline a bank would start from.
    """

    name = "rules"

    def fit(self, X: pd.DataFrame, y=None):
        self.big_amount = float(np.quantile(X["log_amount_usd"], 0.99))
        self.mid_amount = float(np.quantile(X["log_amount_usd"], 0.75))
        return self

    def flags(self, X: pd.DataFrame) -> pd.DataFrame:
        a = X["log_amount_usd"]
        return pd.DataFrame({
            "large_amount": a > self.big_amount,
            "fan_out_burst": X["s_out_n_24h"] >= np.log1p(5),
            "fan_in_burst": X["d_in_n_24h"] >= np.log1p(5),
            "pass_through": (X["s_in_n_24h"] > 0) & (X["s_passthrough_24h"].abs() < np.log(1.25)),
            "new_counterparty_large": (X["pair_n_all"] == 0) & (a > self.mid_amount),
            "money_returns": X["rev_pair_n_all"] > 0,
            "cross_currency": X["cross_currency"] > 0,
        })

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        f = self.flags(X)
        return f.sum(axis=1).to_numpy(float) + X["log_amount_usd"].to_numpy(float) / 100.0


# --------------------------------------------------------------------------- gradient boosting


class LGBMScorer:
    def __init__(self, name: str, columns: list[str], seed: int = 0, **params):
        self.name, self.columns, self.seed = name, list(columns), seed
        # at most 600 trees: scoring cost grows with the number of trees, and a daily batch of a
        # million transactions has to be scored on a CPU in minutes
        import os

        self.params = {"learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 50,
                       "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                       "reg_lambda": 1.0, "n_estimators": 600, "n_jobs": os.cpu_count() or 1,
                       "verbose": -1} | params

    def fit(self, X, y, X_val, y_val, early_stopping: int = 60, neg_rate: float | None = None):
        """Train with early stopping on validation average precision.

        Negative downsampling: with ~0.1% positives, all positives plus a random `neg_rate` share of the
        negatives carry nearly all the information. Kept negatives get weight 1/neg_rate, so the loss is
        an unbiased estimate of the full-data loss; validation (for early stopping only) is thinned the
        same way. Default: keep ~100 negatives per positive, never fewer than 10% of them.
        """
        import inspect

        import lightgbm as lgb

        y = np.asarray(y)
        y_val = np.asarray(y_val)
        pos = max(float(y.sum()), 1.0)
        if neg_rate is None:
            neg_rate = float(min(1.0, max(0.1, 100 * pos / max(len(y) - pos, 1))))
        rng = np.random.default_rng(self.seed)
        keep = (y == 1) | (rng.random(len(y)) < neg_rate)
        keep_v = (y_val == 1) | (rng.random(len(y_val)) < neg_rate)
        w = np.where(y[keep] == 1, 1.0, 1.0 / neg_rate)
        w_val = np.where(y_val[keep_v] == 1, 1.0, 1.0 / neg_rate)
        self.neg_rate = neg_rate

        self.model = lgb.LGBMClassifier(random_state=self.seed,
                                        scale_pos_weight=float(np.sqrt((len(y) - pos) / pos)), **self.params)
        Xv, yv = X_val[self.columns].iloc[np.flatnonzero(keep_v)], y_val[keep_v]
        # LightGBM >= 4.7 renamed eval_set to eval_X / eval_y; Colab may ship an older version
        if "eval_X" in inspect.signature(self.model.fit).parameters:
            evals = {"eval_X": (Xv,), "eval_y": (yv,), "eval_sample_weight": [w_val]}
        else:
            evals = {"eval_set": [(Xv, yv)], "eval_sample_weight": [w_val]}
        self.model.fit(X[self.columns].iloc[np.flatnonzero(keep)], y[keep], sample_weight=w,
                       eval_metric="average_precision", **evals,
                       callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.columns])[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row SHAP values (log-odds) from LightGBM's TreeSHAP; last column is the bias."""
        c = self.model.booster_.predict(X[self.columns], pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.columns, index=X.index)

    def importance(self) -> pd.Series:
        b = self.model.booster_
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)


# --------------------------------------------------------------------------- graph neural network

def _torch():
    import torch

    return torch


def build_gnn(node_in: int, edge_in: int, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
    torch = _torch()
    nn = torch.nn

    class DirSAGELayer(nn.Module):
        """h' = h + ReLU(LN(W_self h + W_in mean_{u->v} h_u + W_out mean_{v->w} h_w)).

        Separate weights for incoming and outgoing neighbours: direction is the whole point in AML
        (a fan-in collector and a fan-out distributor look identical in an undirected graph).
        """

        def __init__(self, d):
            super().__init__()
            self.self_lin = nn.Linear(d, d)
            self.in_lin = nn.Linear(d, d, bias=False)
            self.out_lin = nn.Linear(d, d, bias=False)
            self.norm = nn.LayerNorm(d)

        def forward(self, h, A_in, A_out):
            m = self.self_lin(h) + self.in_lin(torch.sparse.mm(A_in, h)) + self.out_lin(torch.sparse.mm(A_out, h))
            return h + torch.relu(self.norm(m))

    class EdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(node_in=node_in, edge_in=edge_in, hidden=hidden, layers=layers, dropout=dropout)
            self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.ReLU())
            self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU())
            self.layers = nn.ModuleList(DirSAGELayer(hidden) for _ in range(layers))
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, 1))

        def embed(self, x, A_in, A_out):
            h = self.node_enc(x)
            for layer in self.layers:
                h = self.drop(layer(h, A_in, A_out))
            return h

        def forward(self, x, A_in, A_out, src, dst, e):
            h = self.embed(x, A_in, A_out)
            return self.head(torch.cat([h[src], h[dst], self.edge_enc(e)], dim=1)).squeeze(-1)

    return EdgeGNN()


class GNNScorer:
    """Trains and applies the EdgeGNN one day at a time.

    For each day d the network sees the graph of all transactions up to the end of day d
    (exactly what a next-morning batch monitor has) and scores that day's transactions.
    Training walks over the training days in random order; validation days pick the epoch.
    """

    name = "gnn"

    def __init__(self, edge_columns: list[str], hidden: int = 64, layers: int = 3, dropout: float = 0.1,
                 lr: float = 2e-3, weight_decay: float = 1e-5, epochs: int = 40, patience: int = 6,
                 seed: int = 0, device: str | None = None):
        self.edge_columns = list(edge_columns)
        self.hp = dict(hidden=hidden, layers=layers, dropout=dropout)
        self.lr, self.wd, self.epochs, self.patience, self.seed = lr, weight_decay, epochs, patience, seed
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.history: list[dict] = []

    # ---- scaling (fit on training data only)

    def _fit_scalers(self, snaps: Snapshots, E: pd.DataFrame, train_days):
        last = snaps.prefix(max(train_days))
        base = snaps.node_base(last)
        active = (base[:, 0] + base[:, 1]) > 0
        self.node_mu, self.node_sd = base[active].mean(0), base[active].std(0) + 1e-6
        rows = np.flatnonzero(np.isin(snaps.day_of, train_days))
        e = E[self.edge_columns].to_numpy(np.float32)[rows].astype(np.float64)
        self.edge_mu, self.edge_sd = e.mean(0), e.std(0) + 1e-6

    # ---- one day's tensors

    def _day_inputs(self, snaps: Snapshots, E: np.ndarray, day: int):
        torch = _torch()
        dev = self.device
        m = snaps.prefix(day)
        x = (snaps.node_base(m) - self.node_mu) / self.node_sd
        s = torch.as_tensor(snaps.src[:m], device=dev)
        d = torch.as_tensor(snaps.dst[:m], device=dev)
        n = snaps.n

        def mean_adj(rows, cols):
            deg = torch.bincount(rows, minlength=n).clamp(min=1).float()
            vals = 1.0 / deg[rows]
            return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n), check_invariants=False).coalesce()

        A_in, A_out = mean_adj(d, s), mean_adj(s, d)    # row v: mean over senders u->v / over receivers v->w
        rows = snaps.rows_of(day)
        return (torch.as_tensor(x, dtype=torch.float32, device=dev), A_in, A_out,
                torch.as_tensor(snaps.src[rows], device=dev), torch.as_tensor(snaps.dst[rows], device=dev),
                torch.as_tensor(E[rows], dtype=torch.float32, device=dev), rows)

    def _edge_matrix(self, E: pd.DataFrame) -> np.ndarray:
        Em = E[self.edge_columns].to_numpy(np.float32, copy=True)
        Em -= self.edge_mu.astype(np.float32)
        Em /= self.edge_sd.astype(np.float32)
        return Em

    def fit(self, snaps: Snapshots, E: pd.DataFrame, y: np.ndarray, train_days, val_days, verbose: bool = True):
        torch = _torch()
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._fit_scalers(snaps, E, train_days)
        Em = self._edge_matrix(E)
        self.model = build_gnn(len(NODE_BASE), len(self.edge_columns), **self.hp).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        tr_rows = np.flatnonzero(np.isin(snaps.day_of, train_days))
        pos = max(float(y[tr_rows].sum()), 1.0)
        pos_weight = torch.tensor(np.sqrt((len(tr_rows) - pos) / pos), device=self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        cache = {d: self._day_inputs(snaps, Em, d) for d in list(train_days) + list(val_days)}   # graphs are fixed
        yt = torch.as_tensor(np.array(y, dtype=np.float32), device=self.device)

        best, best_state, bad = -1.0, None, 0
        val_rows = np.flatnonzero(np.isin(snaps.day_of, val_days))
        for epoch in range(self.epochs):
            t_start = time.time()
            self.model.train()
            losses = []
            for d in rng.permutation(train_days):
                x, A_in, A_out, s, dd, e, rows = cache[d]
                if len(rows) == 0:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model(x, A_in, A_out, s, dd, e), yt[rows])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            val_scores = self._predict(snaps, Em, val_days, cache)[val_rows]
            ap = average_precision_score(y[val_rows], val_scores) if y[val_rows].sum() > 0 else float("nan")
            self.history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_ap": ap,
                                 "seconds": round(time.time() - t_start, 1)})
            if verbose:
                print(f"epoch {epoch + 1:3d}  loss {np.mean(losses):.4f}  val AP {ap:.4f}")
            if ap > best:
                best, best_state, bad = ap, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_val_ap = best
        del cache
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return self

    def _predict(self, snaps: Snapshots, Em: np.ndarray, days, cache: dict | None = None) -> np.ndarray:
        torch = _torch()
        out = np.full(len(snaps.t), np.nan)
        self.model.eval()
        with torch.no_grad():
            for d in days:
                x, A_in, A_out, s, dd, e, rows = cache[d] if cache else self._day_inputs(snaps, Em, d)
                if len(rows):
                    out[rows] = torch.sigmoid(self.model(x, A_in, A_out, s, dd, e)).cpu().numpy()
        return out

    def predict(self, snaps: Snapshots, E: pd.DataFrame, days) -> np.ndarray:
        """Scores for the rows of `days` (NaN elsewhere)."""
        return self._predict(snaps, self._edge_matrix(E), days)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def __getstate__(self):   # pickle on CPU, rebuild the network class on load
        state = self.__dict__.copy()
        state["model_state"] = {k: v.cpu() for k, v in self.model.state_dict().items()}
        state["model_config"] = self.model.config
        state.pop("model")
        state["device"] = "cpu"
        return state

    def __setstate__(self, state):
        cfg, weights = state.pop("model_config"), state.pop("model_state")
        self.__dict__.update(state)
        self.model = build_gnn(**cfg)
        self.model.load_state_dict(weights)
        self.model.eval()


def rank_average(*scores: np.ndarray) -> np.ndarray:
    """Average of within-array percentile ranks: a scale-free ensemble of different models."""
    return np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in scores], axis=0)
