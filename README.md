# Following the money: graph machine learning for anti-money laundering

A bank sees hundreds of thousands of payments a day. About **0.1%** are money laundering, and the analysts can review a
few hundred alerts. Which payments should they look at?

A laundering payment looks ordinary on its own row. What gives it away is the network around it: money split across
ten accounts (fan-out), collected again (scatter-gather), or sent round in a circle back to where it started (cycle).
This project learns those shapes from the transaction graph and compares the result, honestly, with the red-flag rules
banks run today.

![CI](https://github.com/namiqsarxanli-creator/graph-aml/actions/workflows/ci.yml/badge.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/namiqsarxanli-creator/graph-aml/blob/main/notebooks/GraphAML_step_by_step.ipynb)

**Try it:** `docker compose up --build`, then open http://127.0.0.1:8090. You get a synthetic bank day: the alert queue
of the model next to the rules, and for each alert the money trail around it, with cycles and the reasons behind the
score.

**What this project shows**
- Graph ML on the IBM AML benchmark (5M transactions): a directed GraphSAGE-style GNN written in plain PyTorch, against
  red-flag rules, LightGBM on transaction history, and LightGBM with hand-built graph features.
- Fair comparisons by construction: three information levels (transaction, real-time history, next-morning graph)
  kept apart, and the GNN sees exactly what the strongest baseline sees.
- Leakage tests that delete the future and recompute every feature; a one-time test evaluation with cluster-bootstrap
  intervals (whole laundering schemes resampled together) and paired differences between models.
- Metrics an AML team uses: schemes detected at a fixed number of alerts per day, alerts needed to catch 50% / 80% of
  schemes, and an explicit cost/value scenario for choosing the daily alert budget.
- Production: FastAPI daily-batch scoring with API keys, alert reasons from TreeSHAP, a money-trail endpoint,
  drift monitoring, a training/serving skew test, Docker, CI, HTTPS deployment recipe.

## Results on IBM HI-Small

5.08M transactions, 515k accounts, 0.089% laundering. Train 09-01 to 09-06, validation 09-07 to 09-08, **test 09-09 to
09-10 (≈ 863k transactions, evaluated once)**. Alert budget: 2,412 per day (0.5% of a typical day). 95% cluster-bootstrap
intervals; whole laundering schemes are resampled together.

| Model | Average precision | Laundering transactions in alerts | Laundered USD in alerts |
|---|---|---|---|
| Red-flag rules (what banks run today) | 0.002 [0.002, 0.003] | 2.0% [1.0, 3.0] | 17% |
| LightGBM, transaction + account history | 0.063 [0.051, 0.078] | 54.3% [50.8, 58.0] | 25% |
| LightGBM + graph features | 0.114 [0.098, 0.140] | 57.3% [54.3, 61.3] | 62% |
| Graph neural network | 0.150 [0.123, 0.188] | 47.1% [43.5, 51.3] | 76% |
| **GNN + LightGBM-graph (served)** | **0.194 [0.165, 0.240]** | **59.5% [56.6, 63.1]** | 62% |

- **Rules barely beat random.** Their alerts contain 2% of the laundering; the same budget with the served model
  contains 60%. Average precision is ~90 times higher.
- **The graph roughly doubles performance.** Adding graph features to the same LightGBM almost doubles average precision
  (0.063 → 0.114, intervals do not overlap). The GNN alone reaches 0.150, and combining it with
  graph-feature LightGBM gives 0.194. On validation the ensemble beat every simpler model beyond noise, so it is served.
- **The GNN finds the big money.** Its alerts hold 76% of laundered dollars while containing fewer laundering rows, so
  it puts the high-value transactions first (the dollar intervals are wide: a few large schemes dominate the amount).
- **Known schemes vs the rest.** Of the schemes listed in IBM's pattern file, the served model alerts on 82–100% of
  every type (cycles 82%, scatter-gather 100%, fan-out 90%). Laundering rows that IBM does not attach to a listed
  scheme are much harder (29%). Full table: [`reports/results_ibm.md`](reports/results_ibm.md).
- **A data artefact was removed first.** The last 8 days of the IBM file (09-11 to 09-18) contain almost only the tail
  ends of laundering schemes: 59% of their rows are laundering. Left in the test split, they would reward any model that
  learns "quiet day = laundering". `data.drop_simulation_tail` drops them, and the notebook shows why.

The IBM-trained bundle is in `models/ibm/` (served model: ensemble). The default bundle in `models/` is trained on
synthetic data (`python scripts/train.py --data synthetic`), so that CI, tests and the synthetic demo run without a
Kaggle account. To serve the IBM model, copy `models/ibm/*` over `models/`.

## How it works

**Data.** [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Altman et al., NeurIPS 2023): a multi-agent simulation of banks, people and companies, with laundering injected as
known schemes. Every laundering transaction is linked to its scheme (`*_Patterns.txt`), which real bank data almost
never offers. That is what makes "which schemes does each model catch?" answerable.

**Split.** Whole calendar days, in time order: ~60% train, ~20% validation (all choices), ~20% test (looked at once).

**Three information levels.** Each model sees one of them:

| Level | What it knows | Used by |
|---|---|---|
| Transaction | amount (USD-normalised), format, currency, conversion, hour | all models |
| Real-time history | both accounts' activity strictly before this minute: counts and sums, 24h bursts, first payment between the pair, money received and passed on within a day | rules, LightGBM |
| Daily graph | the graph up to the end of the transaction's day, as an analyst sees it the next morning: degrees, distinct counterparties, share of money forwarded, neighbours' profiles (2 hops) | LightGBM + graph, GNN |

**GNN.** Each account starts with ten graph statistics. Each of three layers mixes in, separately, the accounts it pays
and the accounts that pay it (direction matters: a collector and a distributor are different roles). A transaction is
scored from the sender's and receiver's embeddings plus its own features. One graph snapshot per day; no PyTorch
Geometric, so the same code runs in Colab, CI and the CPU-only Docker image.

**Choosing a model.** On validation, the simplest model whose gap to the best one is within noise (paired 95%
bootstrap interval includes zero) is served. A complex model has to earn its place.

**Evaluation.** Average precision; precision, recall and laundered amount at `k` alerts per day; **schemes@k**, the
share of laundering schemes with at least one alerted transaction (one alert is enough for an investigator to follow
the money); minority-class F1 at a validation-chosen threshold (the metric of the IBM paper). Intervals come from a
cluster bootstrap over schemes.

## Run it

| Where | How |
|---|---|
| Colab (recommended, GPU) | open `notebooks/GraphAML_step_by_step.ipynb`, add `KAGGLE_USERNAME` / `KAGGLE_KEY` as Colab secrets, run all |
| Local, synthetic | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt`, then `python scripts/train.py` and `pytest -q` |
| Local, IBM | put `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/`, then `python scripts/train.py --data ibm` |
| API + demo | `docker compose up --build` → http://127.0.0.1:8090 (docs at `/docs`) |

The notebook is written in Azerbaijani, step by step; `scripts/make_notebook.py` generates it.

## API

| Endpoint | |
|---|---|
| `GET /health` | served model, training data, test metrics |
| `POST /alerts` | CSV in IBM layout (history + day to score) → today's alert queue, highest risk first, with reasons and rule flags |
| `POST /score/batch` | same input → CSV with a risk score for every scored transaction |
| `POST /drift` | PSI of every feature against the training period |
| `GET /demo/case`, `GET /demo/graph` | synthetic bank day and the money trail around one transaction (demo page) |

Set `AML_API_KEY` to require the `X-API-Key` header. Deployment: [`deploy/README.md`](deploy/README.md).

## Project layout

```
src/graphaml/    data.py (IBM loading, schemes, split) · synthetic.py · features.py · models.py (rules, LightGBM, GNN)
                 pipeline.py · metrics.py · business.py · explain.py · serving.py · api.py · monitoring.py · plots.py
notebooks/       GraphAML_step_by_step.ipynb (Colab walkthrough)
scripts/         train.py · make_notebook.py
tests/           leakage, metrics, models, explanations, API and training/serving skew
models/          aml_bundle.pkl, model_meta.json, MODEL_CARD.md
```

## Limits

- IBM data is simulated. It is the field's standard benchmark because real labelled AML data is not public, but
  absolute numbers will not carry over to a real bank; the comparison between approaches is what carries over.
- Graph models use same-day information (next-morning batch), rules and tabular models only the past. This is how the
  two kinds of systems run in practice, and it is stated wherever results are compared.
- Mean-aggregation GNNs cannot count cycles exactly. Port numbering, ego IDs and reverse message passing
  (Egressy et al., AAAI 2024) are the next step for cycle-heavy schemes.
- Currency conversion uses fixed approximate rates; the economics are scenarios, not measured costs.
- The synthetic generator (tests, demo) is simpler than IBM's simulator.

## References

- Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu. *Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models.* NeurIPS Datasets and Benchmarks, 2023.
- Egressy, von Niederhäusern, Blanuša, Altman, Wattenhofer, Atasu. *Provably Powerful Graph Neural Networks for Directed
  Multigraphs.* AAAI, 2024.
- Hamilton, Ying, Leskovec. *Inductive Representation Learning on Large Graphs* (GraphSAGE). NeurIPS, 2017.
