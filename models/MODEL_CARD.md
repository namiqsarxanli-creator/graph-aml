# Model card: graph-based AML transaction ranking

**What it does.** Ranks one day's transactions by laundering risk so that analysts review the top `k` per day. It
does not block payments and does not decide anything on its own: every alert goes to a human investigator.

**Models in the bundle** (`aml_bundle.pkl`): red-flag rules, LightGBM on transaction + history features, LightGBM with
graph features, a directed GNN, and their rank-average ensemble. `model_meta.json` names the served one and why
(the simplest model not significantly worse than the best on validation).

**Training data.** See `trained_on` in `model_meta.json`. The committed bundle is trained on synthetic data so that the
repository runs without downloads; the IBM-trained bundle comes from the Colab notebook (step 11).

**Inputs.** Transactions in the IBM AML layout: timestamp, banks, accounts, amounts and currencies, payment format.
Account identifiers are only used to build the graph; they are never features. No personal attributes (name, age,
nationality, address) are used, and none exist in the data.

**Metrics.** `test_metrics` in `model_meta.json`, with 95% cluster-bootstrap intervals in `test_ci_low` /
`test_ci_high`. The test split was evaluated once, after the model choice.

**Known limits.**
- Trained on simulated data; a real deployment needs re-training and re-validation on the bank's own reviewed alerts.
- Laundering patterns change when launderers learn what is detected. Monitor `/drift` and alert precision weekly.
- The model learns from labelled schemes only; novel schemes unlike any training pattern may rank low. Keep the rules
  running alongside and review a random sample of non-alerted transactions.
- Accounts with no history (new customers) get less information from the graph; their scores rely on transaction
  features.

**Feedback loop risk.** If the model is retrained only on alerts it raised, it reinforces its own blind spots. Keep a
random review sample as unbiased labels.
