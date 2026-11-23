import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def std():
    """Small synthetic dataset in the standardized layout (same code path as IBM data)."""
    from graphaml import synthetic

    return synthetic.make_dataset(n_accounts=600, n_days=6, attempts_per_day=6, seed=3)


@pytest.fixture(scope="session")
def exp(std):
    from graphaml import pipeline

    return pipeline.build_experiment(std, verbose=False)
