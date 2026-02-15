import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import sharpe_mc  # noqa: E402


def test_required_reps_values():
    assert sharpe_mc.required_reps(alpha=0.05, epsilon=0.01, p=0.5) == 9604


def test_required_reps_bonferroni():
    assert sharpe_mc.required_reps(alpha=0.05, epsilon=0.01, p=0.5, m_cells=24) == 23687


def test_seed_reproducibility():
    rows1 = sharpe_mc.run_experiment(
        dgps=("iid_normal",),
        n_grid=(120,),
        s_true_grid=(0.5,),
        reps=50,
        seed=99,
        max_workers=1,
        run_sanity=False,
    )
    rows2 = sharpe_mc.run_experiment(
        dgps=("iid_normal",),
        n_grid=(120,),
        s_true_grid=(0.5,),
        reps=50,
        seed=99,
        max_workers=1,
        run_sanity=False,
    )
    assert rows1 == rows2


def test_run_experiment_shape():
    rows = sharpe_mc.run_experiment(reps=5, seed=7, max_workers=1, run_sanity=False)
    expected = len(sharpe_mc.DGP_LIST) * len(sharpe_mc.N_GRID) * len(sharpe_mc.S_TRUE_GRID) * 4
    assert len(rows) == expected


def test_dgp_sanity_smoke():
    n = 32
    for dgp in sharpe_mc.DGP_LIST:
        series = sharpe_mc.simulate_dgp(dgp, n=n, S_true=0.5, seed=123)
        assert len(series) == n
        assert np.isfinite(series).all()


def test_no_nan_summary():
    rows = sharpe_mc.run_cell("iid_normal", n=120, S_true=0.5, reps=200, seed=123, n_trials=10)
    keys = [
        "bias",
        "rmse",
        "coverage_95",
        "reject_rate_H0_S_le_0",
        "se_ratio",
        "psr_reject_rate",
        "dsr_reject_rate",
    ]
    for row in rows:
        for key in keys:
            assert np.isfinite(row[key])
