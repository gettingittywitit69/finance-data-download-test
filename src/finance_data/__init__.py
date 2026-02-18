"""Public API for finance_data utilities."""

from __future__ import annotations

# ---- Optional legacy helpers (may not exist / may be deprecated) ----
try:
    from .datasets import ensure_french_datasets, fetch_french25_excess, fetch_french49_excess
except Exception:
    ensure_french_datasets = None  # type: ignore[assignment]
    fetch_french25_excess = None  # type: ignore[assignment]
    fetch_french49_excess = None  # type: ignore[assignment]

# ---- Fama-French loaders (now direct Ken French ZIP download; no pandas_datareader) ----
from .french import (
    load_all_strategies_long,
    load_us_ff5_factors,
    load_us_industries_30,
    load_us_industries_49,
    load_us_momentum_factor,
    load_us_research_factors_wide,
    load_us_size_bm_25,
    load_us_size_deciles,
    pivot_family,
)

# ---- Models / simulation ----
from .ar_garch import (
    compute_path_moments,
    fit_ar_garch_t,
    run_parameter_uncertainty_experiment,
    sample_params_normal,
    simulate_ar_garch_t_paths,
)

# ---- Metrics ----
from .metrics import (
    bootstrap_psr,
    deflated_sharpe_ratio,
    effective_num_tests,
    expected_max_sharpe,
    min_track_record_length,
    observed_fdr,
    probabilistic_sharpe_ratio,
    sample_kurtosis,
    sample_skewness,
    sharpe_ratio,
    tangency_portfolio,
)

# ---- Pipeline ----
from .pipeline.zoo import (
    FactorModelSpec,
    FactorPanel,
    MonthlyPanel,
    TestConfig,
    alpha_sharpe,
    apply_flag_rules,
    build_factor_panel,
    combine_families,
    compute_family_stats,
    effective_trials,
    factor_alpha,
    global_screen,
    load_portfolio_family,
    run_dsr,
    run_ofdr,
    sensitivity_grid,
)

# ---- Reporting ----
from .spreads import compute_spread_stats, dsr_row, psr_mintrl_row, summarize_spreads, verdict
from .survival import compute_survival_map, plot_survival_map, rolling_windows

__all__ = [
    # pipeline API
    "FactorModelSpec",
    "FactorPanel",
    "MonthlyPanel",
    "TestConfig",
    "alpha_sharpe",
    "apply_flag_rules",
    "build_factor_panel",
    "combine_families",
    "compute_family_stats",
    "effective_trials",
    "factor_alpha",
    "global_screen",
    "load_portfolio_family",
    "run_dsr",
    "run_ofdr",
    "sensitivity_grid",
    # models/sim
    "compute_path_moments",
    "fit_ar_garch_t",
    "run_parameter_uncertainty_experiment",
    "sample_params_normal",
    "simulate_ar_garch_t_paths",
    # metrics
    "bootstrap_psr",
    "deflated_sharpe_ratio",
    "effective_num_tests",
    "expected_max_sharpe",
    "min_track_record_length",
    "observed_fdr",
    "probabilistic_sharpe_ratio",
    "sample_kurtosis",
    "sample_skewness",
    "sharpe_ratio",
    "tangency_portfolio",
    # FF loaders
    "load_all_strategies_long",
    "load_us_ff5_factors",
    "load_us_industries_30",
    "load_us_industries_49",
    "load_us_momentum_factor",
    "load_us_research_factors_wide",
    "load_us_size_bm_25",
    "load_us_size_deciles",
    "pivot_family",
    # spreads/survival
    "compute_spread_stats",
    "dsr_row",
    "psr_mintrl_row",
    "summarize_spreads",
    "verdict",
    "compute_survival_map",
    "plot_survival_map",
    "rolling_windows",
]

# expose legacy helpers only if available
if ensure_french_datasets is not None:
    __all__.append("ensure_french_datasets")
if fetch_french25_excess is not None:
    __all__.append("fetch_french25_excess")
if fetch_french49_excess is not None:
    __all__.append("fetch_french49_excess")
