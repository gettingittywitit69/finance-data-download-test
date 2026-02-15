"""
Pipeline helpers for running DSR/oFDR screens across Fama-French strategy families.

The module keeps data handling light: it expects monthly return panels in decimal
form and builds tidy outputs that can be sliced in notebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import metrics, spreads
from ..french import load_us_ff5_factors, load_us_momentum_factor, load_us_research_factors_wide

__all__ = [
    "FactorModelSpec",
    "TestConfig",
    "MonthlyPanel",
    "FactorPanel",
    "build_factor_panel",
    "load_portfolio_family",
    "align_excess_returns",
    "compute_family_stats",
    "effective_trials",
    "run_dsr",
    "run_ofdr",
    "factor_alpha",
    "alpha_sharpe",
    "apply_flag_rules",
    "sensitivity_grid",
    "combine_families",
    "global_screen",
]


@dataclass
class FactorModelSpec:
    """Simple factor model spec used for alpha regressions."""

    name: str
    factors: Sequence[str]


@dataclass
class TestConfig:
    """Configuration for DSR/oFDR sweeps."""

    alpha_grid: Sequence[float] = (0.05,)
    prior_grid: Sequence[float] = (0.10,)
    flag_rules: Sequence[str] = ("union",)
    m_eff_mode: str = "family"
    custom_m_eff: Optional[float] = None
    min_obs: int = 24
    periods_per_year: int = 12
    ofdr_q: float = 0.20
    sr_benchmark: float = 0.0
    sr_alt: Optional[float] = None


@dataclass
class MonthlyPanel:
    """Monthly return panel with optional metadata."""

    data: pd.DataFrame
    name: str
    tier: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class FactorPanel:
    """Factors panel plus risk-free series."""

    data: pd.DataFrame
    rf: Optional[pd.Series] = None
    meta: Dict[str, object] = field(default_factory=dict)


def _coverage_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Lightweight coverage stats for a panel."""
    trimmed = df.dropna(how="all")
    if trimmed.empty:
        return {
            "start": None,
            "end": None,
            "n_obs": 0,
            "n_assets": 0,
            "missing_share": np.nan,
        }
    miss = float(trimmed.isna().mean().mean())
    return {
        "start": trimmed.index.min(),
        "end": trimmed.index.max(),
        "n_obs": trimmed.shape[0],
        "n_assets": trimmed.shape[1],
        "missing_share": miss,
    }


def _pivot_long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy long (date/strategy_id/return_excess) to wide."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot(index="date", columns="strategy_id", values="return_excess")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide.dropna(how="all")


def build_factor_panel(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_momentum: bool = True,
) -> FactorPanel:
    """
    Fetch mandatory factor series and assemble a wide panel with RF.

    Returns a FactorPanel with columns like Mkt_RF, SMB, HML, RMW, CMA, UMD when
    available. Missing optional series (e.g., momentum) are silently skipped.
    """
    ff5_long, rf_ff5 = load_us_ff5_factors(start_date=start_date, end_date=end_date)
    ff5_wide = _pivot_long_to_wide(ff5_long)

    frames = []
    if not ff5_wide.empty:
        frames.append(ff5_wide)

    if include_momentum:
        mom_long = load_us_momentum_factor(start_date=start_date, end_date=end_date)
        mom_wide = _pivot_long_to_wide(mom_long)
        if not mom_wide.empty:
            frames.append(mom_wide)

    ff3_wide, rf_alt = load_us_research_factors_wide(start_date=start_date, end_date=end_date)
    ff3_wide = ff3_wide.rename(columns={"Mkt-RF": "Mkt_RF"})
    if not ff3_wide.empty:
        frames.append(ff3_wide)

    factors = pd.concat(frames, axis=1, join="outer") if frames else pd.DataFrame()
    factors = factors.loc[~factors.index.duplicated()].sort_index()
    rf_used = rf_ff5 if rf_ff5 is not None else rf_alt
    rf_used = rf_used.reindex(factors.index) if rf_used is not None and not factors.empty else rf_used

    meta = _coverage_summary(factors)
    meta["rf_source"] = "FF5" if rf_ff5 is not None else "unknown"
    meta["include_momentum"] = include_momentum
    return FactorPanel(data=factors, rf=rf_used, meta=meta)


def load_portfolio_family(
    name: str,
    tier: str,
    fetcher: Callable[[], pd.DataFrame],
) -> MonthlyPanel:
    """
    Load a portfolio family using a provided fetcher callable.

    The fetcher must return a wide monthly DataFrame indexed by date with portfolio
    returns in decimal form (already excess or raw).
    """
    df = fetcher()
    if df is None or df.empty:
        raise ValueError(f"Fetcher for {name} returned no data")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    meta = _coverage_summary(df)
    meta.update({"family": name, "tier": tier})
    return MonthlyPanel(data=df, name=name, tier=tier, meta=meta)


def align_excess_returns(
    panel: MonthlyPanel,
    factors: FactorPanel,
    *,
    subtract_rf: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Align a family panel to the factor dates and, optionally, subtract RF.

    Returns the aligned excess-return DataFrame and a coverage dictionary.
    """
    data = panel.data.copy()
    data = data.loc[~data.index.duplicated()]
    aligned = data
    if subtract_rf and factors.rf is not None:
        rf_aligned = factors.rf.reindex(aligned.index)
        aligned = aligned.sub(rf_aligned, axis=0)
    if factors.data is not None and not factors.data.empty:
        aligned = aligned.reindex(factors.data.index)
    aligned = aligned.dropna(how="all")
    coverage = _coverage_summary(aligned)
    coverage["family"] = panel.name
    coverage["tier"] = panel.tier
    return aligned, coverage


def compute_family_stats(
    excess_df: pd.DataFrame,
    *,
    min_obs: int = 24,
) -> pd.DataFrame:
    """Compute Sharpe stats for a family after dropping under-sampled columns."""
    if excess_df is None or excess_df.empty:
        return pd.DataFrame()
    cols = []
    for col in excess_df.columns:
        if excess_df[col].dropna().shape[0] >= min_obs:
            cols.append(col)
    if not cols:
        return pd.DataFrame()
    filtered = excess_df[cols]
    return spreads.summarize_spreads(filtered)


def effective_trials(excess_df: pd.DataFrame) -> float:
    """Wrapper around metrics.effective_num_tests with sensible fallback."""
    m_eff = metrics.effective_num_tests(excess_df)
    if not np.isfinite(m_eff) or m_eff <= 1.0:
        return float(excess_df.shape[1])
    return float(m_eff)


def run_dsr(
    stats: pd.DataFrame,
    m_eff: float,
    alpha_grid: Sequence[float],
) -> pd.DataFrame:
    """Compute DSR for each alpha in alpha_grid; returns multi-index (alpha, portfolio)."""
    if stats is None or stats.empty:
        return pd.DataFrame(columns=["dsr", "dsr_pass", "sr0_dsr", "sigma_sr0"])
    m_eff_used = float(m_eff)
    dsr_null = spreads._dsr_null_from_spreads(stats["sr_rel"], m_eff_used)
    frames = []
    for alpha in alpha_grid:
        res = stats.apply(
            lambda row: spreads.dsr_row(row, sr0_dsr=dsr_null["sr0_dsr"], alpha=float(alpha)),
            axis=1,
        )
        res["alpha"] = float(alpha)
        frames.append(res)
    out = pd.concat(frames, axis=0, ignore_index=False)
    out = out.set_index(["alpha"], append=True)
    out.index = out.index.set_names(["portfolio", "alpha"])
    out = out.swaplevel(0, 1).sort_index()
    return out


def run_ofdr(
    excess_df: pd.DataFrame,
    m_eff: float,
    prior_grid: Sequence[float],
    *,
    sr_benchmark: float = 0.0,
    sr_alt: Optional[float] = None,
) -> pd.DataFrame:
    """Compute oFDR for each prior in prior_grid; returns multi-index (prior, portfolio)."""
    if excess_df is None or excess_df.empty:
        return pd.DataFrame(columns=["ofdr", "p_value", "sr0", "sr1", "sr_hat"])
    frames = []
    for prior in prior_grid:
        rows = []
        for name, series in excess_df.items():
            res = metrics.observed_fdr(
                series,
                sr_benchmark=sr_benchmark,
                sr_alt=sr_alt,
                prior_h1=float(prior),
                m_eff=m_eff,
            )
            res["portfolio"] = name
            res["prior"] = float(prior)
            rows.append(res)
        frame = pd.DataFrame(rows).set_index("portfolio")
        frame["prior"] = float(prior)
        frames.append(frame)
    out = pd.concat(frames, axis=0, ignore_index=False)
    out = out.set_index(["prior"], append=True)
    out.index = out.index.set_names(["portfolio", "prior"])
    out = out.swaplevel(0, 1).sort_index()
    return out


def _ols_alpha(y: pd.Series, X: pd.DataFrame):
    """Plain OLS helper returning alpha, t-stat, and n_obs."""
    y_clean = y.dropna()
    df = pd.concat([y_clean, X], axis=1, join="inner").dropna()
    n_obs = df.shape[0]
    if n_obs == 0:
        return np.nan, np.nan, 0
    y_vec = df.iloc[:, 0].values
    X_mat = df.iloc[:, 1:].values
    X_design = np.column_stack([np.ones(n_obs), X_mat])
    beta, _, rank, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
    if rank < X_design.shape[1]:
        return np.nan, np.nan, n_obs
    resid = y_vec - X_design @ beta
    dof = n_obs - X_design.shape[1]
    if dof <= 0:
        return np.nan, np.nan, n_obs
    sigma2 = float(resid @ resid) / dof
    xtx = X_design.T @ X_design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, n_obs
    se_alpha = math.sqrt(sigma2 * xtx_inv[0, 0]) if xtx_inv[0, 0] > 0 else np.nan
    alpha_hat = float(beta[0])
    t_alpha = alpha_hat / se_alpha if np.isfinite(se_alpha) and se_alpha > 0 else np.nan
    return alpha_hat, t_alpha, n_obs


def factor_alpha(
    excess_df: pd.DataFrame,
    factors: pd.DataFrame,
    model: FactorModelSpec,
    *,
    min_obs: int = 24,
) -> pd.DataFrame:
    """
    Regress excess returns on the given factor model; returns alphas/t-stats.

    Output columns: alpha_monthly, alpha_ann, t_alpha, n_obs, model.
    """
    if excess_df is None or excess_df.empty or factors is None or factors.empty:
        return pd.DataFrame()
    if not model.factors:
        raise ValueError("FactorModelSpec must include at least one factor")
    missing = [f for f in model.factors if f not in factors.columns]
    if missing:
        raise KeyError(f"Missing factors for model {model.name}: {missing}")
    factors_used = factors[model.factors].copy()
    records = []
    for name, series in excess_df.items():
        alpha_hat, t_alpha, n_obs = _ols_alpha(series, factors_used)
        if n_obs < min_obs:
            alpha_hat = np.nan
            t_alpha = np.nan
        records.append(
            {
                "portfolio": name,
                "model": model.name,
                "alpha_monthly": alpha_hat,
                "alpha_ann": alpha_hat * 12.0 if np.isfinite(alpha_hat) else np.nan,
                "t_alpha": t_alpha,
                "n_obs": n_obs,
            }
        )
    return pd.DataFrame(records).set_index("portfolio")


def alpha_sharpe(alpha_series: pd.Series) -> float:
    """Sharpe ratio of an alpha time series."""
    if alpha_series is None or alpha_series.empty:
        return np.nan
    return metrics.sharpe_ratio(alpha_series)


def apply_flag_rules(
    dsr_df: pd.DataFrame,
    ofdr_df: pd.DataFrame,
    *,
    q: float = 0.20,
    rules: Sequence[str] = ("union",),
) -> pd.DataFrame:
    """Apply flagging rules to DSR/oFDR grids; returns a boolean DataFrame."""
    portfolios = set()
    if dsr_df is not None and not dsr_df.empty:
        portfolios.update(dsr_df.index.get_level_values("portfolio"))
    if ofdr_df is not None and not ofdr_df.empty:
        portfolios.update(ofdr_df.index.get_level_values("portfolio"))
    portfolios = sorted(portfolios)
    out = pd.DataFrame(index=portfolios)

    dsr_any = (
        dsr_df.groupby("portfolio")["dsr_pass"].any() if dsr_df is not None and not dsr_df.empty else pd.Series(False, index=portfolios)
    )
    dsr_all = (
        dsr_df.groupby("portfolio")["dsr_pass"].all() if dsr_df is not None and not dsr_df.empty else pd.Series(False, index=portfolios)
    )
    ofdr_min = (
        ofdr_df.groupby("portfolio")["ofdr"].min() if ofdr_df is not None and not ofdr_df.empty else pd.Series(np.inf, index=portfolios)
    )
    ofdr_any = ofdr_min <= q
    ofdr_all = (
        ofdr_df.groupby("portfolio")["ofdr"].max() <= q if ofdr_df is not None and not ofdr_df.empty else pd.Series(False, index=portfolios)
    )

    for rule in rules:
        rule_lower = rule.lower()
        if rule_lower == "dsr_only":
            out[rule] = dsr_any.reindex(portfolios, fill_value=False)
        elif rule_lower == "ofdr_only":
            out[rule] = ofdr_any.reindex(portfolios, fill_value=False)
        elif rule_lower == "union":
            out[rule] = (dsr_any | ofdr_any).reindex(portfolios, fill_value=False)
        elif rule_lower == "intersection":
            out[rule] = (dsr_any & ofdr_any).reindex(portfolios, fill_value=False)
        elif rule_lower == "alpha_sweep":
            out[rule] = (dsr_all | ofdr_all).reindex(portfolios, fill_value=False)
        else:
            raise ValueError(f"Unknown flag rule: {rule}")
    return out


def sensitivity_grid(
    dsr_df: pd.DataFrame,
    ofdr_df: pd.DataFrame,
    *,
    alphas: Sequence[float],
    priors: Sequence[float],
    q: float = 0.20,
) -> Dict[str, pd.DataFrame]:
    """Build sensitivity summaries across alpha/prior grids."""
    records = []
    freq: Dict[str, int] = {}
    for alpha in alphas:
        dsr_slice = dsr_df.xs(alpha, level="alpha")["dsr_pass"] if dsr_df is not None and not dsr_df.empty else pd.Series(dtype=bool)
        for prior in priors:
            ofdr_slice = (
                ofdr_df.xs(prior, level="prior")["ofdr"] if ofdr_df is not None and not ofdr_df.empty else pd.Series(dtype=float)
            )
            low_ofdr = ofdr_slice <= q
            union_names = set(dsr_slice[dsr_slice].index) | set(low_ofdr[low_ofdr].index)
            records.append(
                {
                    "alpha": float(alpha),
                    "prior_h1": float(prior),
                    "dsr_pass_cnt": int(dsr_slice.sum()) if not dsr_slice.empty else 0,
                    "low_ofdr_cnt": int(low_ofdr.sum()) if not low_ofdr.empty else 0,
                    "flagged_total": len(union_names),
                    "top_flags": ", ".join(sorted(union_names)[:3]),
                }
            )
            for name in union_names:
                freq[name] = freq.get(name, 0) + 1
    summary = pd.DataFrame(records).sort_values(["alpha", "prior_h1"])
    flag_freq = pd.DataFrame({"hits": pd.Series(freq)}).sort_values("hits", ascending=False)
    return {"summary": summary, "frequency": flag_freq}


def combine_families(families: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack families into a single panel with MultiIndex columns (family, portfolio)."""
    frames = []
    for family, df in families.items():
        fam_df = df.copy()
        fam_df.columns = pd.MultiIndex.from_product([[family], fam_df.columns])
        frames.append(fam_df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.loc[~combined.index.duplicated()]
    return combined


def _screen_family(
    excess_df: pd.DataFrame,
    config: TestConfig,
    *,
    m_eff_override: Optional[float] = None,
) -> Dict[str, object]:
    """Run the full DSR/oFDR pipeline for one family."""
    stats = compute_family_stats(excess_df, min_obs=config.min_obs)
    if stats.empty:
        return {"stats": stats, "dsr": pd.DataFrame(), "ofdr": pd.DataFrame(), "flags": pd.DataFrame(), "m_eff": np.nan}
    m_eff_used = (
        float(m_eff_override)
        if m_eff_override is not None and np.isfinite(m_eff_override)
        else effective_trials(excess_df[stats.index])
    )
    dsr_df = run_dsr(stats, m_eff_used, config.alpha_grid)
    ofdr_df = run_ofdr(
        excess_df[stats.index],
        m_eff_used,
        config.prior_grid,
        sr_benchmark=config.sr_benchmark,
        sr_alt=config.sr_alt,
    )
    flags = apply_flag_rules(dsr_df, ofdr_df, q=config.ofdr_q, rules=config.flag_rules)
    return {"stats": stats, "dsr": dsr_df, "ofdr": ofdr_df, "flags": flags, "m_eff": m_eff_used}


def global_screen(
    families: Mapping[str, pd.DataFrame],
    config: TestConfig,
) -> Dict[str, object]:
    """
    Run family-level and global screens across a mapping of {family: excess_df}.

    Returns a dict with per-family outputs and a global combined screen.
    """
    family_results: Dict[str, object] = {}
    for name, df in families.items():
        m_eff_override = None
        if config.m_eff_mode == "global" and config.custom_m_eff is not None:
            m_eff_override = config.custom_m_eff
        elif config.m_eff_mode == "custom":
            m_eff_override = config.custom_m_eff
        family_results[name] = _screen_family(df, config, m_eff_override=m_eff_override)

    combined = combine_families(families)
    if combined.empty:
        global_out = {"stats": pd.DataFrame(), "dsr": pd.DataFrame(), "ofdr": pd.DataFrame(), "flags": pd.DataFrame(), "m_eff": np.nan}
    else:
        flat_cols = [f"{fam}::{col}" for fam, col in combined.columns]
        combined.columns = flat_cols
        m_eff_global = (
            float(config.custom_m_eff)
            if config.m_eff_mode == "custom" and config.custom_m_eff is not None
            else effective_trials(combined)
        )
        global_out = _screen_family(combined, config, m_eff_override=m_eff_global)
    return {"families": family_results, "global": global_out}
