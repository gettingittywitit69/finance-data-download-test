"""
Utilities for evaluating portfolio spreads versus a benchmark (e.g., FF25 vs. market).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import metrics


def summarize_spreads(spread_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sample moments for each spread column.

    Returns a DataFrame sorted by relative Sharpe ratio (sr_rel).
    """
    if spread_df is None or spread_df.empty:
        raise ValueError("spread_df must be a non-empty DataFrame")

    rows = []
    for col in spread_df.columns:
        series = spread_df[col].dropna()
        rows.append(
            {
                "portfolio": col,
                "n_obs": series.shape[0],
                "mu_d": series.mean(),
                "sigma_d": series.std(ddof=0),
                "sr_rel": metrics.sharpe_ratio(series),
                "skew": metrics.sample_skewness(series),
                "kurt": metrics.sample_kurtosis(series),
            }
        )
    df = pd.DataFrame(rows).set_index("portfolio")
    return df.sort_values("sr_rel", ascending=False)


def psr_mintrl_row(
    row: pd.Series,
    *,
    alpha: float = 0.05,
    periods_per_year: int = 12,
) -> pd.Series:
    """
    Probabilistic Sharpe Ratio summary for a single spread row.

    Parameters
    ----------
    row : Series
        Row containing sr_rel and n_obs.
    alpha : float
        One-sided significance level (1 - confidence).
    periods_per_year : int
        Observations per year, used to convert MinTRL into years.
    """
    sr_hat = row.get("sr_rel")
    n_obs = row.get("n_obs")
    if n_obs is None:
        n_obs = np.nan

    if n_obs < 2 or not np.isfinite(sr_hat):
        return pd.Series(
            {
                "psr_z": np.nan,
                "psr": np.nan,
                "psr_pass": False,
                "min_trl_obs": np.nan,
                "min_trl_years": np.nan,
                "enough_history": False,
            }
        )

    z_alpha = norm.ppf(1.0 - alpha)
    z_stat = sr_hat * np.sqrt(n_obs)  # SR0 = 0 -> sigma_SR = 1/sqrt(T)
    psr = norm.cdf(z_stat)
    min_trl_obs = np.inf
    if sr_hat > 0:
        min_trl_obs = (z_alpha / sr_hat) ** 2

    return pd.Series(
        {
            "psr_z": z_stat,
            "psr": psr,
            "psr_pass": psr >= 1.0 - alpha,
            "min_trl_obs": min_trl_obs,
            "min_trl_years": min_trl_obs / float(periods_per_year)
            if np.isfinite(min_trl_obs)
            else np.inf,
            "enough_history": n_obs >= min_trl_obs if np.isfinite(min_trl_obs) else False,
        }
    )


def dsr_row(
    row: pd.Series,
    *,
    sr0_dsr: Optional[float],
    alpha: float = 0.05,
) -> pd.Series:
    """
    Deflated Sharpe Ratio summary for a single spread row.

    Parameters
    ----------
    row : Series
        Row containing sr_rel, n_obs, skew, kurt.
    sr0_dsr : float
        DSR null Sharpe benchmark SR0 (already adjusted for multiple tests).
    alpha : float
        One-sided significance level (1 - confidence).
    """
    sr_hat = row.get("sr_rel")
    n_obs = row.get("n_obs")
    skew = row.get("skew")
    kurt = row.get("kurt")

    if n_obs is None:
        n_obs = np.nan

    if n_obs < 2 or not np.isfinite(sr_hat):
        return pd.Series(
            {
                "sr0_dsr": np.nan,
                "sigma_sr0": np.nan,
                "dsr": np.nan,
                "dsr_pass": False,
            }
        )

    sr0_val = float(sr0_dsr) if sr0_dsr is not None else np.nan
    if not np.isfinite(sr0_val):
        return pd.Series(
            {
                "sr0_dsr": np.nan,
                "sigma_sr0": np.nan,
                "dsr": np.nan,
                "dsr_pass": False,
            }
        )

    sigma_sr0 = metrics._sigma_sr(sr0_val, skew, kurt, int(n_obs))

    if not np.isfinite(sigma_sr0) or sigma_sr0 <= 0.0:
        return pd.Series(
            {
                "sr0_dsr": sr0_val,
                "sigma_sr0": np.nan,
                "dsr": np.nan,
                "dsr_pass": False,
            }
        )

    z_dsr = (sr_hat - sr0_val) / sigma_sr0
    dsr = norm.cdf(z_dsr)

    return pd.Series(
        {
            "sr0_dsr": sr0_val,
            "sigma_sr0": sigma_sr0,
            "dsr": dsr,
            "dsr_pass": dsr >= 1.0 - alpha,
        }
    )


def verdict(row: pd.Series, *, alpha: float = 0.05) -> str:
    """
    Human-readable verdict string summarizing PSR/DSR results.
    """
    conf_pct = (1.0 - alpha) * 100.0
    if not row.get("enough_history", False):
        need = row.get("min_trl_years")
        if not np.isfinite(need):
            return "Data too short or SR_rel <= 0; cannot conclude."
        return f"Data too short for {conf_pct:.0f}% PSR; need about {need:.1f} years."
    if row.get("dsr_pass", False):
        return f"Trust: spread beats market with >={conf_pct:.0f}% DSR after selection."
    if row.get("psr_pass", False):
        return f"PSR passes at {conf_pct:.0f}% but not after multiple-test deflation."
    return "No robust outperformance vs market."


def _dsr_null_from_spreads(sr_rel: pd.Series, m_eff: float) -> dict[str, float]:
    """
    Compute the DSR null SR0 using cross-spread Sharpe variance and K_eff.

    Uses the False Strategy Theorem shift:
        SR0 = sqrt(V_hat) * [(1-gamma)*Phi^-1(1-1/K_eff)
                             + gamma*Phi^-1(1-1/(K_eff*e))]
    where V_hat is the sample variance of the observed spread Sharpes.
    """
    sr_clean = sr_rel.dropna()
    sr_mean = float(sr_clean.mean()) if sr_clean.size else np.nan
    sr_var = float(sr_clean.var(ddof=1)) if sr_clean.size >= 2 else np.nan

    if not np.isfinite(m_eff) or m_eff <= 1.0 or not np.isfinite(sr_var) or sr_var < 0.0:
        return {
            "sr_mean": sr_mean,
            "sr_var": sr_var,
            "sr0_dsr": np.nan,
        }

    euler_gamma = 0.5772156649
    z1 = norm.ppf(1.0 - 1.0 / m_eff)
    z2 = norm.ppf(1.0 - 1.0 / (m_eff * math.e))
    sr0 = math.sqrt(sr_var) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    return {
        "sr_mean": sr_mean,
        "sr_var": sr_var,
        "sr0_dsr": sr0,
    }


def compute_spread_stats(
    spread_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    m_eff: Optional[float] = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """
    Convenience wrapper to compute spread stats, PSR/DSR, and verdicts.

    PSR/MinTRL use SR0 = 0 with sigma_SR0 = 1/sqrt(T). DSR shifts the null
    to SR0 = E[max SR] using the cross-spread Sharpe variance and K_eff from
    the return correlation matrix, then evaluates the variance at that SR0
    with each spread's skew/kurtosis.
    """
    stats = summarize_spreads(spread_df)
    psr_results = stats.apply(
        lambda row: psr_mintrl_row(row, alpha=alpha, periods_per_year=periods_per_year),
        axis=1,
    )
    stats = stats.join(psr_results)

    m_eff_used = (
        float(m_eff) if m_eff is not None and np.isfinite(m_eff) else float(spread_df.shape[1])
    )
    dsr_null = _dsr_null_from_spreads(stats["sr_rel"], m_eff_used)
    stats = stats.assign(
        sr_rel_mean_all=dsr_null["sr_mean"],
        sr_rel_var_all=dsr_null["sr_var"],
        m_eff_used=m_eff_used,
        sr0_dsr_global=dsr_null["sr0_dsr"],
    )
    dsr_results = stats.apply(
        lambda row: dsr_row(row, alpha=alpha, sr0_dsr=dsr_null["sr0_dsr"]), axis=1
    )
    stats = stats.join(dsr_results)
    stats["verdict"] = stats.apply(lambda row: verdict(row, alpha=alpha), axis=1)
    return stats
