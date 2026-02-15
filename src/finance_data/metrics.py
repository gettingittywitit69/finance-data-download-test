import math
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats

    _HAVE_SCIPY = True
except ImportError:  # optional dependency
    _HAVE_SCIPY = False
    _stats = None

try:
    from scipy.stats import norm as _scipy_norm
except Exception:  # optional dependency
    _scipy_norm = None

_NORMAL_DIST = NormalDist()


def _sigma_sr(sr_ref: float, skew: float, kurt: float, n_obs: int):
    """Variance-aware SR standard error used across PSR/DSR/MinTRL."""
    if n_obs <= 1 or not np.isfinite(sr_ref):
        return np.nan
    var = (
        1.0
        - skew * sr_ref
        + 0.25 * (kurt - 1.0) * sr_ref ** 2
    ) / float(n_obs)
    if not np.isfinite(var) or var <= 0.0:
        return np.nan
    return math.sqrt(var)


def _norm_cdf(x):
    """Standard normal CDF helper, using SciPy if available."""
    if _scipy_norm is not None:
        return float(_scipy_norm.cdf(x))
    return float(_NORMAL_DIST.cdf(x))


def _norm_ppf(p):
    """Standard normal inverse CDF helper, using SciPy if available."""
    if _scipy_norm is not None:
        return float(_scipy_norm.ppf(p))
    return float(_NORMAL_DIST.inv_cdf(p))


def sharpe_ratio(r):
    """
    Sharpe ratio: mean(r) / std(r) with ddof=0 (T in denominator), ignoring NaNs.

    This matches the estimator used in ssrn-5520741, computed at the
    observation frequency (no annualization).
    """
    r_clean = r.dropna()
    if r_clean.shape[0] < 2:
        return np.nan
    mean_r = float(r_clean.mean())
    sigma_r = float(r_clean.std(ddof=0))
    if sigma_r <= 0.0:
        return np.nan
    return mean_r / sigma_r


def sample_skewness(r):
    """
    Unbiased sample skewness (Fisher), ignoring NaNs.
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    if n_obs < 3:
        return np.nan
    if _HAVE_SCIPY:
        return float(_stats.skew(r_clean.values, bias=False))
    mean_r = float(r_clean.mean())
    s2 = float(((r_clean - mean_r) ** 2).sum() / (n_obs - 1))
    if s2 <= 0.0:
        return np.nan
    s = math.sqrt(s2)
    z = (r_clean - mean_r) / s
    skew = (n_obs / ((n_obs - 1) * (n_obs - 2))) * float((z ** 3).sum())
    return skew


def sample_kurtosis(r):
    """
    Sample kurtosis (non-excess; Normal is about 3), ignoring NaNs.
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    if n_obs < 4:
        return np.nan
    if _HAVE_SCIPY:
        return float(_stats.kurtosis(r_clean.values, fisher=False, bias=False))
    mean_r = float(r_clean.mean())
    s2 = float(((r_clean - mean_r) ** 2).sum() / (n_obs - 1))
    if s2 <= 0.0:
        return np.nan
    s = math.sqrt(s2)
    z = (r_clean - mean_r) / s
    term1 = (n_obs * (n_obs + 1)) / ((n_obs - 1) * (n_obs - 2) * (n_obs - 3)) * float(
        (z ** 4).sum()
    )
    term2 = (3 * (n_obs - 1) ** 2) / ((n_obs - 2) * (n_obs - 3))
    excess_kurt = term1 - term2
    kurtosis = excess_kurt + 3.0
    return kurtosis


def probabilistic_sharpe_ratio(
    r,
    sr_benchmark: float = 0.0,
    mode: str = "paper",
):
    """
    Probabilistic Sharpe Ratio (PSR) for a 1D excess-return series.

    Parameters
    ----------
    r : pd.Series
        Excess returns at the observation frequency.
    sr_benchmark : float
        Benchmark Sharpe ratio SR0 in the null hypothesis H0: SR <= SR0.
    mode : {'paper', 'approx'}
        - 'paper': matches the formula in ssrn-5520741 (eqs. (5) and (9)),
          i.e. the variance term is evaluated at SR0 and uses 1/T.
        - 'approx': reproduces the earlier notebook implementation that
          evaluates the variance at SR_hat and uses (T-1).

    Returns
    -------
    psr : float
        PSR = P(true SR >= SR0) under the specified mode.
    sr_hat : float
        Sample Sharpe ratio.
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    if n_obs < 2:
        return np.nan, np.nan

    sr_hat = sharpe_ratio(r_clean)
    skew = sample_skewness(r_clean)
    kurt = sample_kurtosis(r_clean)
    if not np.isfinite(sr_hat) or not np.isfinite(skew) or not np.isfinite(kurt):
        return np.nan, sr_hat

    mode_lower = mode.lower()

    if mode_lower == "approx":
        # Previous implementation: variance evaluated at SR_hat and scaled by T.
        denom_raw = 1.0 - skew * sr_hat + 0.25 * (kurt - 1.0) * sr_hat ** 2
        denom = math.sqrt(max(1e-16, denom_raw))
        z = ((sr_hat - sr_benchmark) * math.sqrt(float(n_obs))) / denom
        psr = _norm_cdf(z)
        return psr, sr_hat

    # Default: 'paper' mode – eq. (5) + eq. (9), variance at SR0 with 1/T.
    sigma_sr0_sq = (
        1.0
        - skew * sr_benchmark
        + 0.25 * (kurt - 1.0) * sr_benchmark ** 2
    ) / float(n_obs)
    sigma_sr0_sq = max(1e-16, sigma_sr0_sq)
    sigma_sr0 = math.sqrt(sigma_sr0_sq)
    z = (sr_hat - sr_benchmark) / sigma_sr0
    psr = _norm_cdf(z)
    return psr, sr_hat


def _bootstrap_resample_1d(
    arr: np.ndarray,
    rng: np.random.Generator,
    block_size: Optional[int],
) -> np.ndarray:
    """
    Bootstrap helper for 1D arrays with optional block resampling.

    Parameters
    ----------
    arr : np.ndarray
        1D array of observations (already NaN-free).
    rng : np.random.Generator
        Random generator used for draws.
    block_size : int, optional
        Block length for simple moving-block bootstrap. If None or <=1, draws
        iid samples with replacement.
    """
    n_obs = arr.shape[0]
    if n_obs == 0:
        return arr

    if block_size is None or block_size <= 1:
        idx = rng.integers(0, n_obs, size=n_obs)
        return arr[idx]

    block = max(1, min(int(block_size), n_obs))
    starts = np.arange(0, n_obs - block + 1)
    draws_per_sample = int(math.ceil(n_obs / float(block)))
    chosen = rng.integers(0, starts.shape[0], size=draws_per_sample)
    pieces = [arr[start : start + block] for start in starts[chosen]]
    sample = np.concatenate(pieces)
    return sample[:n_obs]


def bootstrap_psr(
    r: pd.Series,
    *,
    sr_benchmark: float = 0.0,
    mode: str = "paper",
    n_boot: int = 500,
    ci_level: float = 0.90,
    block_size: Optional[int] = None,
    random_state: Optional[np.random.Generator | int] = None,
    return_samples: bool = True,
):
    """
    Bootstrap the PSR to quantify uncertainty from sample moments.

    Steps:
      1. Resample the return series (iid or moving blocks).
      2. Re-estimate SR/skew/kurt on each draw.
      3. Feed moments into the PSR formula to build a PSR distribution.

    Parameters
    ----------
    r : pd.Series
        Return series to resample (NaNs are dropped).
    sr_benchmark : float
        Null Sharpe ratio SR0 in the PSR test.
    mode : {'paper', 'approx'}
        Passed through to `probabilistic_sharpe_ratio`.
    n_boot : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level for the PSR interval (e.g., 0.90 -> middle 90%).
    block_size : int, optional
        Block length for moving-block bootstrap. If None/<=1, uses iid draws.
    random_state : np.random.Generator or int, optional
        Seed or Generator for reproducibility.
    return_samples : bool
        Whether to include the raw PSR sample array in the result.

    Returns
    -------
    dict with keys:
        - base_psr, base_sr, base_skew, base_kurt, n_obs
        - psr_samples (optional numpy array of valid draws)
        - psr_mean, psr_median, psr_std
        - psr_ci_lower, psr_ci_upper, psr_ci_width, ci_level
        - bias_vs_base, abs_error_vs_base, rel_error_vs_base
        - n_boot, n_valid
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    base_psr, base_sr = probabilistic_sharpe_ratio(
        r_clean, sr_benchmark=sr_benchmark, mode=mode
    )
    base_skew = sample_skewness(r_clean)
    base_kurt = sample_kurtosis(r_clean)

    if (
        n_obs < 2
        or not np.isfinite(base_sr)
        or not np.isfinite(base_skew)
        or not np.isfinite(base_kurt)
    ):
        result = {
            "base_psr": base_psr,
            "base_sr": base_sr,
            "base_skew": base_skew,
            "base_kurt": base_kurt,
            "n_obs": n_obs,
            "psr_samples": np.array([]) if return_samples else None,
            "psr_mean": np.nan,
            "psr_median": np.nan,
            "psr_std": np.nan,
            "psr_ci_lower": np.nan,
            "psr_ci_upper": np.nan,
            "psr_ci_width": np.nan,
            "ci_level": ci_level,
            "bias_vs_base": np.nan,
            "abs_error_vs_base": np.nan,
            "rel_error_vs_base": np.nan,
            "n_boot": n_boot,
            "n_valid": 0,
        }
        if not return_samples:
            result.pop("psr_samples")
        return result

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    arr = r_clean.values
    psr_draws = np.full(n_boot, np.nan)

    for i in range(n_boot):
        sample = _bootstrap_resample_1d(arr, rng, block_size)
        psr_i, _ = probabilistic_sharpe_ratio(
            pd.Series(sample), sr_benchmark=sr_benchmark, mode=mode
        )
        psr_draws[i] = psr_i

    valid = psr_draws[np.isfinite(psr_draws)]
    n_valid = valid.shape[0]

    if n_valid == 0:
        ci_lower = ci_upper = ci_width = psr_mean = psr_median = psr_std = np.nan
    else:
        alpha = max(0.0, min(1.0, 1.0 - ci_level))
        lo_q = 100.0 * alpha / 2.0
        hi_q = 100.0 * (1.0 - alpha / 2.0)
        ci_lower, ci_upper = np.nanpercentile(valid, [lo_q, hi_q])
        ci_width = ci_upper - ci_lower
        psr_mean = float(np.nanmean(valid))
        psr_median = float(np.nanmedian(valid))
        psr_std = float(np.nanstd(valid, ddof=0))

    bias_vs_base = (
        psr_median - base_psr if np.isfinite(psr_median) and np.isfinite(base_psr) else np.nan
    )
    abs_error_vs_base = abs(bias_vs_base) if np.isfinite(bias_vs_base) else np.nan
    rel_error_vs_base = (
        abs_error_vs_base / abs(base_psr)
        if np.isfinite(abs_error_vs_base) and np.isfinite(base_psr) and base_psr != 0.0
        else np.nan
    )

    result = {
        "base_psr": base_psr,
        "base_sr": base_sr,
        "base_skew": base_skew,
        "base_kurt": base_kurt,
        "n_obs": n_obs,
        "psr_samples": valid if return_samples else None,
        "psr_mean": psr_mean,
        "psr_median": psr_median,
        "psr_std": psr_std,
        "psr_ci_lower": ci_lower,
        "psr_ci_upper": ci_upper,
        "psr_ci_width": ci_width,
        "ci_level": ci_level,
        "bias_vs_base": bias_vs_base,
        "abs_error_vs_base": abs_error_vs_base,
        "rel_error_vs_base": rel_error_vs_base,
        "n_boot": n_boot,
        "n_valid": n_valid,
    }
    if not return_samples:
        result.pop("psr_samples")
    return result

def effective_num_tests(ex_window):
    """
    Effective number of tests K_eff from the correlation matrix of returns.

    Uses the participation ratio of the eigenvalues of the correlation matrix
    as suggested in ssrn-5520741 and related work.
    """
    if ex_window is None or ex_window.empty:
        return np.nan
    ex_valid = ex_window.dropna(axis=1, how="all")
    if ex_valid.shape[1] == 0:
        return np.nan
    corr = ex_valid.corr()
    if corr.isna().all().all():
        return np.nan
    eigvals = np.linalg.eigvalsh(corr.values)
    num = float(np.sum(eigvals)) ** 2
    denom = float(np.sum(eigvals ** 2))
    if denom <= 0.0:
        return np.nan
    return num / denom


def deflated_sharpe_ratio(
    r,
    m_eff: float,
    mode: str = "paper",
):
    """
    Deflated Sharpe Ratio (DSR) for a 1D excess-return series.

    Parameters
    ----------
    r : pd.Series
        Excess returns at the observation frequency.
    m_eff : float
        Effective number of independent trials (K_eff).
    mode : {'paper', 'approx'}
        - 'paper': uses the False Strategy Theorem shift from eq. (26),
          i.e. SR0 is set to E[max SR] ≈ sigma_SR * f(K_eff).
        - 'approx': reproduces the earlier notebook implementation that
          uses SR0 = z_{1 - 1/M_eff} * sigma_SR and the same sigma_SR
          in the denominator.

    Returns
    -------
    dsr : float
        Deflated Sharpe Ratio in probability space.
    sr_hat : float
        Sample Sharpe ratio.
    sr0 : float
        Deflated Sharpe benchmark SR0 used in the test.
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    if n_obs < 2 or not np.isfinite(m_eff) or m_eff <= 1.0:
        return np.nan, np.nan, np.nan

    sr_hat = sharpe_ratio(r_clean)
    skew = sample_skewness(r_clean)
    kurt = sample_kurtosis(r_clean)
    if not np.isfinite(sr_hat) or not np.isfinite(skew) or not np.isfinite(kurt):
        return np.nan, sr_hat, np.nan

    mode_lower = mode.lower()

    if mode_lower == "approx":
        # Previous implementation: variance at SR_hat with T, SR0 via a single z-quantile.
        denom_n = float(n_obs)
        sigma_sr_sq = (
            1.0 - skew * sr_hat + 0.25 * (kurt - 1.0) * sr_hat ** 2
        ) / denom_n
        if sigma_sr_sq <= 0.0:
            return np.nan, sr_hat, np.nan
        sigma_sr = math.sqrt(sigma_sr_sq)
        z_alpha = _norm_ppf(1.0 - 1.0 / m_eff)
        sr0 = z_alpha * sigma_sr
        z_dsr = (sr_hat - sr0) / sigma_sr
        dsr = _norm_cdf(z_dsr)
        return dsr, sr_hat, sr0

    # Default: 'paper' mode – False Strategy Theorem shift (eq. (26)).
    sigma_sr = _sigma_sr(sr_hat, skew, kurt, n_obs)
    if not np.isfinite(sigma_sr):
        return np.nan, sr_hat, np.nan

    sr0 = expected_max_sharpe(m_eff=m_eff, sigma_sr=sigma_sr)

    z_dsr = (sr_hat - sr0) / sigma_sr
    dsr = _norm_cdf(z_dsr)
    return dsr, sr_hat, sr0


def expected_max_sharpe(m_eff: float, sigma_sr: float):
    """
    Expected value of the maximum Sharpe ratio across m_eff independent trials.

    Uses the False Strategy Theorem approximation from eq. (26) in ssrn-5520741.
    """
    if not np.isfinite(m_eff) or m_eff <= 1.0 or not np.isfinite(sigma_sr):
        return 0.0
    euler_gamma = 0.5772156649
    z1 = _norm_ppf(1.0 - 1.0 / m_eff)
    z2 = _norm_ppf(1.0 - 1.0 / (m_eff * math.e))
    return sigma_sr * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)


def min_track_record_length(
    r: pd.Series,
    sr_benchmark: float = 0.0,
    conf_level: float = 0.99,
    periods_per_year: int = 12,
    m_eff: Optional[float] = None,
):
    """
    Minimum Track Record Length (MinTRL) needed for PSR >= conf_level.

    Parameters
    ----------
    r : pd.Series
        Excess returns at the observation frequency.
    sr_benchmark : float
        Null Sharpe ratio SR0.
    conf_level : float
        Target one-sided confidence level for PSR (e.g., 0.99).
    periods_per_year : int
        Number of observations per year (12 for monthly, 252 for daily).
    m_eff : float, optional
        Effective number of independent trials. When provided (>1), SR0 is
        shifted by the False Strategy Theorem to guard against multiple tests.

    Returns
    -------
    dict with keys:
        - sr_hat: observed Sharpe ratio
        - skew, kurt: moment estimates
        - min_obs: required observations
        - min_years: required years (min_obs / periods_per_year)
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    sr_hat = sharpe_ratio(r_clean)
    skew = sample_skewness(r_clean)
    kurt = sample_kurtosis(r_clean)
    if n_obs < 2 or not np.isfinite(sr_hat) or not np.isfinite(skew) or not np.isfinite(kurt):
        return {
            "sr_hat": np.nan,
            "skew": skew,
            "kurt": kurt,
            "min_obs": np.nan,
            "min_years": np.nan,
        }

    sr0 = sr_benchmark
    sigma_sr_hat = _sigma_sr(sr_hat, skew, kurt, n_obs)
    if m_eff is not None and m_eff > 1.0 and np.isfinite(sigma_sr_hat):
        sr0 += expected_max_sharpe(m_eff=m_eff, sigma_sr=sigma_sr_hat)

    z_alpha = _norm_ppf(conf_level)
    numerator = 1.0 - skew * sr0 + 0.25 * (kurt - 1.0) * sr0 ** 2
    delta = sr_hat - sr0
    if not np.isfinite(numerator) or numerator <= 0.0 or not np.isfinite(delta) or delta <= 0.0:
        min_obs = np.inf
    else:
        min_obs = numerator / ((delta / z_alpha) ** 2)

    min_years = min_obs / float(periods_per_year) if np.isfinite(min_obs) else np.inf
    return {
        "sr_hat": sr_hat,
        "skew": skew,
        "kurt": kurt,
        "min_obs": min_obs,
        "min_years": min_years,
    }


def observed_fdr(
    r: pd.Series,
    sr_benchmark: float = 0.0,
    sr_alt: Optional[float] = None,
    prior_h1: float = 0.1,
    m_eff: Optional[float] = None,
):
    """
    Observed Bayesian tail-area False Discovery Rate (oFDR), eq. (22)-(24).

    Parameters
    ----------
    r : pd.Series
        Excess returns.
    sr_benchmark : float
        Null Sharpe ratio (SR0).
    sr_alt : float, optional
        Alternative Sharpe ratio (SR1). If None, uses the observed SR_hat.
    prior_h1 : float
        Prior probability of skill (P[H1]).
    m_eff : float, optional
        Effective number of independent trials. When provided, SR0 and SR1 are
        shifted using the expected maximum Sharpe to deflate discoveries.

    Returns
    -------
    dict with keys:
        - sr_hat
        - sr0, sr1: effective SR0/SR1 after adjustments
        - p_value: tail probability under H0
        - tail_h1: tail probability under H1
        - ofdr: posterior P[H0 | SR_hat >= observed]
    """
    r_clean = r.dropna()
    n_obs = r_clean.shape[0]
    sr_hat = sharpe_ratio(r_clean)
    skew = sample_skewness(r_clean)
    kurt = sample_kurtosis(r_clean)
    if (
        n_obs < 2
        or not np.isfinite(sr_hat)
        or not np.isfinite(skew)
        or not np.isfinite(kurt)
        or not (0.0 <= prior_h1 <= 1.0)
    ):
        return {
            "sr_hat": np.nan,
            "sr0": np.nan,
            "sr1": np.nan,
            "p_value": np.nan,
            "tail_h1": np.nan,
            "ofdr": np.nan,
        }

    sr1_base = sr_hat if sr_alt is None or not np.isfinite(sr_alt) else float(sr_alt)
    sr0 = float(sr_benchmark)
    sr1 = float(sr1_base)

    sigma_sr_hat = _sigma_sr(sr_hat, skew, kurt, n_obs)
    if m_eff is not None and m_eff > 1.0 and np.isfinite(sigma_sr_hat):
        shift = expected_max_sharpe(m_eff=m_eff, sigma_sr=sigma_sr_hat)
        sr0 += shift
        sr1 += shift

    sigma_sr0 = _sigma_sr(sr0, skew, kurt, n_obs)
    sigma_sr1 = _sigma_sr(sr1, skew, kurt, n_obs)
    if not np.isfinite(sigma_sr0) or not np.isfinite(sigma_sr1) or sigma_sr0 <= 0.0 or sigma_sr1 <= 0.0:
        return {
            "sr_hat": sr_hat,
            "sr0": sr0,
            "sr1": sr1,
            "p_value": np.nan,
            "tail_h1": np.nan,
            "ofdr": np.nan,
        }

    z0 = (sr_hat - sr0) / sigma_sr0
    p_value = 1.0 - _norm_cdf(z0)  # P[SR_hat >= observed | H0]
    z1 = (sr_hat - sr1) / sigma_sr1
    tail_h1 = 1.0 - _norm_cdf(z1)  # P[SR_hat >= observed | H1]

    p0 = 1.0 - prior_h1
    p1 = prior_h1
    denom = p_value * p0 + tail_h1 * p1
    ofdr = (p_value * p0) / denom if denom > 0.0 else np.nan

    return {
        "sr_hat": sr_hat,
        "sr0": sr0,
        "sr1": sr1,
        "p_value": p_value,
        "tail_h1": tail_h1,
        "ofdr": ofdr,
    }


def tangency_portfolio(
    excess_returns: pd.DataFrame,
    ridge: float = 1e-6,
    min_obs: int = 24,
):
    """
    Compute the max-Sharpe tangency portfolio from excess returns (r - RF).

    Parameters
    ----------
    excess_returns : pd.DataFrame
        Excess returns of the risky assets (already net of risk-free).
    ridge : float
        Small diagonal ridge added to the covariance for numerical stability.
    min_obs : int
        Minimum observations required; below this returns NaNs.

    Returns
    -------
    dict with keys:
        - weights: pd.Series normalized to sum to 1 on the kept assets
        - mean: portfolio mean using those weights
        - vol: portfolio volatility using those weights
        - sr: Sharpe ratio of the normalized portfolio
        - sr_max: theoretical max Sharpe sqrt(mu^T Sigma^{-1} mu)
        - kept_assets / dropped_assets
    """
    if excess_returns is None or excess_returns.empty:
        return {
            "weights": pd.Series(dtype=float),
            "mean": np.nan,
            "vol": np.nan,
            "sr": np.nan,
            "sr_max": np.nan,
            "kept_assets": [],
            "dropped_assets": [],
        }

    ex_valid = excess_returns.dropna(axis=1, how="all").dropna(axis=0, how="all")
    dropped_assets = [c for c in excess_returns.columns if c not in ex_valid.columns]
    t_obs, n_assets = ex_valid.shape
    if t_obs < min_obs or n_assets == 0:
        return {
            "weights": pd.Series(dtype=float),
            "mean": np.nan,
            "vol": np.nan,
            "sr": np.nan,
            "sr_max": np.nan,
            "kept_assets": list(ex_valid.columns),
            "dropped_assets": dropped_assets,
        }

    mu = ex_valid.mean()
    cov = ex_valid.cov()
    if ridge > 0.0:
        cov = cov + np.eye(n_assets) * ridge

    cov_inv = np.linalg.pinv(cov.values)
    mu_vec = mu.values.reshape(-1, 1)
    w_raw = cov_inv @ mu_vec  # direction of max Sharpe weights
    w_raw = w_raw.flatten()

    if not np.isfinite(w_raw).all():
        return {
            "weights": pd.Series(dtype=float),
            "mean": np.nan,
            "vol": np.nan,
            "sr": np.nan,
            "sr_max": np.nan,
            "kept_assets": list(ex_valid.columns),
            "dropped_assets": dropped_assets,
        }

    denom = w_raw.sum()
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        norm = np.linalg.norm(w_raw, 1)
        weights_np = w_raw / norm if norm > 0.0 else np.full_like(w_raw, np.nan)
    else:
        weights_np = w_raw / denom

    weights = pd.Series(weights_np, index=ex_valid.columns)
    mean_p = float(np.dot(weights_np, mu.values))
    vol_p = math.sqrt(float(weights_np @ cov.values @ weights_np))
    sr_p = mean_p / vol_p if vol_p > 0.0 else np.nan

    sr_max = float(math.sqrt(max(0.0, mu.values @ cov_inv @ mu.values)))

    return {
        "weights": weights,
        "mean": mean_p,
        "vol": vol_p,
        "sr": sr_p,
        "sr_max": sr_max,
        "kept_assets": list(ex_valid.columns),
        "dropped_assets": dropped_assets,
    }
