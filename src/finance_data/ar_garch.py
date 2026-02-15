"""AR(1)-GARCH(1,1) with Student-t utilities and parameter uncertainty."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult

EPSILON = 1e-12
MIN_NU = 2.05  # ensure finite variance for Student-t draws


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    """Return a 1D float64 array with NaNs removed."""
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Input returns array is empty or non-finite.")
    return arr


def _resolve_param_names(params_index: Tuple[str, ...]) -> Tuple[str, str, str, str, str, str]:
    """Map arch result parameter names to (mu, phi, omega, alpha, beta, nu)."""
    candidates = {
        "mu": ("Const", "const", "mu"),
        "phi": ("y[1]", "AR[1]", "ar.L1"),
        "omega": ("omega",),
        "alpha": ("alpha[1]",),
        "beta": ("beta[1]",),
        "nu": ("nu",),
    }
    index_set = set(params_index)
    names: Dict[str, str] = {}
    for key, options in candidates.items():
        for opt in options:
            if opt in index_set:
                names[key] = opt
                break
        if key not in names:
            raise KeyError(f"Unable to find parameter name for {key} in {params_index}.")
    return names["mu"], names["phi"], names["omega"], names["alpha"], names["beta"], names["nu"]


def fit_ar_garch_t(returns: np.ndarray) -> Dict[str, object]:
    """
    Fit an AR(1)-GARCH(1,1) with Student-t innovations using arch.

    Parameter order is (mu, phi, omega, alpha, beta, nu), corresponding to
    the intercept, AR(1) coefficient, GARCH constant, ARCH term, GARCH term,
    and Student-t degrees of freedom.

    Returns a dictionary with keys:
        - params: np.ndarray of shape (6,)
        - cov: np.ndarray of shape (6, 6)
        - res: ARCHModelResult
    """
    r = _clean_returns(returns)
    model = arch_model(
        r,
        mean="ARX",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    res: ARCHModelResult = model.fit(update_freq=0, disp="off", show_warning=False)

    mu_name, phi_name, omega_name, alpha_name, beta_name, nu_name = _resolve_param_names(
        tuple(res.params.index)
    )
    param_order = (mu_name, phi_name, omega_name, alpha_name, beta_name, nu_name)
    params = res.params.loc[list(param_order)].to_numpy(dtype=np.float64)
    cov_df = res.param_cov.loc[list(param_order), list(param_order)]
    cov_arr = cov_df.to_numpy(dtype=np.float64)
    cov = 0.5 * (cov_arr + cov_arr.T)

    return {"params": params, "cov": cov, "res": res}


def _ensure_positive_definite(cov: np.ndarray, base_jitter: float = 1e-10) -> np.ndarray:
    """Add diagonal jitter until the covariance matrix is positive definite."""
    cov_pd = np.array(cov, dtype=np.float64, copy=True)
    cov_pd = 0.5 * (cov_pd + cov_pd.T)
    for i in range(7):
        try:
            np.linalg.cholesky(cov_pd)
            return cov_pd
        except np.linalg.LinAlgError:
            jitter = base_jitter * (10.0 ** i)
            cov_pd = cov_pd + np.eye(cov_pd.shape[0], dtype=np.float64) * jitter
    # Final attempt; may still raise but we return the jittered version.
    return cov_pd


def sample_params_normal(
    theta_hat: np.ndarray,
    cov_hat: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw parameters from N(theta_hat, cov_hat) with diagonal jitter if needed.

    Parameters
    ----------
    theta_hat : np.ndarray
        Point estimate of parameters, ordered as (mu, phi, omega, alpha, beta, nu).
    cov_hat : np.ndarray
        Covariance matrix of theta_hat.
    n_draws : int
        Number of samples.
    rng : np.random.Generator
        Random number generator used for draws.
    """
    theta = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov_hat, dtype=np.float64)
    cov = _ensure_positive_definite(cov)
    return rng.multivariate_normal(
        mean=theta,
        cov=cov,
        size=int(n_draws),
        check_valid="ignore",
        method="cholesky",
    )


def simulate_ar_garch_t_paths(
    theta: np.ndarray,
    n_paths: int,
    horizon: int,
    r0: float,
    h0: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate AR(1)-GARCH(1,1)-t return and variance paths.

    Uses variance-1 Student-t shocks scaled by sqrt(h_t):
        eps_t = sqrt(h_t) * z_t,   z_t ~ t_nu(0, 1)
        r_t   = mu + phi * r_{t-1} + eps_t
        h_t   = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
    """
    mu, phi, omega, alpha, beta, nu = (float(x) for x in theta)
    nu_eff = max(nu, MIN_NU)
    scale = math.sqrt((nu_eff - 2.0) / nu_eff)
    z = rng.standard_t(df=nu_eff, size=(horizon, n_paths)).astype(np.float64) * scale

    returns_paths = np.empty((horizon, n_paths), dtype=np.float64)
    vols_paths = np.empty((horizon, n_paths), dtype=np.float64)

    r_prev = np.full(n_paths, float(r0), dtype=np.float64)
    h_prev = np.full(n_paths, max(float(h0), EPSILON), dtype=np.float64)
    eps_prev = np.zeros(n_paths, dtype=np.float64)

    for t in range(horizon):
        h_t = omega + alpha * (eps_prev ** 2) + beta * h_prev
        h_t = np.maximum(h_t, EPSILON)
        eps_t = np.sqrt(h_t) * z[t]
        r_t = mu + phi * r_prev + eps_t

        returns_paths[t] = r_t
        vols_paths[t] = h_t

        r_prev = r_t
        h_prev = h_t
        eps_prev = eps_t

    return returns_paths, vols_paths


def compute_path_moments(returns_paths: np.ndarray, risk_free: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Compute Sharpe ratio, skewness, and excess kurtosis for each simulated path.

    Sharpe ratio uses sample mean / sample std (ddof=0). Skewness and kurtosis
    are based on standardized central moments with population denominators,
    and kurtosis is returned as excess kurtosis (Normal -> 0).
    """
    x = np.asarray(returns_paths, dtype=np.float64)
    centered = x - float(risk_free)
    mean_r = centered.mean(axis=0)
    std_r = centered.std(axis=0, ddof=0)
    sharpe = np.divide(mean_r, std_r, out=np.full_like(mean_r, np.nan), where=std_r > 0.0)

    centered -= mean_r  # broadcasts over time axis
    m2 = np.mean(centered ** 2, axis=0)
    m3 = np.mean(centered ** 3, axis=0)
    m4 = np.mean(centered ** 4, axis=0)

    denom_skew = np.power(m2, 1.5)
    skew = np.divide(m3, denom_skew, out=np.full_like(m3, np.nan), where=denom_skew > 0.0)
    kurt = np.divide(m4, m2 ** 2, out=np.full_like(m4, np.nan), where=m2 > 0.0) - 3.0

    return {"sharpe": sharpe, "skew": skew, "kurtosis": kurt}


def run_parameter_uncertainty_experiment(
    returns: np.ndarray,
    n_param_draws: int,
    n_paths_per_draw: int,
    horizon: int,
    risk_free: float = 0.0,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Propagate parameter uncertainty via Monte Carlo simulation.

    Steps:
      1. Fit AR(1)-GARCH(1,1)-t.
      2. Sample parameter draws from the asymptotic Normal approximation.
      3. Simulate return paths for each draw and compute path moments.
      4. Stack results across draws.
    """
    rng = np.random.default_rng(seed)
    fit = fit_ar_garch_t(returns)
    theta_hat = fit["params"]
    cov_hat = fit["cov"]

    theta_draws = sample_params_normal(theta_hat, cov_hat, n_param_draws, rng)
    n_total = n_param_draws * n_paths_per_draw
    sharpe_all = np.empty(n_total, dtype=np.float64)
    skew_all = np.empty(n_total, dtype=np.float64)
    kurt_all = np.empty(n_total, dtype=np.float64)

    r_clean = _clean_returns(returns)
    r0 = float(r_clean[-1])
    h_last = float(np.asarray(fit["res"].conditional_volatility)[-1])
    h0 = max(h_last ** 2, EPSILON)

    offset = 0
    for theta in theta_draws:
        ret_paths, _ = simulate_ar_garch_t_paths(
            theta=theta,
            n_paths=n_paths_per_draw,
            horizon=horizon,
            r0=r0,
            h0=h0,
            rng=rng,
        )
        moments = compute_path_moments(ret_paths, risk_free=risk_free)
        sl = slice(offset, offset + n_paths_per_draw)
        sharpe_all[sl] = moments["sharpe"]
        skew_all[sl] = moments["skew"]
        kurt_all[sl] = moments["kurtosis"]
        offset += n_paths_per_draw

    return {
        "sharpe": sharpe_all,
        "skew": skew_all,
        "kurtosis": kurt_all,
        "theta_draws": theta_draws,
    }


def _demo_simulated_returns(seed: int = 1234) -> np.ndarray:
    """Generate a small synthetic AR-GARCH-t series for the __main__ example."""
    rng = np.random.default_rng(seed)
    t = 800
    mu_true, phi_true = 0.0005, 0.1
    omega_true, alpha_true, beta_true = 0.00002, 0.05, 0.9
    nu_true = 8.0

    scale = math.sqrt((nu_true - 2.0) / nu_true)
    z = rng.standard_t(df=nu_true, size=t).astype(np.float64) * scale
    r = np.empty(t, dtype=np.float64)
    h = np.empty(t, dtype=np.float64)

    r_prev = 0.0
    h_prev = omega_true / (1.0 - alpha_true - beta_true)
    eps_prev = 0.0
    for i in range(t):
        h_t = omega_true + alpha_true * (eps_prev ** 2) + beta_true * h_prev
        h_t = max(h_t, EPSILON)
        eps_t = math.sqrt(h_t) * z[i]
        r_t = mu_true + phi_true * r_prev + eps_t
        r[i] = r_t
        h[i] = h_t
        r_prev, h_prev, eps_prev = r_t, h_t, eps_t
    return r


if __name__ == "__main__":
    demo_returns = _demo_simulated_returns()
    results = run_parameter_uncertainty_experiment(
        demo_returns,
        n_param_draws=100,
        n_paths_per_draw=100,
        horizon=240,
        risk_free=0.0,
        seed=42,
    )

    def _summaries(arr: np.ndarray) -> Dict[str, float]:
        arr_f = arr[np.isfinite(arr)]
        if arr_f.size == 0:
            return {"mean": np.nan, "p05": np.nan, "median": np.nan, "p95": np.nan}
        return {
            "mean": float(np.mean(arr_f)),
            "p05": float(np.percentile(arr_f, 5)),
            "median": float(np.percentile(arr_f, 50)),
            "p95": float(np.percentile(arr_f, 95)),
        }

    print("Parameter draws (mu, phi, omega, alpha, beta, nu):")
    print(results["theta_draws"][:3])
    print("\nSharpe summary:", _summaries(results["sharpe"]))
    print("Skewness summary:", _summaries(results["skew"]))
    print("Excess kurtosis summary:", _summaries(results["kurtosis"]))
