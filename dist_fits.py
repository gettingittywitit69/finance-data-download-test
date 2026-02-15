"""Lightweight distribution fitting utilities (Student-t and SGT) using NumPy/SciPy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import beta, gamma


# ----------------------------
# Student-t helpers and fitting
# ----------------------------

def _std_t_logpdf(t: np.ndarray, nu: float) -> np.ndarray:
    """
    Log-pdf of a standard Student-t with df=nu evaluated at t.

    Parameters
    ----------
    t : np.ndarray
        Points at which to evaluate the log-pdf.
    nu : float
        Degrees of freedom (nu > 0).
    """
    t = np.asarray(t, dtype=float)
    nu = float(nu)
    half_nu = 0.5 * nu
    log_norm = (
        np.log(gamma(0.5 * (nu + 1.0)))
        - np.log(gamma(half_nu))
        - 0.5 * (np.log(np.pi) + np.log(nu))
    )
    return log_norm - 0.5 * (nu + 1.0) * np.log1p((t * t) / nu)


def student_t_loglik(params: np.ndarray, x: np.ndarray) -> float:
    """
    Negative log-likelihood for a location-scale Student-t distribution.

    Parameters
    ----------
    params : array-like
        Unconstrained parameters (mu, log_sigma, log_nu_minus_4).
    x : np.ndarray
        1D array of observations.
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = float(params[0])
    sigma = float(np.exp(params[1]))
    nu = 4.0 + float(np.exp(params[2]))

    if not np.isfinite(sigma) or sigma <= 0.0:
        return 1e20
    if not np.isfinite(nu) or nu <= 4.0:
        return 1e20

    z = (x - mu) / sigma
    logpdf = _std_t_logpdf(z, nu) - np.log(sigma)
    if not np.all(np.isfinite(logpdf)):
        return 1e20

    nll = -np.sum(logpdf)

    # Mild penalties to discourage extreme parameter values that can
    # lead to numerical issues during optimization.
    if sigma < 1e-8:
        nll += (1e-8 - sigma) * 1e6
    if nu > 1e6:
        nll += (nu - 1e6) * 1e-3
    return float(nll)


def fit_student_t_mle(x: np.ndarray, init: np.ndarray | None = None) -> dict:
    """
    Fit a location-scale Student-t by maximum likelihood.

    Parameters
    ----------
    x : np.ndarray
        1D array of observations.
    init : array-like, optional
        Initial guess in constrained space (mu, sigma, nu). If None, uses
        sample mean, sample std (ddof=1), and nu0=10.

    Returns
    -------
    dict with keys: mu, sigma, nu, success, hess_inv, fun
    """
    x = np.asarray(x, dtype=float).ravel()
    if init is None:
        mu0 = float(np.mean(x))
        sigma0 = float(np.std(x, ddof=1))
        if sigma0 <= 0.0 or not np.isfinite(sigma0):
            sigma0 = 1.0
        nu0 = 10.0
    else:
        mu0, sigma0, nu0 = init
        sigma0 = float(sigma0)
        nu0 = float(nu0)
        if sigma0 <= 0.0:
            sigma0 = 1.0
        if nu0 <= 4.0:
            nu0 = 5.0

    theta0 = np.array([mu0, np.log(sigma0), np.log(nu0 - 4.0)], dtype=float)
    res = minimize(
        student_t_loglik,
        theta0,
        args=(x,),
        method="L-BFGS-B",
    )
    mu_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    nu_hat = float(4.0 + np.exp(res.x[2]))

    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "nu": nu_hat,
        "success": bool(res.success),
        "hess_inv": res.hess_inv,
        "fun": float(res.fun),
    }


def student_t_moments(mu: float, sigma: float, nu: float) -> dict:
    """
    Analytic moments for a location-scale Student-t with nu > 4.

    Returns
    -------
    dict with mean, var, m4, skew, kurtosis, excess_kurtosis.
    """
    mu = float(mu)
    sigma = float(sigma)
    nu = float(nu)
    if nu <= 4.0:
        raise ValueError("nu must be > 4 for finite fourth moment.")

    var_t = nu / (nu - 2.0)
    kurt_t = 3.0 + 6.0 / (nu - 4.0)
    m4_t = kurt_t * (var_t ** 2)

    var_x = (sigma ** 2) * var_t
    m4_x = (sigma ** 4) * m4_t

    return {
        "mean": mu,
        "var": var_x,
        "m4": m4_x,
        "skew": 0.0,
        "kurtosis": kurt_t,
        "excess_kurtosis": kurt_t - 3.0,
    }


# ---------------------------------
# Skewed Generalized t (SGT) helpers
# ---------------------------------

def _sgt_m_v(sigma: float, lam: float, p: float, q: float) -> tuple[float, float]:
    """
    Compute m and v scaling constants for the SGT with mean=mu and var=sigma**2.
    """
    sigma = float(sigma)
    lam = float(lam)
    p = float(p)
    q = float(q)

    B1 = beta(1.0 / p, q)
    B2 = beta(2.0 / p, q - 1.0 / p)
    B3 = beta(3.0 / p, q - 2.0 / p)

    denom = (1.0 + 3.0 * lam * lam) * (B3 / B1) - 4.0 * (lam ** 2) * (B2 ** 2) / (B1 ** 2)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan, np.nan
    v = (q ** (-1.0 / p)) / np.sqrt(denom)
    m = lam * v * sigma * (2.0 * (q ** (1.0 / p)) * B2 / B1)
    return m, v


def sgt_pdf(x: np.ndarray, mu: float, sigma: float, lam: float, p: float, q: float) -> np.ndarray:
    """
    Skewed Generalized t probability density function.
    """
    x = np.asarray(x, dtype=float)
    mu = float(mu)
    sigma = float(sigma)
    lam = float(lam)
    p = float(p)
    q = float(q)

    m, v = _sgt_m_v(sigma, lam, p, q)
    if not np.isfinite(m) or not np.isfinite(v) or sigma <= 0.0 or abs(lam) >= 1.0 or p <= 0.0 or q <= 0.0:
        return np.full_like(x, np.nan, dtype=float)

    y = x - mu + m
    sgn_y = np.sign(y)
    B1 = beta(1.0 / p, q)
    v_sigma = v * sigma
    abs_term = np.abs(y) ** p
    skew_term = (1.0 + lam * sgn_y) ** p
    denom_term = q * (v_sigma ** p) * skew_term
    core = 1.0 + abs_term / denom_term

    const = (p / (2.0 * v_sigma * (q ** (1.0 / p)) * B1))
    pdf = const * (core ** (-(1.0 / p + q)))
    return pdf


def sgt_loglik(params: np.ndarray, x: np.ndarray) -> float:
    """
    Negative log-likelihood for the Skewed Generalized t distribution.
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = float(params[0])
    sigma = float(np.exp(params[1]))
    lam = float(np.tanh(params[2]))
    p = float(np.exp(params[3]))
    q = float(np.exp(params[4]))

    # Basic constraints
    if sigma <= 0.0 or abs(lam) >= 1.0 or p <= 0.0 or q <= 0.0:
        return 1e20
    if p * q <= 2.0:
        return 1e20

    pdf_vals = sgt_pdf(x, mu, sigma, lam, p, q)
    if np.isnan(pdf_vals).any():
        return 1e20

    pdf_clip = np.clip(pdf_vals, 1e-300, np.inf)
    nll = -np.sum(np.log(pdf_clip))

    # Penalty to discourage invalid high-order moments if optimizer wanders there.
    if p * q <= 4.0 or (q - 4.0 / p) <= 0.0:
        nll += 1e6
    return float(nll)


def fit_sgt_mle(x: np.ndarray, init: np.ndarray | None = None) -> dict:
    """
    Fit the Skewed Generalized t distribution by maximum likelihood.

    Parameters
    ----------
    x : np.ndarray
        1D array of observations.
    init : array-like, optional
        Initial guess (mu, sigma, lam, p, q) in constrained space.

    Returns
    -------
    dict with keys: mu, sigma, lam, p, q, success, fun, hess_inv.
    """
    x = np.asarray(x, dtype=float).ravel()
    if init is None:
        mu0 = float(np.mean(x))
        sigma0 = float(np.std(x, ddof=1))
        if sigma0 <= 0.0 or not np.isfinite(sigma0):
            sigma0 = 1.0
        lam0 = 0.0
        p0 = 2.0
        q0 = 5.0
    else:
        mu0, sigma0, lam0, p0, q0 = init
        sigma0 = float(sigma0 if sigma0 > 0 else 1.0)
        lam0 = float(np.clip(lam0, -0.99, 0.99))
        p0 = float(p0 if p0 > 0 else 2.0)
        q0 = float(q0 if q0 > 0 else 5.0)

    theta0 = np.array(
        [
            mu0,
            np.log(sigma0),
            np.arctanh(lam0),
            np.log(p0),
            np.log(q0),
        ],
        dtype=float,
    )

    res = minimize(
        sgt_loglik,
        theta0,
        args=(x,),
        method="L-BFGS-B",
    )

    mu_hat = float(res.x[0])
    sigma_hat = float(np.exp(res.x[1]))
    lam_hat = float(np.tanh(res.x[2]))
    p_hat = float(np.exp(res.x[3]))
    q_hat = float(np.exp(res.x[4]))

    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "lam": lam_hat,
        "p": p_hat,
        "q": q_hat,
        "success": bool(res.success),
        "fun": float(res.fun),
        "hess_inv": res.hess_inv,
    }


def sgt_moments(mu: float, sigma: float, lam: float, p: float, q: float) -> dict:
    """
    Analytic central moments for the Skewed Generalized t distribution.

    Returns
    -------
    dict with mean, var, m3, m4, skew, kurtosis, excess_kurtosis.
    """
    mu = float(mu)
    sigma = float(sigma)
    lam = float(lam)
    p = float(p)
    q = float(q)

    B1 = beta(1.0 / p, q)
    B2 = beta(2.0 / p, q - 1.0 / p)
    B3 = beta(3.0 / p, q - 2.0 / p)
    B4 = beta(4.0 / p, q - 3.0 / p)
    B5 = beta(5.0 / p, q - 4.0 / p)

    m_val, v_val = _sgt_m_v(sigma, lam, p, q)
    if not np.isfinite(m_val) or not np.isfinite(v_val):
        return {k: np.nan for k in ["mean", "var", "m3", "m4", "skew", "kurtosis", "excess_kurtosis"]}

    v_sigma = v_val * sigma
    mu2 = (v_sigma ** 2) * (q ** (2.0 / p)) * (
        (1.0 + 3.0 * lam * lam) * (B3 / B1)
        - 4.0 * (lam ** 2) * (B2 ** 2) / (B1 ** 2)
    )

    mu3 = (2.0 * (q ** (3.0 / p)) * lam * (v_sigma ** 3) / (B1 ** 3)) * (
        8.0 * (lam ** 2) * (B2 ** 3)
        - 3.0 * (1.0 + 3.0 * lam * lam) * B1 * B2 * B3
        + 2.0 * (1.0 + lam * lam) * (B1 ** 2) * B4
    )

    mu4 = ((q ** (4.0 / p)) * (v_sigma ** 4) / (B1 ** 4)) * (
        -48.0 * (lam ** 4) * (B2 ** 4)
        + 24.0 * (lam ** 2) * (1.0 + 3.0 * lam * lam) * B1 * (B2 ** 2) * B3
        - 32.0 * (lam ** 2) * (1.0 + lam * lam) * (B1 ** 2) * B2 * B4
        + (1.0 + 10.0 * lam * lam + 5.0 * (lam ** 4)) * (B1 ** 3) * B5
    )

    skew = mu3 / (mu2 ** 1.5) if mu2 > 0 else np.nan
    kurtosis = mu4 / (mu2 ** 2) if mu2 > 0 else np.nan

    return {
        "mean": mu,
        "var": mu2,
        "m3": mu3,
        "m4": mu4,
        "skew": skew,
        "kurtosis": kurtosis,
        "excess_kurtosis": kurtosis - 3.0 if np.isfinite(kurtosis) else np.nan,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # Quick Student-t check
    x = rng.standard_t(df=8.0, size=5000)
    res_t = fit_student_t_mle(x)
    mom_t = student_t_moments(res_t["mu"], res_t["sigma"], res_t["nu"])
    print("Student-t MLE:", res_t)
    print("Student-t moments:", mom_t)

    # Quick SGT moment check (lam=0 symmetric generalized t)
    mu0, sigma0, lam0, p0, q0 = 0.0, 1.0, 0.0, 2.0, 5.0
    mom_sgt = sgt_moments(mu0, sigma0, lam0, p0, q0)
    print("SGT moments (mu=0,sigma=1,lam=0,p=2,q=5):", mom_sgt)
