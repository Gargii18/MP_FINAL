"""Module 1 - Regression / trend analysis of IUCN Red List totals.

The original script fitted a cubic polynomial with scipy.optimize.curve_fit. A polynomial is linear in its
parameters, so ordinary least squares gives identical estimates (and identical standard errors) while being
faster and numerically more stable. This module also adds prediction intervals, alternative models and a
hold-out back-test so the reliability of the extrapolation can actually be assessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MODEL_KINDS = {
    "Linear": ("poly", 1),
    "Quadratic": ("poly", 2),
    "Cubic": ("poly", 3),
    "Exponential (log-linear)": ("exp", 1),
}
_COEF_NAMES = ["Intercept", "Linear", "Quadratic", "Cubic"]


@dataclass
class Fit:
    name: str
    kind: str
    deg: int
    beta: np.ndarray
    xtx_inv: np.ndarray
    s2: float
    dof: int
    n: int
    base_year: int
    r2: float
    adj_r2: float
    rse: float
    aic: float
    bic: float

    def _design(self, t):
        return np.vander(np.asarray(t, float), self.deg + 1, increasing=True)

    def fitted(self, years) -> np.ndarray:
        eta = self._design(np.asarray(years, float) - self.base_year) @ self.beta
        return np.exp(eta) if self.kind == "exp" else eta

    def predict(self, years, level: float = 0.95) -> pd.DataFrame:
        """Point forecast with a two-sided prediction interval at the given confidence level."""
        years = np.asarray(years, float)
        X = self._design(years - self.base_year)
        eta = X @ self.beta
        se = np.sqrt(self.s2 * (1.0 + np.einsum("ij,jk,ik->i", X, self.xtx_inv, X)))
        tcrit = stats.t.ppf(0.5 + level / 2.0, self.dof) if self.dof > 0 else np.nan
        lo, hi = eta - tcrit * se, eta + tcrit * se
        if self.kind == "exp":
            eta, lo, hi = np.exp(eta), np.exp(lo), np.exp(hi)
        return pd.DataFrame(
            {"Year": years.astype(int), "Predicted": eta, "Lower": lo, "Upper": hi}
        )


def fit_model(name: str, years, y) -> Fit:
    kind, deg = MODEL_KINDS[name]
    years = np.asarray(years, float)
    y = np.asarray(y, float)
    base = int(years.min())
    t = years - base
    z = np.log(y) if kind == "exp" else y
    X = np.vander(t, deg + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    n, k = X.shape
    dof = n - k
    resid_z = z - X @ beta
    s2 = float(resid_z @ resid_z / dof) if dof > 0 else np.nan
    yhat = np.exp(X @ beta) if kind == "exp" else X @ beta
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else np.nan
    rse = float(np.sqrt(rss / dof)) if dof > 0 else np.nan
    aic = n * np.log(rss / n) + 2 * (k + 1)
    bic = n * np.log(rss / n) + np.log(n) * (k + 1)
    return Fit(name, kind, deg, beta, np.linalg.pinv(X.T @ X), s2, dof, n, base, r2, adj, rse, aic, bic)


def coef_table(fit: Fit) -> pd.DataFrame:
    se = np.sqrt(np.diag(fit.s2 * fit.xtx_inv))
    t = fit.beta / se
    p = 2 * stats.t.sf(np.abs(t), fit.dof)
    if fit.kind == "poly":
        names = _COEF_NAMES[: fit.deg + 1]
    else:
        names = ["Intercept (log scale)", "Growth rate (log scale)"]
    return pd.DataFrame(
        {"Coefficient": names, "Estimate": fit.beta, "Std. error": se, "t value": t, "p value": p}
    )


def cagr(years, values) -> float:
    """Compound annual growth rate between the first and last point."""
    years = np.asarray(years, float)
    values = np.asarray(values, float)
    span = years[-1] - years[0]
    return float((values[-1] / values[0]) ** (1.0 / span) - 1.0)


def series(df: pd.DataFrame, column: str):
    d = df.dropna(subset=[column])
    return d["Year"].to_numpy(float), d[column].to_numpy(float)


def compare_models(years, y, holdout: int = 5) -> pd.DataFrame:
    """In-sample fit statistics plus a hold-out back-test.

    The back-test trains each model on the early years and forecasts the last `holdout` observations,
    which is a far more honest check of extrapolation quality than in-sample R-squared.
    """
    years = np.asarray(years, float)
    y = np.asarray(y, float)
    rows = []
    for name in MODEL_KINDS:
        full = fit_model(name, years, y)
        tr = fit_model(name, years[:-holdout], y[:-holdout])
        err = y[-holdout:] - tr.fitted(years[-holdout:])
        rows.append(
            {
                "Model": name,
                "R²": full.r2,
                "Adj. R²": full.adj_r2,
                "AIC": full.aic,
                "BIC": full.bic,
                "Back-test RMSE": float(np.sqrt(np.mean(err**2))),
                "Back-test MAPE %": float(np.mean(np.abs(err / y[-holdout:])) * 100),
            }
        )
    return pd.DataFrame(rows)
