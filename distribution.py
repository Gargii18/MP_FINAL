"""Module 2 - Distribution fitting for country-level threatened-species counts.

Improvements over the original script
-------------------------------------
* The Geometric distribution was estimated but never compared; it is now part of the AIC/BIC table.
* Poisson and Geometric MLEs are closed-form (lambda = mean, p = 1 / (1 + mean)).
* The Negative Binomial is fitted by profile likelihood over the size parameter `r`. For fixed r the MLE of the
  mean is the sample mean, so a robust 1-D bounded search replaces the fragile 2-D Nelder-Mead used before.
* A chi-square goodness-of-fit test and an observed-vs-expected table are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class DistFit:
    name: str
    params: dict
    k: int
    loglik: float
    n: int
    frozen: object = field(repr=False, default=None)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.k * np.log(self.n)


def describe(x) -> dict:
    x = np.asarray(x, float)
    mean, var = x.mean(), x.var(ddof=1)
    return {
        "Countries / territories": len(x),
        "Minimum": x.min(),
        "Median": float(np.median(x)),
        "Mean": mean,
        "Maximum": x.max(),
        "Std. deviation": float(np.sqrt(var)),
        "Variance": var,
        "Variance / Mean (dispersion index)": var / mean if mean > 0 else np.nan,
        "Skewness": float(stats.skew(x)),
        "Kurtosis (Pearson)": float(stats.kurtosis(x, fisher=False)),
    }


def fit_all(x) -> dict[str, DistFit]:
    """Fit Poisson, Geometric and Negative Binomial by maximum likelihood."""
    x = np.round(np.asarray(x, float)).astype(int)
    n = len(x)
    mean = float(x.mean())
    if mean <= 0:
        raise ValueError("Mean of the data is zero - cannot fit count distributions.")
    fits: dict[str, DistFit] = {}

    # Poisson
    pois = stats.poisson(mean)
    fits["Poisson"] = DistFit("Poisson", {"lambda": mean}, 1, float(pois.logpmf(x).sum()), n, pois)

    # Geometric on {0,1,2,...}: pmf = (1-p)^x p  ==  NegBin(size=1, p)
    p_geo = 1.0 / (1.0 + mean)
    geo = stats.nbinom(1, p_geo)
    fits["Geometric"] = DistFit("Geometric", {"p": p_geo}, 1, float(geo.logpmf(x).sum()), n, geo)

    # Negative binomial (profile likelihood over log r, mean fixed at sample mean)
    def nll(log_r: float) -> float:
        r = np.exp(log_r)
        return -float(stats.nbinom.logpmf(x, r, r / (r + mean)).sum())

    res = minimize_scalar(nll, bounds=(-6.0, 14.0), method="bounded")
    r_hat = float(np.exp(res.x))
    p_nb = r_hat / (r_hat + mean)
    nb = stats.nbinom(r_hat, p_nb)
    fits["Negative Binomial"] = DistFit(
        "Negative Binomial",
        {"size r": r_hat, "p": p_nb, "mean": r_hat * (1 - p_nb) / p_nb},
        2,
        float(nb.logpmf(x).sum()),
        n,
        nb,
    )
    return fits


def _interval_probs(frozen, edges: np.ndarray) -> np.ndarray:
    """P(edges[i] <= X < edges[i+1]) for integer-valued X; last edge may be inf."""
    cdf = np.array([frozen.cdf(e - 1) if np.isfinite(e) else 1.0 for e in edges])
    return np.diff(cdf)


def _default_edges(x: np.ndarray) -> np.ndarray:
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    edges = [e for e in edges if e <= x.max()] + [np.inf]
    return np.array(edges, float)


def _bin_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isinf(b):
            labels.append(f"{int(a)}+")
        elif b - a == 1:
            labels.append(f"{int(a)}")
        else:
            labels.append(f"{int(a)}–{int(b) - 1}")
    return labels


def observed_vs_expected(x, fits: dict[str, DistFit]) -> pd.DataFrame:
    """Observed counts per log-spaced bin and the counts expected under each fitted distribution."""
    x = np.round(np.asarray(x, float))
    edges = _default_edges(x)
    upper = np.where(np.isinf(edges[1:]), x.max() + 1, edges[1:])
    obs, _ = np.histogram(x, bins=np.append(edges[:-1], upper[-1]))
    out = pd.DataFrame({"Bin": _bin_labels(edges), "Observed": obs})
    for name, f in fits.items():
        out[name] = len(x) * _interval_probs(f.frozen, edges)
    return out


def chi_square_gof(x, fit: DistFit, max_bins: int = 10, min_expected: float = 5.0) -> dict:
    """Pearson chi-square goodness-of-fit using (approximately) equiprobable bins under the FITTED distribution.

    Bin edges are quantiles of the fitted model, so every bin has a healthy expected count and a model that puts
    almost no mass where the data actually sit (e.g. Poisson on over-dispersed counts) is penalised heavily.
    """
    x = np.round(np.asarray(x, float))
    n = len(x)
    m = int(max(3, min(max_bins, n // min_expected)))
    cuts = np.unique(fit.frozen.ppf(np.linspace(0, 1, m + 1)[1:-1])).astype(float)
    bounds = np.concatenate([[-np.inf], cuts, [np.inf]])
    E = n * np.diff(fit.frozen.cdf(bounds))
    O = np.bincount(np.searchsorted(cuts, x, side="left"), minlength=len(E)).astype(float)
    # merge any bin whose expected count is still below the threshold into its neighbour
    Om, Em, o_acc, e_acc = [], [], 0.0, 0.0
    for o, e in zip(O, E):
        o_acc += o
        e_acc += e
        if e_acc >= min_expected:
            Om.append(o_acc)
            Em.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0 or o_acc > 0:
        if Em:
            Om[-1] += o_acc
            Em[-1] += e_acc
        else:
            Om, Em = [o_acc], [e_acc]
    Om, Em = np.array(Om), np.array(Em)
    chi2 = float(np.sum((Om - Em) ** 2 / Em))
    dof = len(Om) - 1 - fit.k
    pval = float(stats.chi2.sf(chi2, dof)) if dof >= 1 else np.nan
    return {"chi2": chi2, "dof": dof, "p_value": pval}


def comparison_table(x, fits: dict[str, DistFit]) -> pd.DataFrame:
    rows = []
    for name, f in fits.items():
        gof = chi_square_gof(x, f)
        rows.append(
            {
                "Distribution": name,
                "Parameters": ", ".join(f"{k} = {v:.4g}" for k, v in f.params.items()),
                "Log-likelihood": f.loglik,
                "AIC": f.aic,
                "BIC": f.bic,
                "χ² (GOF)": gof["chi2"],
                "df": gof["dof"],
                "p-value": gof["p_value"],
            }
        )
    out = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    out.insert(1, "ΔAIC", out["AIC"] - out["AIC"].min())
    return out
