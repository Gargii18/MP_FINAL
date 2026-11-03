"""Module 3 - Markov chain analysis of IUCN status transitions.

Key point about the data
------------------------
The input file lists only species whose category CHANGED between 2021 and 2022. The raw transition matrix is therefore
"conditional on a change": every diagonal entry is zero (except absorbing/empty rows). The original script's eigenvector
"steady state" of that matrix is not meaningful (with EX absorbing it is trivially 100% EX, and when several eigenvalues
equal 1 the chosen eigenvector is arbitrary). This module replaces it with:

* an absorbing-chain analysis (probability of eventual extinction, expected number of status changes before extinction), and
* an optional "annual change rate" assumption c that turns the change-only matrix into a full one-year matrix
  P_year = (1 - c) * I + c * P_change, so multi-year projections become possible. c is an ASSUMPTION supplied by the user
  (e.g. the share of assessed species that changed category in a typical year) and is not estimated from the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import RANK, STATES


def transition_counts(species: pd.DataFrame, states: list[str] = STATES) -> pd.DataFrame:
    tc = pd.crosstab(species["Status_2021"], species["Status_2022"])
    return tc.reindex(index=states, columns=states, fill_value=0)


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise counts. Rows with no observations become self-loops (absorbing)."""
    P = np.array(counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0), dtype=float)  # writable copy
    for i in range(P.shape[0]):
        if P[i].sum() == 0:
            P[i, i] = 1.0
    return pd.DataFrame(P, index=counts.index, columns=counts.columns)


def direction_summary(species: pd.DataFrame) -> pd.Series:
    a = species["Status_2021"].map(RANK)
    b = species["Status_2022"].map(RANK)
    direction = np.where(b > a, "Worsened", np.where(b < a, "Improved", "Unchanged"))
    return pd.Series(direction).value_counts()


def with_annual_change_rate(P: pd.DataFrame, c: float) -> pd.DataFrame:
    """P_year = (1 - c) I + c P_change  (rows that are absorbing stay absorbing)."""
    n = P.shape[0]
    M = (1 - c) * np.eye(n) + c * P.to_numpy()
    return pd.DataFrame(M, index=P.index, columns=P.columns)


def project(P: pd.DataFrame, start: pd.Series, steps: int) -> pd.DataFrame:
    """Distribution over states after 0..steps transitions, starting from `start` (probabilities)."""
    v = start.reindex(P.index).fillna(0.0).to_numpy(float)
    v = v / v.sum() if v.sum() > 0 else v
    M = P.to_numpy()
    rows = [v.copy()]
    for _ in range(steps):
        v = v @ M
        rows.append(v.copy())
    return pd.DataFrame(rows, columns=P.index).rename_axis("Step").reset_index()


def absorbing_analysis(P: pd.DataFrame) -> dict:
    """Absorption probabilities and expected steps to absorption for every transient state.

    Returns an empty dict if the chain has no absorbing state.
    """
    M = P.to_numpy()
    states = list(P.index)
    absorbing = [i for i in range(len(states)) if np.isclose(M[i, i], 1.0)]
    transient = [i for i in range(len(states)) if i not in absorbing]
    if not absorbing or not transient:
        return {}
    Q = M[np.ix_(transient, transient)]
    R = M[np.ix_(transient, absorbing)]
    try:
        N = np.linalg.inv(np.eye(len(transient)) - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(np.eye(len(transient)) - Q)
    B = N @ R
    steps = N.sum(axis=1)
    # Transient states that cannot reach any absorbing state would have inf steps; flag them
    reach = B.sum(axis=1)
    return {
        "absorbing": [states[i] for i in absorbing],
        "absorption_prob": pd.DataFrame(B, index=[states[i] for i in transient], columns=[states[i] for i in absorbing]),
        "expected_steps": pd.Series(np.where(reach > 0.999, steps, np.inf), index=[states[i] for i in transient]),
    }


def stationary_distribution(P: pd.DataFrame) -> pd.Series | None:
    """Unique stationary distribution if the chain is irreducible, else None."""
    M = P.to_numpy()
    n = M.shape[0]
    A = np.vstack([M.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.all(sol > -1e-9) and np.allclose(A @ sol, b, atol=1e-8) and np.linalg.matrix_rank(M - np.eye(n)) == n - 1:
        return pd.Series(np.clip(sol, 0, None), index=P.index)
    return None


def sankey_data(counts: pd.DataFrame, states: list[str] = STATES):
    labels = [f"{s} (2021)" for s in states] + [f"{s} (2022)" for s in states]
    src, tgt, val = [], [], []
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            v = int(counts.loc[a, b])
            if v > 0:
                src.append(i)
                tgt.append(len(states) + j)
                val.append(v)
    return labels, src, tgt, val
