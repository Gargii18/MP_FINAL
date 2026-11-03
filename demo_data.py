"""Synthetic DEMO data.

These generators exist only so the app runs end-to-end on a fresh deployment (e.g. before the real project datasets
are added). The numbers are random draws that mimic the *shape* of the real data - they are NOT real IUCN records
and must never be quoted as findings. The app labels them clearly whenever they are in use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import RANK, STATES

# taxonomic group -> (relative size, tilt applied to "worsening" transitions; >1 means more likely to worsen)
_TAXA = {
    "MAMMALIA": (0.10, 1.0),
    "AVES": (0.18, 0.7),
    "REPTILIA": (0.10, 1.3),
    "AMPHIBIA": (0.12, 1.6),
    "ACTINOPTERYGII": (0.14, 1.2),
    "INSECTA": (0.08, 1.1),
    "ANTHOZOA": (0.05, 1.5),
    "MAGNOLIOPSIDA": (0.15, 1.4),
    "POLYPODIOPSIDA": (0.04, 1.2),
    "GASTROPODA": (0.04, 1.3),
}

# P(status 2022 | status 2021, a change occurred)  - self-transitions are impossible by construction
_BASE = {
    "LC": {"NT": 0.60, "VU": 0.30, "EN": 0.07, "CR": 0.03},
    "NT": {"LC": 0.45, "VU": 0.40, "EN": 0.10, "CR": 0.05},
    "VU": {"LC": 0.15, "NT": 0.25, "EN": 0.45, "CR": 0.13, "EX": 0.02},
    "EN": {"LC": 0.02, "NT": 0.05, "VU": 0.30, "CR": 0.60, "EX": 0.03},
    "CR": {"VU": 0.20, "EN": 0.55, "EX": 0.25},
    "EX": {"CR": 1.0},
}
_START_WEIGHTS = {"LC": 0.30, "NT": 0.14, "VU": 0.18, "EN": 0.17, "CR": 0.20, "EX": 0.0}


def demo_distribution(seed: int = 7, n: int = 240) -> pd.DataFrame:
    """Heavy-tailed, over-dispersed counts (gamma-Poisson) plus a handful of 'hotspot' regions."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(shape=0.55, scale=70.0, size=n))
    counts[:6] = rng.integers(300, 1100, size=6)
    rng.shuffle(counts)
    return pd.DataFrame({"Name": [f"Demo Region {i + 1:03d}" for i in range(n)], "Total": counts})


def demo_status_changes(seed: int = 11, n: int = 1200) -> pd.DataFrame:
    """Species whose Red List category changed between 2021 and 2022 (parsed schema, one row per species)."""
    rng = np.random.default_rng(seed)
    taxa = list(_TAXA)
    t_prob = np.array([_TAXA[t][0] for t in taxa])
    t_prob = t_prob / t_prob.sum()
    s_prob = np.array([_START_WEIGHTS[s] for s in STATES])
    s_prob = s_prob / s_prob.sum()

    rows = []
    for i in range(n):
        tax = rng.choice(taxa, p=t_prob)
        s21 = rng.choice(STATES, p=s_prob)
        targets = list(_BASE[s21])
        w = np.array([_BASE[s21][t] * (_TAXA[tax][1] if RANK[t] > RANK[s21] else 1.0) for t in targets])
        s22 = rng.choice(targets, p=w / w.sum())
        genuine = rng.random() < (0.35 if RANK[s22] < RANK[s21] else 0.55)
        rows.append(
            {
                "Scientific name": f"Demo species {i + 1:04d}",
                "Common name": f"Demo {tax.title()} {i + 1}",
                "Taxonomy": tax,
                "Status_2021": s21,
                "Status_2022": s22,
                "Reason for change": "Genuine change" if genuine else "Non-genuine change",
            }
        )
    return pd.DataFrame(rows)
