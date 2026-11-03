"""Shared constants: IUCN status order, colours and the historical Red List totals."""
import numpy as np
import pandas as pd

# Ordered from least to most severe (same order as the original script)
STATES = ["LC", "NT", "VU", "EN", "CR", "EX"]
RANK = {s: i for i, s in enumerate(STATES)}

STATUS_LABELS = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
    "EX": "Extinct",
}

STATUS_COLORS = {
    "LC": "#4c9f50",
    "NT": "#a6c34a",
    "VU": "#f2c230",
    "EN": "#f28c28",
    "CR": "#d64545",
    "EX": "#4a4a4a",
}

# Report-style 3-tier risk grouping (Low / Medium / High), matching the report's Table 4
RISK_TIER = {"LC": "Low", "NT": "Low", "VU": "Medium", "EN": "High", "CR": "High", "EX": "High"}

# ---------------------------------------------------------------------------
# Historical IUCN Red List totals (taken unchanged from analysis_and_prediction.py)
# ---------------------------------------------------------------------------
_YEARS = [
    1996, 2000, 2002, 2003, 2004, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
]
_ASSESSED = [
    np.nan, 16507, 16697, 22424, 38046, 40174, 41415, 44838, 47677, 55926, 61914,
    65518, 71576, 76199, 79837, 85604, 91523, 96951, 112432, 128918, 142577,
    150388, 157190, 166061, 172620,
]
_THREATENED = [
    10533, 11046, 11167, 12259, 15503, 16116, 16306, 16928, 17291, 18351, 19570,
    20219, 21286, 22413, 23250, 24307, 25821, 26840, 30178, 35765, 40084, 42108,
    44016, 46337, 48646,
]


def history_df() -> pd.DataFrame:
    """Return the IUCN Red List history table (Year, Total_Assessed, Total_Threatened, Ratio)."""
    df = pd.DataFrame(
        {"Year": _YEARS, "Total_Assessed": _ASSESSED, "Total_Threatened": _THREATENED}
    )
    df["Ratio"] = df["Total_Threatened"] / df["Total_Assessed"]
    return df
