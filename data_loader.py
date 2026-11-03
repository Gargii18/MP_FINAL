"""Data loading / parsing for the two user datasets.

Dataset A  - Distribution_Data.csv           (country-level threatened-species counts: columns `Name`, `Total`)
Dataset B  - Markov_Random_Forest (1).xlsx   (IUCN Red List category changes 2021 -> 2022, with taxonomy header rows)

Both parsers are deliberately forgiving about column-name capitalisation / spacing and return clean tables plus a
short data-quality report that the app shows to the user.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import STATES

_FULL_NAMES = {
    "least concern": "LC",
    "near threatened": "NT",
    "vulnerable": "VU",
    "endangered": "EN",
    "critically endangered": "CR",
    "extinct": "EX",
    "extinct in the wild": "EW",
    "data deficient": "DD",
}


# The status-change workbook continues across sheets: only the first sheet has a header row, and the group heading
# (reptiles) is missing at the top of "Table 2". Rows there are filed under this heading so they are not mislabelled.
MISSING_HEADINGS = {"table 2": "REPTILES (Reptilia)"}


def _read_workbook(buf) -> pd.DataFrame:
    """Read every sheet of an Excel workbook as one table (later sheets reuse the first sheet's header)."""
    sheets = pd.read_excel(buf, sheet_name=None, header=None)
    names = list(sheets)
    if len(names) == 1:
        buf.seek(0)
        return pd.read_excel(buf)
    first = sheets[names[0]]
    header = [str(c).strip() if pd.notna(c) else f"Unnamed: {i}" for i, c in enumerate(first.iloc[0])]
    named = [h for h in header if not h.startswith("Unnamed")]
    head = first.iloc[1:].copy()
    head.columns = header
    frames = [head]
    for n in names[1:]:
        d = sheets[n].dropna(how="all")
        if d.empty:
            continue
        if str(d.iloc[0, 0]).strip().lower().startswith("scientific"):
            d = d.iloc[1:]
        cols = named if d.shape[1] == len(named) else header[: d.shape[1]]
        d = d.iloc[:, : len(cols)].copy()
        d.columns = cols
        key = str(n).strip().lower()
        if key in MISSING_HEADINGS and pd.notna(d.iloc[0, 2]):  # first row is a species, not a group heading
            d = pd.concat([pd.DataFrame([[MISSING_HEADINGS[key]] + [np.nan] * (len(cols) - 1)], columns=cols), d])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def read_any(data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file from raw bytes."""
    name = filename.lower()
    buf = io.BytesIO(data)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return _read_workbook(buf)
    try:
        return pd.read_csv(buf)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, encoding="latin-1")


def find_file(folder: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    for c in df.columns:
        low = str(c).lower()
        if all(k in low for k in keywords):
            return c
    return None


def normalise_status(s: pd.Series) -> pd.Series:
    """'Endangered' / ' en ' / 'CR (PE)' -> 'EN' / 'EN' / 'CR'. Unrecognised values become NaN."""
    s = s.astype("string").str.strip()
    low = s.str.lower()
    mapped = low.map(_FULL_NAMES)
    code = s.str.upper().str.extract(r"^([A-Z]{2})\b", expand=False)
    out = mapped.fillna(code)
    return out.astype(object).where(out.notna(), np.nan)


# ---------------------------------------------------------------------------
# Dataset A: distribution of threatened species across countries
# ---------------------------------------------------------------------------
def parse_distribution(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    c_total = next((c for c in df.columns if c.lower() == "total"), None) or _find_col(df, "total")
    if c_total is None:
        raise ValueError("Could not find a 'Total' column in the distribution file.")
    c_name = next((c for c in df.columns if c.lower() == "name"), None) or _find_col(df, "name") or df.columns[0]

    out = pd.DataFrame(
        {
            "Name": df[c_name].astype(str).str.strip(),
            "Total": pd.to_numeric(df[c_total].astype(str).str.replace(",", "", regex=False), errors="coerce"),
        }
    )
    n_raw = len(out)
    out = out.dropna(subset=["Total"])
    # drop aggregate rows that would distort a country-level distribution
    agg = out["Name"].str.lower().isin({"total", "world", "global", "all", "grand total"})
    out = out[~agg & (out["Total"] >= 0)].copy()
    out["Total"] = out["Total"].round().astype(int)
    return out.reset_index(drop=True), {"rows_in_file": n_raw, "rows_used": len(out), "rows_dropped": n_raw - len(out)}


# ---------------------------------------------------------------------------
# Dataset B: IUCN status changes 2021 -> 2022
# ---------------------------------------------------------------------------
def parse_status_changes(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the raw Red List 'category changes' sheet into one clean row per species.

    The sheet interleaves taxonomy header rows (e.g. 'MAMMALIA') with species rows. A header row has a scientific
    name but no common name and no status values. (The original script used only 'no common name', which would
    silently mistake any species without a common name for a header and drop it.)
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    c_sci = _find_col(df, "scientific")
    c_com = _find_col(df, "common")
    c21 = _find_col(df, "2021")
    c22 = _find_col(df, "2022")
    c_reason = _find_col(df, "reason")
    missing = [n for n, c in [("Scientific name", c_sci), ("IUCN status 2021", c21), ("IUCN status 2022", c22)] if c is None]
    if missing:
        raise ValueError(
            "Status-change file is missing: " + ", ".join(missing)
            + ". Expected columns like 'Scientific name', 'IUCN Red List (2021)', 'IUCN Red List (2022)'."
        )

    s21_raw, s22_raw = df[c21], df[c22]
    is_header = df[c_sci].notna() & s21_raw.isna() & s22_raw.isna()
    if c_com is not None:
        is_header &= df[c_com].isna()

    taxonomy = df[c_sci].where(is_header).ffill().fillna("Unknown")
    species = pd.DataFrame(
        {
            "Scientific name": df[c_sci],
            "Common name": df[c_com] if c_com else np.nan,
            "Taxonomy": taxonomy.astype(str).str.strip(),
            "Status_2021": normalise_status(s21_raw),
            "Status_2022": normalise_status(s22_raw),
            "Reason for change": df[c_reason] if c_reason else np.nan,
        }
    )[~is_header & df[c_sci].notna()]

    n_species = len(species)
    valid = species["Status_2021"].isin(STATES) & species["Status_2022"].isin(STATES)
    species_clean = species[valid].reset_index(drop=True)
    quality = {
        "taxonomy_header_rows": int(is_header.sum()),
        "species_rows": n_species,
        "rows_used": len(species_clean),
        "rows_dropped_unsupported_status": n_species - len(species_clean),
        "taxonomic_groups": int(species_clean["Taxonomy"].nunique()),
        "unchanged_rows": int((species_clean["Status_2021"] == species_clean["Status_2022"]).sum()),
    }
    return species_clean, quality
