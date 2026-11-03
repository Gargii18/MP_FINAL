"""Biodiversity Loss Analysis & IUCN Status Prediction System  -  Streamlit prototype.

Run locally:   streamlit run app.py
Integrates the four modules of the project report / analysis script in one web app:
  1. Regression & forecasting of IUCN Red List totals
  2. Distribution fitting of threatened-species counts across countries
  3. Markov-chain analysis of IUCN status transitions
  4. Random-Forest prediction of the next IUCN status (+ algorithm benchmark and batch scoring)
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import classifier as clf
from core import distribution as dist
from core import markov as mk
from core import regression as reg
from core.constants import RANK, STATES, STATUS_COLORS, STATUS_LABELS, history_df
from core.data_loader import find_file, parse_distribution, parse_status_changes, read_any
from core.demo_data import demo_distribution, demo_status_changes

warnings.filterwarnings("ignore")

APP_TITLE = "Biodiversity Loss Analysis & IUCN Status Prediction"
DATA_DIR = Path(__file__).parent / "data"
GREEN = "#2e7d32"

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .badge {display:inline-block; padding:0.25rem 0.8rem; border-radius:999px; color:white; font-weight:700;}
      .result-card {border:1px solid rgba(128,128,128,.35); border-radius:14px; padding:1.1rem 1.3rem;}
      .small-note {font-size:0.85rem; opacity:0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small UI helpers (kept compatible with old and new Streamlit versions)
# ---------------------------------------------------------------------------
def show_chart(fig: go.Figure, key: str | None = None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def badge(status: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLORS[status]}">{status} · {STATUS_LABELS[status]}</span>'


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------------------
# Data resolution: upload  ->  file in ./data  ->  synthetic demo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _parse_dist(raw_bytes: bytes, name: str):
    return parse_distribution(read_any(raw_bytes, name))


@st.cache_data(show_spinner=False)
def _parse_status(raw_bytes: bytes, name: str):
    return parse_status_changes(read_any(raw_bytes, name))


@st.cache_data(show_spinner=False)
def _demo_dist():
    return demo_distribution()


@st.cache_data(show_spinner=False)
def _demo_status():
    return demo_status_changes()


def resolve_distribution(upload):
    """Return (dataframe, quality, source_label, is_demo)."""
    try:
        if upload is not None:
            df, q = _parse_dist(upload.getvalue(), upload.name)
            return df, q, f"Uploaded: {upload.name}", False
        f = find_file(DATA_DIR, ["Distribution*.csv", "distribution*.csv", "Distribution*.xlsx"])
        if f is not None:
            df, q = _parse_dist(f.read_bytes(), f.name)
            return df, q, f"Repository file: {f.name}", False
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read distribution data: {e}\n\nFalling back to demo data.")
    df = _demo_dist()
    return df, {"rows_in_file": len(df), "rows_used": len(df), "rows_dropped": 0}, "Synthetic DEMO data", True


def resolve_status(upload):
    try:
        if upload is not None:
            df, q = _parse_status(upload.getvalue(), upload.name)
            return df, q, f"Uploaded: {upload.name}", False
        f = find_file(DATA_DIR, ["Markov*.xlsx", "markov*.xlsx", "Status*.xlsx", "Markov*.csv"])
        if f is not None:
            df, q = _parse_status(f.read_bytes(), f.name)
            return df, q, f"Repository file: {f.name}", False
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read status-change data: {e}\n\nFalling back to demo data.")
    df = _demo_status()
    q = {
        "taxonomy_header_rows": 0, "species_rows": len(df), "rows_used": len(df),
        "rows_dropped_unsupported_status": 0, "taxonomic_groups": df["Taxonomy"].nunique(),
        "unchanged_rows": int((df["Status_2021"] == df["Status_2022"]).sum()),
    }
    return df, q, "Synthetic DEMO data", True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Overview",
    "📈 1 · Trends & Forecast",
    "📊 2 · Distribution Fitting",
    "🔗 3 · Markov Chain",
    "🌳 4 · Status Predictor",
    "🗂️ Data & Pipeline",
    "ℹ️ About & Method",
]

with st.sidebar:
    st.markdown(f"### 🌿 {APP_TITLE}")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Datasets**")
    up_dist = st.file_uploader("Distribution data (Name, Total)", type=["csv", "xlsx"], key="up_dist")
    up_stat = st.file_uploader("Status-change data (IUCN 2021 → 2022)", type=["xlsx", "csv"], key="up_stat")

dist_df, dist_q, dist_src, dist_demo = resolve_distribution(up_dist)
status_df, status_q, status_src, status_demo = resolve_status(up_stat)

with st.sidebar:
    st.caption(f"Distribution: **{dist_src}**")
    st.caption(f"Status changes: **{status_src}**")
    if dist_demo or status_demo:
        st.warning("Demo data in use: results in modules 2–4 are illustrative, not real findings.", icon="⚠️")
    st.divider()
    st.caption(
        "Project team (per report): Gargi Gupta, Sakshi Amrutkar, Diya Mokal, Himanshi Motghare. "
        "Guide: Dr. Dipa Dharmadhikari · MIT Chhatrapati Sambhajinagar · 2025-26"
    )


def demo_banner(is_demo: bool, what: str):
    if is_demo:
        st.warning(
            f"**Synthetic demo data** is being used for the {what}. Upload your real file in the sidebar "
            f"(or add it to the repository's `data/` folder) to see real results.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Editable Red List history (shared by Overview and Module 1)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = history_df()


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d[["Year", "Total_Assessed", "Total_Threatened"]]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Year"]).drop_duplicates("Year", keep="last").sort_values("Year").reset_index(drop=True)
    d["Year"] = d["Year"].astype(int)
    d["Ratio"] = d["Total_Threatened"] / d["Total_Assessed"]
    return d


# ===========================================================================
# PAGE: Overview
# ===========================================================================
def page_overview():
    st.title("🌿 Biodiversity Loss Analysis & IUCN Status Prediction")
    st.caption(
        "Prototype integrating the project *Biodiversity Loss Analysis and Species Extinction Prediction System*: "
        "statistical analysis + machine learning on IUCN Red List data."
    )
    demo_banner(dist_demo or status_demo, "distribution / Markov / Random-Forest modules")

    h = clean_history(st.session_state["history"])
    last = h.dropna(subset=["Total_Threatened"]).iloc[-1]
    prev = h.dropna(subset=["Total_Threatened"]).iloc[-2]
    c = st.columns(4)
    c[0].metric(f"Species assessed ({int(last.Year)})", f"{int(last.Total_Assessed):,}", f"{(last.Total_Assessed / prev.Total_Assessed - 1) * 100:+.1f}% YoY")
    c[1].metric(f"Threatened ({int(last.Year)})", f"{int(last.Total_Threatened):,}", f"{(last.Total_Threatened / prev.Total_Threatened - 1) * 100:+.1f}% YoY")
    c[2].metric("Threatened share of assessed", pct(last.Ratio))
    yrs, thr = reg.series(h[h.Year >= 2015], "Total_Threatened")
    c[3].metric("Threatened CAGR (2015 →)", pct(reg.cagr(yrs, thr)))

    st.subheader("What this app does")
    cols = st.columns(4)
    cards = [
        ("📈 1 · Trends & Forecast", "Cubic (and alternative) regression on IUCN Red List totals, with prediction intervals and a hold-out back-test."),
        ("📊 2 · Distribution Fitting", "Poisson, Geometric and Negative Binomial fits to country-level threatened-species counts."),
        ("🔗 3 · Markov Chain", "Transition matrix of Red List category changes, extinction-pathway analysis and projections."),
        ("🌳 4 · Status Predictor", "Random Forest that predicts a species' next IUCN status, plus benchmark and batch scoring."),
    ]
    for col, (t, d) in zip(cols, cards):
        with col:
            st.markdown(f"**{t}**")
            st.write(d)

    st.subheader("Red List coverage over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.Year, y=h.Total_Assessed, name="Assessed", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=h.Year, y=h.Total_Threatened, name="Threatened", mode="lines+markers"))
    fig.update_layout(yaxis_title="Number of species", height=380, legend=dict(orientation="h", y=1.1))
    show_chart(fig, "ov_hist")
    st.info(
        "**Reading this chart:** the *share* of assessed species that is threatened has fallen since 2000, while the "
        "*absolute* number of threatened species keeps rising. Both are heavily influenced by assessment effort - the "
        "Red List has been expanded to many more species - so growth in totals is not the same as growth in the true "
        "extinction rate.",
        icon="💡",
    )


# ===========================================================================
# PAGE: Module 1 - Regression & forecast
# ===========================================================================
def page_trends():
    st.title("📈 Module 1 · Trends & Forecast")
    st.caption("Regression analysis of IUCN Red List totals (1996–latest), with uncertainty and back-testing.")

    with st.expander("➕ Add or edit Red List data points (e.g. newer annual releases)"):
        st.caption("Edit values or add a row for a new year; all charts and forecasts below update immediately.")
        edited = st.data_editor(
            st.session_state["history"][["Year", "Total_Assessed", "Total_Threatened"]],
            num_rows="dynamic", key="hist_editor", hide_index=True,
        )
        c1, c2 = st.columns([1, 5])
        if c1.button("Apply"):
            st.session_state["history"] = clean_history(edited)
            st.rerun()
        if c2.button("Reset to original data"):
            st.session_state["history"] = history_df()
            st.rerun()

    h = clean_history(st.session_state["history"])
    if h["Total_Threatened"].notna().sum() < 10:
        st.error("Need at least 10 yearly observations to fit and back-test the models.")
        return

    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1])
    target_label = c1.selectbox("Series", ["Total Threatened", "Total Assessed"])
    model_name = c2.selectbox("Model", list(reg.MODEL_KINDS), index=2, help="The report/script used the cubic model.")
    horizon = c3.slider("Forecast horizon (years)", 1, 10, 6)
    level = c4.slider("Interval level", 0.80, 0.99, 0.95, 0.01)
    col = "Total_Threatened" if target_label == "Total Threatened" else "Total_Assessed"

    years, vals = reg.series(h, col)
    fit = reg.fit_model(model_name, years, vals)
    last_year = int(years.max())
    fy = np.arange(last_year + 1, last_year + horizon + 1)
    pred = fit.predict(fy, level)

    t_hist, t_fit, t_fc, t_cmp = st.tabs(["History & growth", "Model fit", "Forecast", "Model comparison"])

    with t_hist:
        log = st.checkbox("Log scale", value=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.Year, y=h.Total_Assessed, name="Assessed", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=h.Year, y=h.Total_Threatened, name="Threatened", mode="lines+markers"))
        fig.update_layout(title="IUCN Red List assessment trends", yaxis_title="Species", height=400, yaxis_type="log" if log else "linear", legend=dict(orientation="h", y=1.1))
        show_chart(fig, "tr_hist")
        fig2 = px.line(h.dropna(subset=["Ratio"]), x="Year", y="Ratio", markers=True, title="Threatened / assessed ratio")
        fig2.update_layout(height=300, yaxis_tickformat=".0%")
        show_chart(fig2, "tr_ratio")
        rows = []
        for lab, sub in [("Full period", h.dropna(subset=["Total_Assessed"])), ("2015 → latest", h[h.Year >= 2015].dropna(subset=["Total_Assessed"]))]:
            ya, va = reg.series(sub, "Total_Assessed")
            yt, vt = reg.series(sub, "Total_Threatened")
            rows.append({"Period": f"{lab} ({int(ya.min())}–{int(ya.max())})", "Assessed CAGR": pct(reg.cagr(ya, va), 2), "Threatened CAGR": pct(reg.cagr(yt, vt), 2)})
        st.markdown("**Compound annual growth rate (CAGR)**")
        show_df(pd.DataFrame(rows))

    with t_fit:
        m = st.columns(4)
        m[0].metric("R²", f"{fit.r2:.4f}")
        m[1].metric("Adjusted R²", f"{fit.adj_r2:.4f}")
        m[2].metric("Residual std. error", f"{fit.rse:,.1f}")
        m[3].metric("Observations", fit.n)
        ct = reg.coef_table(fit)
        show_df(ct.round(4))
        fitted = fit.fitted(years)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=vals, mode="markers", name="Observed"))
        grid = np.linspace(years.min(), years.max(), 200)
        fig.add_trace(go.Scatter(x=grid, y=fit.fitted(grid), mode="lines", name=f"{model_name} fit"))
        fig.update_layout(title=f"{target_label}: observed vs fitted", height=380, legend=dict(orientation="h", y=1.1))
        show_chart(fig, "tr_fit")
        fr = px.bar(x=years, y=vals - fitted, labels={"x": "Year", "y": "Residual"}, title="Residuals (observed − fitted)")
        fr.update_layout(height=260)
        show_chart(fr, "tr_resid")
        st.caption("Patterned residuals (runs of same sign) indicate the functional form does not fully capture the trend.")

    with t_fc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[fy, fy[::-1]], y=np.r_[pred.Upper, pred.Lower[::-1]], fill="toself", fillcolor="rgba(46,125,50,0.15)", line=dict(width=0), name=f"{int(level * 100)}% prediction interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=vals, mode="markers+lines", name="Observed"))
        fig.add_trace(go.Scatter(x=np.r_[years[-1], fy], y=np.r_[vals[-1], pred.Predicted], mode="lines+markers", line=dict(dash="dash", color=GREEN), name="Forecast"))
        fig.update_layout(title=f"{target_label}: {horizon}-year forecast ({model_name})", height=420, legend=dict(orientation="h", y=1.1))
        show_chart(fig, "tr_fc")

        out = pred.copy()
        out["YoY growth"] = out["Predicted"].pct_change()
        out.loc[out.index[0], "YoY growth"] = out["Predicted"].iloc[0] / vals[-1] - 1
        # implied threatened share using the same model family for both series
        ya, va = reg.series(h, "Total_Assessed")
        yt, vt = reg.series(h, "Total_Threatened")
        fa, ft = reg.fit_model(model_name, ya, va), reg.fit_model(model_name, yt, vt)
        out["Implied threatened share"] = ft.predict(fy, level)["Predicted"].to_numpy() / fa.predict(fy, level)["Predicted"].to_numpy()
        disp = out.copy()
        for cc in ["Predicted", "Lower", "Upper"]:
            disp[cc] = disp[cc].round(0).astype(int).clip(lower=0)
        disp["YoY growth"] = disp["YoY growth"].map(lambda v: f"{v * 100:+.1f}%")
        disp["Implied threatened share"] = disp["Implied threatened share"].map(lambda v: f"{v * 100:.1f}%")
        show_df(disp)

        last_growth = vals[-1] / vals[-2] - 1
        if out["YoY growth"].iloc[0] > max(1.5 * last_growth, last_growth + 0.03):
            st.warning(
                f"**Sanity check:** the model implies {out['YoY growth'].iloc[0] * 100:+.1f}% growth in {fy[0]}, "
                f"but the latest observed year grew only {last_growth * 100:+.1f}%. Polynomial extrapolation can overshoot - "
                "compare with other models in the next tab.",
                icon="⚠️",
            )
        st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False).encode(), f"forecast_{col.lower()}_{model_name.split()[0].lower()}.csv", "text/csv")
        st.caption("Illustrative extrapolation, not an authoritative ecological forecast. Intervals reflect statistical uncertainty only, not model-form error.")

    with t_cmp:
        cmp_ = reg.compare_models(years, vals, holdout=5)
        best = cmp_.loc[cmp_["Back-test MAPE %"].idxmin(), "Model"]
        st.markdown(f"**Best out-of-sample model for {target_label}: {best}** (lowest hold-out error when forecasting the last 5 observed years).")
        show_df(cmp_.round(3))
        fb = px.bar(cmp_, x="Model", y="Back-test MAPE %", title="Hold-out forecast error (lower is better)")
        fb.update_layout(height=320)
        show_chart(fb, "tr_cmp")
        st.caption("In-sample R² always favours flexible models; the back-test trains on early years only and checks forecasts against what actually happened.")


# ===========================================================================
# PAGE: Module 2 - Distribution fitting
# ===========================================================================
def page_distribution():
    st.title("📊 Module 2 · Distribution Fitting")
    st.caption("How are threatened-species counts distributed across countries / territories?")
    demo_banner(dist_demo, "distribution analysis")
    x = dist_df["Total"].to_numpy()
    if len(x) < 10:
        st.error("Need at least 10 rows to fit distributions.")
        return

    t_explore, t_fit, t_top = st.tabs(["Explore", "Fit distributions", "Top countries"])
    desc = dist.describe(x)

    with t_explore:
        stats_df = pd.DataFrame({"Statistic": list(desc), "Value": [f"{v:,.2f}" if isinstance(v, (float, np.floating)) else f"{v:,}" for v in desc.values()]})
        c1, c2 = st.columns([1, 2])
        with c1:
            show_df(stats_df)
        with c2:
            bins = st.slider("Histogram bins", 5, 80, 32)
            logx = st.checkbox("log10(count + 1) scale", value=False)
            plot_df = pd.DataFrame({"v": np.log10(x + 1) if logx else x})
            fig = px.histogram(plot_df, x="v", nbins=bins, labels={"v": "log10(count + 1)" if logx else "Number of threatened species"}, title="Distribution of threatened species per country")
            fig.update_layout(height=340, bargap=0.05)
            show_chart(fig, "ds_hist")
            fb = px.box(plot_df, x="v", points="outliers", title="Box plot", labels={"v": "log10(count + 1)" if logx else "Threatened species"})
            fb.update_layout(height=200)
            show_chart(fb, "ds_box")

    with t_fit:
        fits = dist.fit_all(x)
        table = dist.comparison_table(x, fits)
        best = table.iloc[0]["Distribution"]
        disp_idx = desc["Variance / Mean (dispersion index)"]
        st.success(f"**{best}** has the lowest AIC ({table.iloc[0]['AIC']:.1f}).")
        show_df(table.round(4))
        if disp_idx > 1.5:
            st.info(
                f"Variance/mean = **{disp_idx:.1f}** (≫ 1): the counts are strongly over-dispersed, which a Poisson model "
                "(variance = mean) cannot represent. The Negative Binomial adds a dispersion parameter and typically fits "
                "clustered 'hotspot' data far better.",
                icon="💡",
            )
        oe = dist.observed_vs_expected(x, fits)
        long = oe.melt(id_vars="Bin", var_name="Series", value_name="Countries")
        fig = px.bar(long[long.Series == "Observed"], x="Bin", y="Countries", title="Observed vs expected number of countries per count-bin")
        for name in fits:
            sub = oe[["Bin", name]]
            fig.add_trace(go.Scatter(x=sub["Bin"], y=sub[name], mode="lines+markers", name=name))
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.1))
        show_chart(fig, "ds_oe")
        st.caption("χ² uses (approximately) equiprobable bins under each fitted model. A small p-value means the model is rejected even if it is the best of the three candidates.")

    with t_top:
        n = st.slider("Show top N", 5, min(50, len(dist_df)), 15)
        top = dist_df.sort_values("Total", ascending=False).head(n)
        fig = px.bar(top.iloc[::-1], x="Total", y="Name", orientation="h", title=f"Top {n} countries / territories by threatened species")
        fig.update_layout(height=max(320, 24 * n))
        show_chart(fig, "ds_top")
        share = top["Total"].sum() / dist_df["Total"].sum()
        st.metric(f"Share of all counted threatened species held by top {n}", pct(share))
        show_df(top.reset_index(drop=True))


# ===========================================================================
# PAGE: Module 3 - Markov chain
# ===========================================================================
def page_markov():
    st.title("🔗 Module 3 · Markov Chain Analysis")
    st.caption("How do species move between IUCN categories?")
    demo_banner(status_demo, "Markov analysis")

    counts = mk.transition_counts(status_df)
    P = mk.transition_matrix(counts)
    st.info(
        "**Important:** the dataset lists only species whose category *changed* between 2021 and 2022, so this matrix is "
        "**conditional on a change** (diagonal ≈ 0). It describes *where changing species go*, not the fate of all species.",
        icon="ℹ️",
    )
    st.metric("Species change records analysed", f"{len(status_df):,}")

    t_mat, t_dyn, t_long, t_proj = st.tabs(["Transition matrix", "Direction & reasons", "Extinction pathway", "Projection"])

    with t_mat:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(counts, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Status 2022", y="Status 2021", color="Species"), title="Transition counts")
            fig.update_layout(height=420)
            show_chart(fig, "mk_cnt")
        with c2:
            fig = px.imshow(P, text_auto=".2f", color_continuous_scale="YlOrRd", zmin=0, zmax=1, labels=dict(x="Status 2022", y="Status 2021", color="P"), title="Transition probabilities")
            fig.update_layout(height=420)
            show_chart(fig, "mk_prob")
        labels, src, tgt, val = mk.sankey_data(counts)
        colors = [STATUS_COLORS[s] for s in STATES] * 2
        fig = go.Figure(go.Sankey(node=dict(label=labels, color=colors, pad=15), link=dict(source=src, target=tgt, value=val)))
        fig.update_layout(title="Flow of species between categories (2021 → 2022)", height=450)
        show_chart(fig, "mk_sankey")
        st.download_button("⬇️ Download transition matrix (CSV)", P.to_csv().encode(), "status_transition_matrix.csv", "text/csv")

    with t_dyn:
        d = mk.direction_summary(status_df)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(values=d.values, names=d.index, hole=0.45, title="Direction of change", color=d.index, color_discrete_map={"Worsened": "#d64545", "Improved": "#4c9f50", "Unchanged": "#999"})
            fig.update_layout(height=340)
            show_chart(fig, "mk_dir")
        with c2:
            rs = status_df["Reason for change"].dropna()
            if len(rs):
                vc = rs.value_counts().head(10)
                fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Reason for change", labels={"x": "Species", "y": ""})
                fig.update_layout(height=340, yaxis=dict(autorange="reversed"))
                show_chart(fig, "mk_reason")
            else:
                st.write("No 'Reason for change' column in the data.")
        by_tax = status_df.assign(Dir=np.where(status_df.Status_2022.map(RANK) > status_df.Status_2021.map(RANK), "Worsened", "Improved")).groupby(["Taxonomy", "Dir"]).size().reset_index(name="Species")
        fig = px.bar(by_tax, x="Taxonomy", y="Species", color="Dir", barmode="stack", color_discrete_map={"Worsened": "#d64545", "Improved": "#4c9f50"}, title="Direction of change by taxonomic group")
        fig.update_layout(height=380)
        show_chart(fig, "mk_tax")

    with t_long:
        res = mk.absorbing_analysis(P)
        if res:
            st.markdown(f"Absorbing state(s): **{', '.join(res['absorbing'])}** (no recorded way out).")
            ap = res["absorption_prob"].copy()
            es = res["expected_steps"]
            tab = pd.DataFrame({"Start status": ap.index, **{f"P(end in {c})": ap[c].round(3).to_numpy() for c in ap.columns}, "Expected # of status changes before absorption": [("∞" if np.isinf(v) else f"{v:.1f}") for v in es]})
            show_df(tab)
            st.markdown(
                "Under the change-only chain, a species keeps changing category until it is absorbed. "
                "**Expected status changes** = how many category changes a species of that starting status is expected to go "
                "through before reaching the absorbing state - a *relative* measure of how 'close' each category is to extinction. "
                "It is **not** a time-to-extinction in years."
            )
            fig = px.bar(x=es.index, y=es.replace(np.inf, np.nan).values, labels={"x": "Start status", "y": "Expected status changes"}, title="Expected number of status changes before extinction", color=es.index, color_discrete_map=STATUS_COLORS)
            fig.update_layout(height=320, showlegend=False)
            show_chart(fig, "mk_es")
        else:
            ss = mk.stationary_distribution(P)
            if ss is not None:
                st.markdown("No absorbing state in the data (extinct species do not change category in this file's chain). Long-run distribution of the change-only chain:")
                fig = px.bar(x=ss.index, y=ss.values, color=ss.index, color_discrete_map=STATUS_COLORS, labels={"x": "Status", "y": "Long-run probability"}, title="Stationary distribution")
                fig.update_layout(height=320, showlegend=False)
                show_chart(fig, "mk_ss")
            else:
                st.warning("The chain is reducible without a unique stationary distribution; long-run analysis is not defined.")
        st.caption("The original script's eigenvector 'steady state' is replaced by this analysis: when EX is absorbing the eigen-solution is trivially 100% EX and, with repeated eigenvalue 1, not unique.")

    with t_proj:
        st.markdown(
            "Turn the change-only matrix into a **one-year** matrix with an assumed annual change rate *c*: "
            "`P_year = (1 − c)·I + c·P_change`. *c* is **your assumption** (e.g. the fraction of assessed species whose "
            "category changes in a typical year), not something estimated from this dataset."
        )
        c1, c2 = st.columns(2)
        rate = c1.slider("Annual change rate c", 0.001, 0.20, 0.02, 0.001, format="%.3f")
        years = c2.slider("Years to project", 1, 100, 25)
        mode = st.radio("Starting population", ["Single status", "Custom mix"], horizontal=True)
        if mode == "Single status":
            s0 = st.selectbox("Start status", STATES[:-1], index=0)
            start = pd.Series({s: float(s == s0) for s in STATES})
        else:
            cols = st.columns(len(STATES))
            defaults = {"LC": 70.0, "NT": 10.0, "VU": 10.0, "EN": 5.0, "CR": 5.0, "EX": 0.0}
            start = pd.Series({s: cols[i].number_input(s, 0.0, 100.0, defaults[s], 1.0, key=f"mix_{s}") for i, s in enumerate(STATES)})
            if start.sum() == 0:
                st.error("Enter at least one non-zero share.")
                return
        Pyr = mk.with_annual_change_rate(P, rate)
        proj = mk.project(Pyr, start, years)
        long = proj.melt(id_vars="Step", var_name="Status", value_name="Share")
        fig = px.area(long, x="Step", y="Share", color="Status", color_discrete_map=STATUS_COLORS, category_orders={"Status": STATES}, title="Projected distribution across IUCN categories", labels={"Step": "Years ahead"})
        fig.update_layout(height=400, yaxis_tickformat=".0%")
        show_chart(fig, "mk_proj")
        end = proj.iloc[-1].drop("Step")
        st.markdown("**Distribution after the horizon:** " + " · ".join(f"{s}: {pct(end[s])}" for s in STATES))
        st.download_button("⬇️ Download projection (CSV)", proj.to_csv(index=False).encode(), "markov_projection.csv", "text/csv")
        st.caption("Scenario analysis under a time-homogeneous, memoryless assumption - real conservation status depends on drivers (habitat loss, policy, climate) that this chain ignores.")


# ===========================================================================
# PAGE: Module 4 - Predictor
# ===========================================================================
@st.cache_resource(show_spinner="Training Random Forest…")
def get_model(species: pd.DataFrame, n_estimators: int, test_size: float, max_depth: int, balanced: bool, seed: int):
    return clf.train_random_forest(species, n_estimators=n_estimators, test_size=test_size, max_depth=(None if max_depth == 0 else max_depth), seed=seed, balanced=balanced)


@st.cache_data(show_spinner="Benchmarking algorithms (cross-validation)…")
def get_benchmark(species: pd.DataFrame, n_estimators: int, balanced: bool, seed: int):
    return clf.compare_algorithms(species, n_estimators=n_estimators, seed=seed, balanced=balanced)


def page_predictor():
    st.title("🌳 Module 4 · IUCN Status Predictor")
    st.caption("Predict a species group's next IUCN status from its current (2021) status and taxonomic group.")
    demo_banner(status_demo, "classifier")

    with st.expander("⚙️ Model settings"):
        c = st.columns(5)
        n_est = c[0].slider("Trees", 20, 500, 100, 10)
        test_size = c[1].slider("Test share", 0.1, 0.5, 0.25, 0.05)
        depth = c[2].slider("Max depth (0 = none)", 0, 20, 0)
        balanced = c[3].checkbox("Balanced class weights", value=True, help="As in the original script. Improves recall for rare classes but usually lowers overall accuracy.")
        seed = c[4].number_input("Random seed", 0, 9999, 0)

    try:
        bundle = get_model(status_df, n_est, test_size, depth, balanced, int(seed))
    except ValueError as e:
        st.error(str(e))
        return

    t_pred, t_perf, t_cmp, t_batch = st.tabs(["🔮 Predict", "📊 Model performance", "⚖️ Algorithm benchmark", "📁 Batch prediction"])

    with t_pred:
        c1, c2 = st.columns(2)
        taxa = bundle["taxonomies"]
        custom = c1.checkbox("My group is not in the list")
        tax = c1.text_input("Taxonomic group") if custom else c1.selectbox("Taxonomic group", taxa)
        status = c2.selectbox("IUCN status in 2021", STATES, format_func=lambda s: f"{s} · {STATUS_LABELS[s]}", index=2)
        tax = (tax or "").strip()
        if not tax:
            st.info("Enter a taxonomic group name to get a prediction.")
            tax = "Unknown"
        r = clf.predict_status(bundle, tax, status)
        if not r["taxonomy_known"]:
            st.warning("This group was not in the training data, so the prediction relies mainly on the 2021 status.", icon="⚠️")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        a, b, cc, d = st.columns([2.2, 1, 1, 1])
        a.markdown(f"**Predicted IUCN status in 2022**<br>{badge(r['prediction'])}", unsafe_allow_html=True)
        b.metric("Confidence", pct(r["confidence"]))
        cc.metric("Risk tier", r["risk_tier"], help="Low = LC/NT · Medium = VU · High = EN/CR/EX")
        d.metric("P(CR or EX)", pct(r["p_cr_or_ex"]))
        st.markdown("</div>", unsafe_allow_html=True)

        t = r["table"]
        fig = px.bar(t, x="IUCN status", y="Probability", color="IUCN status", color_discrete_map=STATUS_COLORS, category_orders={"IUCN status": STATES}, title="Predicted probability of each 2022 status")
        fig.update_layout(height=340, showlegend=False, yaxis_tickformat=".0%")
        show_chart(fig, "pr_bar")
        m = st.columns(3)
        m[0].metric("Worsening", pct(r["p_worse"]))
        m[1].metric("Same category", pct(r["p_same"]))
        m[2].metric("Improving", pct(r["p_better"]))
        st.caption(
            "The model is trained only on species whose category changed, so all probabilities are **conditional on a "
            "change having occurred**. It is a decision-support indicator, not a forecast of extinction."
        )

    with t_perf:
        m = st.columns(4)
        m[0].metric("Hold-out accuracy", pct(bundle["accuracy"]), help=f"{bundle['n_test']} test records")
        m[1].metric("Majority-class baseline", pct(bundle["baseline_accuracy"]))
        m[2].metric("Balanced accuracy", pct(bundle["balanced_accuracy"]))
        m[3].metric("Macro F1", f"{bundle['f1_macro']:.3f}")
        m = st.columns(4)
        m[0].metric(f"{bundle['cv_folds']}-fold CV accuracy", pct(bundle["cv_accuracy"][0]), f"± {bundle['cv_accuracy'][1] * 100:.1f} pp", delta_color="off")
        m[1].metric("Lookup-table CV accuracy", pct(bundle["lookup_cv_accuracy"]), help="Predict the most frequent 2022 status for each (2021 status × group) cell")
        m[2].metric("Direction accuracy", pct(bundle["direction_accuracy"]), help="Correctly predicting worsened / improved / same")
        m[3].metric("Records (train / test)", f"{bundle['n_train']} / {bundle['n_test']}")

        if bundle["cv_accuracy"][0] <= bundle["lookup_cv_accuracy"] + 0.01:
            st.info(
                "With only two input features, the Random Forest performs about the same as (or worse than) a simple "
                "lookup table of the most common outcome per status × group. Adding richer features (habitat, population "
                "trend, range size, threats) is what would let a model add real value - see the report's future scope.",
                icon="💡",
            )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.imshow(bundle["confusion_matrix"], text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix (hold-out set)")
            fig.update_layout(height=400)
            show_chart(fig, "pf_cm")
        with c2:
            imp = bundle["importances"].head(12).iloc[::-1]
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Feature importance", labels={"x": "Importance", "y": ""})
            fig.update_layout(height=400)
            show_chart(fig, "pf_imp")
        rep = bundle["report"].reset_index().rename(columns={"index": "Class"})
        show_df(rep.round(3))
        cc = bundle["class_counts"]
        st.caption("Class distribution of targets: " + ", ".join(f"{k}: {v}" for k, v in cc.items()))

        buf = io.BytesIO()
        import joblib
        joblib.dump({k: bundle[k] for k in ("pipeline", "classes", "taxonomies")}, buf)
        st.download_button("⬇️ Download trained model (.pkl)", buf.getvalue(), "iucn_random_forest_model.pkl", "application/octet-stream")

    with t_cmp:
        st.markdown("Cross-validated comparison of the algorithms listed in the report (Section 3.5) against simple baselines.")
        sig = (status_src, len(status_df), n_est, balanced, int(seed))
        if st.button("Run benchmark"):
            st.session_state["bench"] = (sig, get_benchmark(status_df, n_est, balanced, int(seed)))
        if "bench" in st.session_state and st.session_state["bench"][0] == sig:
            bt = st.session_state["bench"][1]
            show_df(bt.round(3))
            fig = px.bar(bt, x="Algorithm", y="Accuracy", title="Cross-validated accuracy", color="Accuracy", color_continuous_scale="Greens")
            fig.update_layout(height=380, coloraxis_showscale=False)
            show_chart(fig, "cmp_bar")

    with t_batch:
        st.markdown("Upload a CSV/Excel file with columns **Taxonomy** and **Status_2021** to score many groups at once.")
        tmpl = pd.DataFrame({"Taxonomy": bundle["taxonomies"][:3], "Status_2021": ["EN", "VU", "LC"][: len(bundle["taxonomies"][:3])]})
        st.download_button("⬇️ Download template", tmpl.to_csv(index=False).encode(), "batch_template.csv", "text/csv")
        up = st.file_uploader("Batch file", type=["csv", "xlsx"], key="batch_up")
        if up is not None:
            try:
                res = clf.predict_batch(bundle, read_any(up.getvalue(), up.name))
                show_df(res)
                st.download_button("⬇️ Download predictions (CSV)", res.to_csv(index=False).encode(), "batch_predictions.csv", "text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not score file: {e}")


# ===========================================================================
# PAGE: Data & pipeline
# ===========================================================================
def page_data():
    st.title("🗂️ Data & Pipeline")
    st.markdown("**Pipeline:** Data collection → Cleaning → Statistical analysis → Markov analysis → Random Forest → Prediction & visualisation")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution data")
        st.caption(dist_src)
        st.json(dist_q, expanded=False)
        show_df(dist_df.head(50))
        st.download_button("⬇️ Cleaned distribution data (CSV)", dist_df.to_csv(index=False).encode(), "distribution_clean.csv", "text/csv")
    with c2:
        st.subheader("Status-change data")
        st.caption(status_src)
        st.json(status_q, expanded=False)
        show_df(status_df.head(50))
        st.download_button("⬇️ Cleaned status-change data (CSV)", status_df.to_csv(index=False).encode(), "status_changes_clean.csv", "text/csv")

    with st.expander("Expected file formats"):
        st.markdown(
            """
**Distribution file** (`.csv`): one row per country/territory, with a column **Name** and a numeric column **Total**
(thousands separators such as `1,234` are accepted). Aggregate rows named *Total / World / Global* are removed automatically.

**Status-change file** (`.xlsx` or `.csv`): the IUCN Red List table of category changes, with columns
**Scientific name**, **Common name**, **IUCN Red List (2021)**, **IUCN Red List (2022)** and optionally **Reason for change**.
Rows that contain only a scientific name (no common name, no status) are treated as *taxonomic group headers* and are
forward-filled onto the species below them. Status values may be codes (`LC, NT, VU, EN, CR, EX`) or full names.
            """
        )
    with st.expander("Download demo files to see the format"):
        st.download_button("Demo distribution (CSV)", _demo_dist().to_csv(index=False).encode(), "demo_distribution.csv", "text/csv")
        st.download_button("Demo status changes (CSV)", _demo_status().to_csv(index=False).encode(), "demo_status_changes.csv", "text/csv")


# ===========================================================================
# PAGE: About
# ===========================================================================
def page_about():
    st.title("ℹ️ About & Method")
    st.markdown(
        """
### Methods used
| Module | Method | Purpose |
|---|---|---|
| 1 | Polynomial / exponential regression (OLS), prediction intervals, CAGR, hold-out back-test | Quantify and extrapolate growth in assessed & threatened species |
| 2 | Maximum-likelihood fits of Poisson, Geometric, Negative Binomial; AIC/BIC; χ² GOF | Describe how threatened species cluster across countries |
| 3 | Discrete-time Markov chain, absorbing-chain analysis, scenario projection | Describe how species move between IUCN categories |
| 4 | Random Forest (+ Decision Tree, Logistic Regression, Naive Bayes, SVM benchmark) | Predict next IUCN category from current category + taxonomic group |

### How to interpret results responsibly
* **Module 1** extrapolates from totals that are driven partly by *assessment effort*, not only by true biodiversity loss.
* **Module 2** treats countries as independent draws; spatial dependence and country size are ignored.
* **Module 3** is trained only on species that *changed* category - it is conditional on change.
* **Module 4** predicts a *Red List category*, not extinction itself, and uses just two features, so its skill is limited.
* All outputs are decision-support insights and **do not replace expert ecological assessment** (as the report also states).

### Differences from the original script
* `input()` prompts → interactive widgets; Matplotlib → interactive Plotly charts; hard-coded `D:\\` paths → uploads / `data/` folder.
* OLS with prediction intervals replaces `curve_fit` (identical estimates); Geometric now included in model comparison; χ² GOF added.
* Markov "steady state" replaced by absorbing-chain analysis and an explicit-assumption projection.
* Random Forest wrapped in a Pipeline that handles unseen groups; cross-validation, lookup baseline and algorithm benchmark added.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
{
    PAGES[0]: page_overview,
    PAGES[1]: page_trends,
    PAGES[2]: page_distribution,
    PAGES[3]: page_markov,
    PAGES[4]: page_predictor,
    PAGES[5]: page_data,
    PAGES[6]: page_about,
}[page]()
