"""
app.py — LOGVIP Streamlit Dashboard entry point.

Tabs:
  1. Raw QC        — file upload, curve statistics, null/outlier summary
  2. Petrophysics  — parameter controls, interactive log tracks, Vsh comparison
  3. Lithology     — two-mineral and multi-mineral breakdown
  4. Facies        — elbow/silhouette plots, k-selector, facies log
  5. Summary/Export — net-pay table, (illustrative) STOIIP, CSV download

No petrophysical formulas live here — all calculations are delegated to src/.
"""

import sys
import os
import io
import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from io_utils    import load_log_file
from qc          import describe_curves, flag_outliers, outlier_summary
from petrophysics import run_petrophysics_pipeline
from lithology   import two_mineral_split, multimineral_split
from facies      import cluster_facies, elbow_data, silhouette_data
from plotting    import (
    plotly_log_tracks,
    plotly_crossplot,
    plotly_vsh_comparison,
    plotly_elbow,
    static_log_plot,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LOGVIP — Well Log Petrophysics",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

.logvip-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.logvip-header h1 {font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px;}
.logvip-header p  {font-size: 0.95rem; opacity: 0.75; margin: 0.3rem 0 0;}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {font-size: 1.6rem; font-weight: 700; color: #1e40af;}
.metric-card .label {font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;}

.warn-box {
    background: #fff7ed; border: 1px solid #fdba74;
    border-radius: 8px; padding: 0.8rem 1rem;
    color: #92400e; font-size: 0.88rem;
}
.info-box {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-radius: 8px; padding: 0.8rem 1rem;
    color: #1e40af; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logvip-header">
  <h1>🛢 LOGVIP</h1>
  <p>Well Log Petrophysics Interpretation Platform &nbsp;·&nbsp; Teaching-grade, single-well analysis tool</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — file upload + parameter controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload well log file (.xlsx or .csv)",
        type=["xlsx", "xls", "csv"],
        help="Columns expected: DEPT, GR, NPHI, RDEEP, RHOB. A units row is auto-detected.",
    )

    use_sample = st.checkbox("Use built-in sample data", value=not bool(uploaded_file))

    st.markdown("---")
    st.markdown("## ⚙️ GR Baselines")
    gr_method = st.selectbox(
        "Baseline method",
        ["Percentile (5th/95th) — Recommended", "Manual"],
        index=0,
    )
    if "Manual" in gr_method:
        gr_sand_input  = st.number_input("GR Sand (gAPI)", value=30.0, step=1.0)
        gr_shale_input = st.number_input("GR Shale (gAPI)", value=110.0, step=1.0)
    else:
        gr_sand_input  = None
        gr_shale_input = None

    st.markdown("---")
    st.markdown("## 🧪 Petrophysical Constants")
    st.caption("Confirm against mud report / water sample before trusting results.")
    rho_matrix = st.number_input("ρ_matrix (g/cc)", value=2.65, step=0.01,
                                  help="2.65 = clean sandstone (default). Confirm for this field.")
    rho_fluid  = st.number_input("ρ_fluid  (g/cc)", value=1.00, step=0.01,
                                  help="1.0 = fresh water. Use 1.1 for salt water.")
    Rw = st.number_input("Rw (ohm·m)", value=0.135, step=0.005, format="%.3f",
                          help="Formation water resistivity. Confirm from SP log or water sample.")
    C  = st.number_input("Simandoux C", value=0.4, step=0.05, format="%.2f",
                          help="Simandoux constant. Literature default = 0.4.")

    st.markdown("---")
    st.markdown("## ✂️ Cutoff Thresholds")
    st.caption("Industry defaults — validate against this field's data.")
    igr_cut = st.slider("Net-Sand:  Igr <",  0.0, 1.0, 0.50, 0.05)
    phi_cut = st.slider("Net-Reservoir: φ ≥", 0.0, 0.45, 0.15, 0.01)
    sw_cut  = st.slider("Net-Pay:   Sw <",   0.0, 1.0, 0.70, 0.05)

    st.markdown("---")
    st.markdown("## 📊 Vsh Method")
    vsh_method_label = st.selectbox(
        "Primary Vsh model",
        ["Larionov-Tertiary (default)", "Larionov-Older", "Steiber"],
    )
    vsh_map_ui = {
        "Larionov-Tertiary (default)": "larionov_t",
        "Larionov-Older":              "larionov_o",
        "Steiber":                     "steiber",
    }
    vsh_method = vsh_map_ui[vsh_method_label]

    st.markdown("---")
    st.markdown("## 🔬 Facies")
    k_facies = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
    well_name = st.text_input("Well name (for plot titles)", value="LOGVIP-Well")

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_sample_data() -> pd.DataFrame:
    """Generate a reproducible synthetic 300-row well log for demonstration."""
    rng = np.random.default_rng(42)
    n = 300
    depth = np.linspace(2000, 2150, n)
    gr    = np.clip(rng.normal(60, 25, n) + 20 * np.sin(np.linspace(0, 6*np.pi, n)), 5, 280)
    nphi  = np.clip(0.18 + 0.08 * np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 0.03, n), 0.02, 0.55)
    rhob  = np.clip(2.35 - 0.3 * nphi + rng.normal(0, 0.05, n), 1.8, 2.9)
    rdeep = np.clip(np.exp(rng.normal(1.5, 0.8, n)), 0.3, 1500)
    df = pd.DataFrame({"DEPT": depth, "GR": gr, "NPHI": nphi, "RHOB": rhob, "RDEEP": rdeep})
    return df.round(4)


@st.cache_data(show_spinner=False)
def _load_and_clean(buf, file_name: str) -> tuple:
    return load_log_file(buf, required_cols=["DEPT", "GR", "NPHI", "RDEEP", "RHOB"])


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
raw_df   = None
io_report = {}

if uploaded_file and not use_sample:
    buf = io.BytesIO(uploaded_file.read())
    buf.name = uploaded_file.name
    try:
        with st.spinner("Loading file…"):
            raw_df, io_report = _load_and_clean(buf, uploaded_file.name)
        st.sidebar.success(f"✅ {len(raw_df)} rows loaded from {uploaded_file.name}")
    except Exception as exc:
        st.sidebar.error(f"❌ Failed to load file: {exc}")
else:
    raw_df = _load_sample_data()
    io_report = {"units_row_dropped": False, "cols_dropped": [], "nulls_replaced": 0,
                 "rows_before": len(raw_df), "rows_after": len(raw_df)}
    if use_sample:
        st.sidebar.info("ℹ️ Using built-in synthetic sample data.")

# Detect depth column
depth_col = "DEPT" if "DEPT" in raw_df.columns else (
    "DEPTH" if "DEPTH" in raw_df.columns else raw_df.columns[0]
)

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline (cached on params)
# ─────────────────────────────────────────────────────────────────────────────
pipeline_params = dict(
    gr_col="GR", nphi_col="NPHI", rhob_col="RHOB", rdeep_col="RDEEP",
    gr_sand=gr_sand_input, gr_shale=gr_shale_input,
    rho_matrix=rho_matrix, rho_fluid=rho_fluid,
    Rw=Rw, C=C,
    igr_cut=igr_cut, phi_cut=phi_cut, sw_cut=sw_cut,
    vsh_method=vsh_method,
)

@st.cache_data(show_spinner=False)
def _run_pipeline(df_json: str, params_key: str) -> pd.DataFrame:
    df_in = pd.read_json(io.StringIO(df_json), orient="split")
    import json
    params = json.loads(params_key)
    return run_petrophysics_pipeline(df_in, params)

import json
try:
    with st.spinner("Running petrophysics pipeline…"):
        proc_df = _run_pipeline(
            raw_df.to_json(orient="split"),
            json.dumps(pipeline_params, sort_keys=True),
        )
except Exception as exc:
    st.error(f"Pipeline error: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Raw QC",
    "⚗️ Petrophysics",
    "🪨 Lithology",
    "🎨 Facies",
    "📋 Summary / Export",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Raw QC
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Curve Statistics")

    if io_report.get("units_row_dropped"):
        st.info("ℹ️ A units row (row 0) was detected and removed from the data.")
    if io_report.get("cols_dropped"):
        st.warning(f"⚠️ Columns that could not be cast to float and were dropped: {io_report['cols_dropped']}")
    if io_report.get("nulls_replaced", 0) > 0:
        st.info(f"ℹ️ {io_report['nulls_replaced']} LAS null sentinel values (-999.25) replaced with NaN.")

    stats_df = describe_curves(raw_df)
    st.dataframe(stats_df.style.format(precision=4), use_container_width=True)

    st.subheader("⚠️ Physical Range Flags")
    flag_df = flag_outliers(raw_df)
    summ_df = outlier_summary(flag_df)
    if summ_df["n_flagged"].sum() == 0:
        st.success("✅ No values outside physical plausibility ranges.")
    else:
        st.dataframe(summ_df.style.format(precision=2), use_container_width=True)
        n_bad_rows = int(flag_df.any(axis=1).sum())
        st.warning(f"⚠️ {n_bad_rows} rows have at least one out-of-range curve value.")

    st.subheader("📈 Raw Log Preview")
    curves_available = [c for c in ["GR", "NPHI", "RHOB", "RDEEP"] if c in raw_df.columns]
    sel_curve = st.selectbox("Select curve to preview", curves_available)
    if sel_curve:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=raw_df[sel_curve], y=raw_df[depth_col],
            mode="lines", name=sel_curve, line=dict(width=1),
        ))
        fig_raw.update_yaxes(autorange="reversed", title_text="Depth (m)")
        fig_raw.update_xaxes(title_text=sel_curve)
        fig_raw.update_layout(height=450, margin=dict(t=30, l=60, r=20, b=40))
        st.plotly_chart(fig_raw, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Petrophysics
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Assumption check required:</strong> ρ_matrix, ρ_fluid, Rw, and C are 
    notebook defaults from the source data. Confirm against the field's mud report 
    and/or water sample before using these results for decisions.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    net_sand = int(proc_df.get("net_sand", pd.Series(False)).sum()) if "net_sand" in proc_df.columns else 0
    net_res  = int(proc_df.get("net_reservoir", pd.Series(False)).sum()) if "net_reservoir" in proc_df.columns else 0
    net_pay  = int(proc_df.get("net_pay", pd.Series(False)).sum()) if "net_pay" in proc_df.columns else 0
    avg_phi  = proc_df["porosity"].mean() if "porosity" in proc_df.columns else float("nan")
    avg_sw   = proc_df["Sw"].mean() if "Sw" in proc_df.columns else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{net_sand}", "Net Sand rows"),
        (c2, f"{net_res}",  "Net Reservoir rows"),
        (c3, f"{net_pay}",  "Net Pay rows"),
        (c4, f"{avg_phi:.3f}", "Avg φ_eff"),
        (c5, f"{avg_sw:.3f}",  "Avg Sw"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="value">{val}</div>
          <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📊 Interactive Log Tracks")
    fig_tracks = plotly_log_tracks(proc_df, depth_col=depth_col, well_name=well_name)
    st.plotly_chart(fig_tracks, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🪨 Vsh Method Comparison")
        st.caption("Expected ranking: Larionov-T ≤ Steiber ≤ Larionov-O for same Igr (PRD §9 C.11)")
        fig_vsh = plotly_vsh_comparison(proc_df, depth_col=depth_col)
        st.plotly_chart(fig_vsh, use_container_width=True)
    with col_b:
        st.subheader("💧 Neutron–Density Crossplot")
        fig_xp = plotly_crossplot(proc_df, depth_col=depth_col)
        st.plotly_chart(fig_xp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Lithology
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🪨 Two-Mineral Split (GR-based)")
    lith_df = two_mineral_split(proc_df, igr_col="Igr")

    fig_2m = go.Figure()
    fig_2m.add_trace(go.Scatter(
        x=lith_df["lith_sand"], y=lith_df[depth_col],
        mode="lines", name="Sand", fill="tozerox",
        line=dict(color="#f4c430"),
        fillcolor="rgba(244,196,48,0.35)",
    ))
    fig_2m.add_trace(go.Scatter(
        x=lith_df["lith_shale"], y=lith_df[depth_col],
        mode="lines", name="Shale",
        line=dict(color="#8b7355"),
    ))
    fig_2m.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig_2m.update_xaxes(range=[0, 1], title_text="Volume fraction")
    fig_2m.update_layout(height=500, hovermode="y unified")
    st.plotly_chart(fig_2m, use_container_width=True)

    st.markdown("---")
    st.subheader("⚗️ Multi-Mineral Split (RHOB + NPHI optimizer)")
    st.markdown("""
    <div class="info-box">
    ℹ️ Uses SLSQP optimisation to solve 4-mineral volumes (sand, shale, calcite, heavy mineral) 
    that best match RHOB and NPHI at each depth. Rows where the optimizer fails are 
    left blank and reported below.
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Run multi-mineral optimizer (may take a few seconds)"):
        with st.spinner("Running optimizer…"):
            mm_df = multimineral_split(proc_df, rhob_col="RHOB", nphi_col="NPHI")

        n_failed = int((~mm_df["mm_converged"]).sum())
        n_total  = len(mm_df)
        if n_failed > 0:
            st.warning(
                f"⚠️ Optimizer failed to converge on {n_failed} / {n_total} rows "
                f"({round(100*n_failed/n_total)}%). "
                "Those rows have NaN mineral volumes and are excluded from the plot."
            )
        else:
            st.success(f"✅ Optimizer converged on all {n_total} rows.")

        minerals = ["mm_sand", "mm_shale", "mm_calcite", "mm_heavy"]
        colors   = ["#f4c430", "#8b7355", "#a0cbe8", "#e15759"]
        labels   = ["Sand", "Shale", "Calcite", "Heavy Minerals"]

        fig_mm = go.Figure()
        cumulative = np.zeros(len(mm_df))
        for col, color, label in zip(minerals, colors, labels):
            vals = mm_df[col].fillna(0).values
            fig_mm.add_trace(go.Bar(
                x=vals,
                y=mm_df[depth_col],
                orientation="h",
                name=label,
                marker_color=color,
                width=0.5,
            ))

        fig_mm.update_yaxes(autorange="reversed", title_text="Depth (m)")
        fig_mm.update_xaxes(range=[0, 1], title_text="Volume fraction")
        fig_mm.update_layout(
            barmode="stack", height=500,
            title="Multi-Mineral Volume Fractions",
        )
        st.plotly_chart(fig_mm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Facies
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎨 KMeans Facies Classification")
    st.caption(
        "StandardScaler + KMeans on [GR, RHOB_ns, NPHI_ns, RDEEP]. "
        "Use the elbow/silhouette plots below to justify k before relying on results."
    )

    # Check available features
    feat_cols_default = ["GR", "RHOB_ns", "NPHI_ns", "RDEEP"]
    feat_cols = [c for c in feat_cols_default if c in proc_df.columns]
    if not feat_cols:
        st.warning("⚠️ No feature columns available. Run the petrophysics pipeline first.")
    else:
        col_elbow, col_log = st.columns([1, 1])

        with col_elbow:
            st.markdown("#### Cluster Count Selection")
            with st.spinner("Computing elbow / silhouette…"):
                elbow_df   = elbow_data(proc_df, feature_cols=feat_cols)
                sil_df     = silhouette_data(proc_df, feature_cols=feat_cols)
            fig_elbow = plotly_elbow(elbow_df, sil_df)
            st.plotly_chart(fig_elbow, use_container_width=True)
            st.caption(f"Selected k = **{k_facies}** (adjust in sidebar)")

        with col_log:
            st.markdown("#### Facies Log")
            with st.spinner(f"Clustering k={k_facies}…"):
                facies_df = cluster_facies(proc_df, feature_cols=feat_cols, k=k_facies)

            fig_facies = go.Figure()
            for f in sorted(facies_df["facies"].unique()):
                if f < 0:
                    continue
                mask = facies_df["facies"] == f
                color = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                         "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                         "#9c755f", "#bab0ac"][f % 10]
                fig_facies.add_trace(go.Scatter(
                    x=[f] * mask.sum(),
                    y=facies_df.loc[mask, depth_col],
                    mode="markers",
                    name=f"Facies {f}",
                    marker=dict(color=color, size=5, symbol="square"),
                ))
            fig_facies.update_yaxes(autorange="reversed", title_text="Depth (m)")
            fig_facies.update_xaxes(title_text="Facies class", tickvals=list(range(k_facies)))
            fig_facies.update_layout(height=500, hovermode="y unified")
            st.plotly_chart(fig_facies, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Summary / Export
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Net-Zone Summary")

    depth_step = float(proc_df[depth_col].diff().median())

    summary_rows = []
    for zone, col in [("Net Sand", "net_sand"), ("Net Reservoir", "net_reservoir"), ("Net Pay", "net_pay")]:
        if col in proc_df.columns:
            n_rows   = int(proc_df[col].sum())
            thickness = round(n_rows * depth_step, 2)
            avg_phi  = round(proc_df.loc[proc_df[col], "porosity"].mean(), 4) if n_rows > 0 else float("nan")
            avg_sw   = round(proc_df.loc[proc_df[col], "Sw"].mean(), 4) if n_rows > 0 else float("nan")
            summary_rows.append({
                "Zone": zone, "Rows": n_rows,
                "Thickness (m)": thickness,
                "Avg φ_eff": avg_phi, "Avg Sw": avg_sw,
            })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("Zone"), use_container_width=True)

    st.markdown("---")
    st.subheader("🔢 STOIIP (Illustrative Only)")
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>STOIIP is provided for illustration only.</strong> The area value below is 
    from the source notebook annotation ("roughly from map") with no cited source or units 
    confirmation. Do not quote this result in any engineering report without verifying 
    A (m²), Boi, and all inputs against authoritative field data.
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        A_m2 = st.number_input("Area A (m²) [illustrative]", value=24_570_000.0, step=100_000.0,
                                format="%.0f")
    with col_s2:
        Boi = st.number_input("Formation vol. factor Boi", value=1.2, step=0.05)
    with col_s3:
        net_pay_h = summary_rows[-1]["Thickness (m)"] if summary_rows else 5.0
        st.metric("Net Pay h (m)", f"{net_pay_h:.2f}")

    if "net_pay" in proc_df.columns and proc_df["net_pay"].any():
        avg_phi_pay = proc_df.loc[proc_df["net_pay"], "porosity"].mean()
        avg_sw_pay  = proc_df.loc[proc_df["net_pay"], "Sw"].mean()
        stoiip = A_m2 * net_pay_h * avg_phi_pay * (1.0 - avg_sw_pay) / Boi
        st.metric("STOIIP (m³) [Illustrative]", f"{stoiip:,.0f}")
    else:
        st.info("No net-pay rows found with current cutoffs.")

    st.markdown("---")
    st.subheader("⬇️ Download Processed Data")

    csv_bytes = proc_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download processed CSV",
        data=csv_bytes,
        file_name=f"{well_name}_processed.csv",
        mime="text/csv",
    )

    html_bytes = proc_df.to_html(index=False).encode("utf-8")
    st.download_button(
        label="🌐 Download processed HTML table",
        data=html_bytes,
        file_name=f"{well_name}_processed.html",
        mime="text/html",
    )

    if st.button("📄 Generate static log plot (PNG)"):
        with st.spinner("Generating matplotlib figure…"):
            fig_static = static_log_plot(proc_df, depth_col=depth_col, well_name=well_name)
            buf_png = io.BytesIO()
            fig_static.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
            buf_png.seek(0)
        st.download_button(
            label="⬇️ Download composite log PNG",
            data=buf_png,
            file_name=f"{well_name}_composite_log.png",
            mime="image/png",
        )
        st.pyplot(fig_static)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>LOGVIP v1.0</strong> — Teaching-grade petrophysics tool.<br>
    Library stack: numpy · pandas · scipy · matplotlib · plotly · scikit-learn · streamlit.<br>
    No geopandas · No python-docx · No lasio.<br>
    All formulas are audited in <code>src/petrophysics.py</code>.
    </div>
    """, unsafe_allow_html=True)
