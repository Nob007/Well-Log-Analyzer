"""
plotting.py — Static (matplotlib) and interactive (Plotly) log plots for LOGVIP.

Provides:
  - static_log_plot(): matplotlib multi-track composite well log.
  - plotly_log_tracks(): Plotly subplot equivalent for Streamlit (hover-enabled).
  - plotly_crossplot(): density-neutron crossplot with lithology lines.
  - plotly_vsh_comparison(): overlay of three Vsh methods on one chart.
  - plotly_elbow(): elbow and silhouette plots for facies k selection.

All functions return figure objects — they do NOT call plt.show() or
fig.show() internally. The caller (app.py) is responsible for rendering.

Approved libraries: numpy, pandas, matplotlib, plotly only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Colour palette for facies (up to 10 classes)
FACIES_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


# ─────────────────────────────────────────────────────────────────────────────
# Static matplotlib composite log
# ─────────────────────────────────────────────────────────────────────────────

def static_log_plot(
    df: pd.DataFrame,
    depth_col: str = "DEPT",
    well_name: str = "Well",
) -> plt.Figure:
    """
    Produce a static multi-track composite well log using matplotlib.

    Tracks (left to right):
      1. GR (shaded sand vs. shale)
      2. RDEEP (log scale)
      3. NPHI / RHOB overlay
      4. Porosity
      5. Permeability (log scale) — labelled "Illustrative only"
      6. Sw
      7. Vsh methods comparison
      8. Facies colour column (if 'facies' column exists)

    Args:
        df (pd.DataFrame): Processed well log DataFrame.
        depth_col (str): Depth column name.
        well_name (str): Title label for the plot.

    Returns:
        matplotlib.figure.Figure: Composite log figure.
    """
    depth = df[depth_col]

    n_tracks = 7 + (1 if "facies" in df.columns else 0)
    fig, axes = plt.subplots(
        1, n_tracks,
        figsize=(3.5 * n_tracks, 12),
        sharey=True,
    )
    fig.suptitle(f"LOGVIP — Composite Log: {well_name}", fontsize=13, y=1.01)

    # Common settings
    for ax in axes:
        ax.set_ylim(depth.max(), depth.min())
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

    track = 0

    # ── Track 1: GR ─────────────────────────────────────────────────────────
    ax = axes[track]; track += 1
    if "GR" in df.columns:
        ax.plot(df["GR"], depth, color="#7cba59", linewidth=0.8)
        ax.set_xlabel("GR (gAPI)")
        ax.set_xlim(0, 150)
        ax.set_title("GR", fontsize=9)
    ax.set_ylabel("Depth (m)")

    # ── Track 2: RDEEP (log scale) ─────────────────────────────────────────
    ax = axes[track]; track += 1
    if "RDEEP" in df.columns:
        ax.semilogx(df["RDEEP"], depth, color="#e34234", linewidth=0.8)
        ax.set_xlabel("RDEEP (Ω·m)")
        ax.set_title("Resistivity", fontsize=9)

    # ── Track 3: NPHI / RHOB overlay ───────────────────────────────────────
    ax = axes[track]; track += 1
    ax2 = ax.twiny()
    if "NPHI" in df.columns:
        ax.plot(df["NPHI"], depth, color="#2196f3", linewidth=0.8, label="NPHI")
        ax.set_xlabel("NPHI (frac)", color="#2196f3", fontsize=8)
        ax.set_xlim(0.45, -0.05)
    if "RHOB" in df.columns:
        ax2.plot(df["RHOB"], depth, color="#f44336", linewidth=0.8, label="RHOB")
        ax2.set_xlabel("RHOB (g/cc)", color="#f44336", fontsize=8)
        ax2.set_xlim(1.95, 2.95)
    ax.set_title("NPHI / RHOB", fontsize=9)

    # ── Track 4: Porosity ──────────────────────────────────────────────────
    ax = axes[track]; track += 1
    if "porosity" in df.columns:
        ax.fill_betweenx(depth, 0, df["porosity"], alpha=0.4, color="#00bcd4")
        ax.plot(df["porosity"], depth, color="#00bcd4", linewidth=0.8)
        ax.set_xlabel("φ_eff (frac)")
        ax.set_xlim(0, 0.45)
    ax.set_title("Porosity", fontsize=9)

    # ── Track 5: Permeability (log scale) ──────────────────────────────────
    ax = axes[track]; track += 1
    if "permeability" in df.columns:
        k_vals = df["permeability"].clip(lower=0.001)
        ax.semilogx(k_vals, depth, color="#9c27b0", linewidth=0.8)
        ax.set_xlabel("k (mD) [Illustrative]")
        ax.set_title("Permeability\n⚠ Illustrative only", fontsize=8, color="#9c27b0")

    # ── Track 6: Sw ────────────────────────────────────────────────────────
    ax = axes[track]; track += 1
    if "Sw" in df.columns:
        ax.fill_betweenx(depth, 0, df["Sw"], alpha=0.3, color="#2196f3")
        ax.plot(df["Sw"], depth, color="#2196f3", linewidth=0.8)
        ax.axvline(0.70, color="red", linestyle="--", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("Sw (frac)")
        ax.set_xlim(0, 1)
    ax.set_title("Water Sat.", fontsize=9)

    # ── Track 7: Vsh comparison ─────────────────────────────────────────────
    ax = axes[track]; track += 1
    for col, color, label in [
        ("Vsh_larionov_t", "#ff5722", "Larionov-T"),
        ("Vsh_larionov_o", "#795548", "Larionov-O"),
        ("Vsh_steiber",    "#607d8b", "Steiber"),
    ]:
        if col in df.columns:
            ax.plot(df[col], depth, color=color, linewidth=0.7, label=label)
    ax.set_xlabel("Vsh (frac)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Vsh Methods", fontsize=9)

    # ── Track 8: Facies column (optional) ─────────────────────────────────
    if "facies" in df.columns and track < n_tracks:
        ax = axes[track]; track += 1
        labels = df["facies"].values
        unique = sorted(set(labels[labels >= 0]))
        for i, row in df.iterrows():
            f = int(row["facies"])
            if f < 0:
                continue
            color = FACIES_COLORS[f % len(FACIES_COLORS)]
            y_top = row[depth_col]
            y_bot = y_top + 0.5
            ax.barh(y=(y_top + y_bot) / 2, width=1, height=y_bot - y_top,
                    color=color, edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_title("Facies", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Plotly log tracks
# ─────────────────────────────────────────────────────────────────────────────

def plotly_log_tracks(
    df: pd.DataFrame,
    depth_col: str = "DEPT",
    well_name: str = "Well",
) -> go.Figure:
    """
    Build an interactive multi-track well log figure using Plotly subplots.

    Tracks match the static log plot. Hover shows exact depth and value.

    Args:
        df (pd.DataFrame): Processed well log DataFrame.
        depth_col (str): Depth column name.
        well_name (str): Title for the figure.

    Returns:
        plotly.graph_objects.Figure
    """
    depth = df[depth_col]
    has_facies = "facies" in df.columns
    n_tracks = 7 + (1 if has_facies else 0)

    fig = make_subplots(
        rows=1, cols=n_tracks,
        shared_yaxes=True,
        subplot_titles=["GR", "RDEEP", "NPHI/RHOB", "Porosity",
                        "Perm (Illus.)", "Sw", "Vsh"] + (["Facies"] if has_facies else []),
        horizontal_spacing=0.01,
    )

    col = 1

    # GR
    if "GR" in df.columns:
        fig.add_trace(go.Scatter(x=df["GR"], y=depth, mode="lines",
                                  name="GR", line=dict(color="#7cba59", width=1)),
                      row=1, col=col)
    col += 1

    # RDEEP (log x-axis set below)
    if "RDEEP" in df.columns:
        fig.add_trace(go.Scatter(x=df["RDEEP"], y=depth, mode="lines",
                                  name="RDEEP", line=dict(color="#e34234", width=1)),
                      row=1, col=col)
        fig.update_xaxes(type="log", row=1, col=col)
    col += 1

    # NPHI + RHOB overlay
    if "NPHI" in df.columns:
        fig.add_trace(go.Scatter(x=df["NPHI"], y=depth, mode="lines",
                                  name="NPHI", line=dict(color="#2196f3", width=1)),
                      row=1, col=col)
    if "RHOB" in df.columns:
        fig.add_trace(go.Scatter(x=df["RHOB"], y=depth, mode="lines",
                                  name="RHOB", line=dict(color="#f44336", width=1, dash="dot")),
                      row=1, col=col)
    col += 1

    # Porosity
    if "porosity" in df.columns:
        fig.add_trace(go.Scatter(x=df["porosity"], y=depth, mode="lines",
                                  name="φ_eff", line=dict(color="#00bcd4", width=1),
                                  fill="tozerox", fillcolor="rgba(0,188,212,0.2)"),
                      row=1, col=col)
    col += 1

    # Permeability
    if "permeability" in df.columns:
        k_vals = df["permeability"].clip(lower=0.001)
        fig.add_trace(go.Scatter(x=k_vals, y=depth, mode="lines",
                                  name="k [Illus.]", line=dict(color="#9c27b0", width=1)),
                      row=1, col=col)
        fig.update_xaxes(type="log", row=1, col=col)
    col += 1

    # Sw
    if "Sw" in df.columns:
        fig.add_trace(go.Scatter(x=df["Sw"], y=depth, mode="lines",
                                  name="Sw", line=dict(color="#2196f3", width=1),
                                  fill="tozerox", fillcolor="rgba(33,150,243,0.15)"),
                      row=1, col=col)
        # Cutoff line
        fig.add_shape(type="line", x0=0.70, x1=0.70, y0=depth.min(), y1=depth.max(),
                      line=dict(color="red", dash="dash", width=1),
                      row=1, col=col)
    col += 1

    # Vsh comparison
    for series, color, label in [
        ("Vsh_larionov_t", "#ff5722", "Larionov-T"),
        ("Vsh_larionov_o", "#795548", "Larionov-O"),
        ("Vsh_steiber",    "#607d8b", "Steiber"),
    ]:
        if series in df.columns:
            fig.add_trace(go.Scatter(x=df[series], y=depth, mode="lines",
                                      name=label, line=dict(color=color, width=1)),
                          row=1, col=col)
    col += 1

    # Facies
    if has_facies:
        labels = df["facies"].values
        unique = sorted(set(labels[labels >= 0]))
        for f in unique:
            mask = labels == f
            fig.add_trace(go.Bar(
                x=[1] * mask.sum(),
                y=depth[mask],
                orientation="h",
                name=f"Facies {f}",
                marker_color=FACIES_COLORS[f % len(FACIES_COLORS)],
                width=0.5,
                showlegend=True,
            ), row=1, col=col)

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", col=1)
    fig.update_layout(
        title=f"LOGVIP — Interactive Log: {well_name}",
        height=700,
        margin=dict(t=80, l=60, r=20, b=40),
        legend=dict(orientation="h", y=-0.08),
        hovermode="y unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Density-neutron crossplot
# ─────────────────────────────────────────────────────────────────────────────

def plotly_crossplot(
    df: pd.DataFrame,
    depth_col: str = "DEPT",
    show_corrected: bool = True,
) -> go.Figure:
    """
    Interactive density-neutron crossplot with standard lithology lines.

    Shows raw and shale-corrected logs side by side (or as selectable traces).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        depth_col (str): Depth column for hover labels.
        show_corrected (bool): If True and RHOB_ns/NPHI_ns exist, add corrected traces.

    Returns:
        plotly.graph_objects.Figure
    """
    phi_line = np.linspace(0, 0.45, 200)
    rho_f = 1.0

    lithologies = {
        "Sandstone": {"rho_ma": 2.65, "phiN_ma": -0.02},
        "Limestone": {"rho_ma": 2.71, "phiN_ma":  0.00},
        "Dolomite":  {"rho_ma": 2.87, "phiN_ma":  0.02},
    }

    fig = go.Figure()

    # Lithology lines
    for name, props in lithologies.items():
        rho_line  = phi_line * rho_f + (1 - phi_line) * props["rho_ma"]
        phiN_line = (1 - phi_line) * props["phiN_ma"]  + phi_line * 1.0
        fig.add_trace(go.Scatter(
            x=phiN_line, y=rho_line,
            mode="lines", name=name,
            line=dict(width=2),
        ))

    # Raw data
    if "NPHI" in df.columns and "RHOB" in df.columns:
        hover = df[depth_col].round(1).astype(str) + " m" if depth_col in df.columns else None
        fig.add_trace(go.Scatter(
            x=df["NPHI"], y=df["RHOB"],
            mode="markers",
            name="Raw",
            marker=dict(size=4, color="#607d8b", opacity=0.6),
            text=hover,
            hovertemplate="NPHI=%{x:.3f}, RHOB=%{y:.3f}<br>%{text}<extra>Raw</extra>",
        ))

    # Corrected data
    if show_corrected and "NPHI_ns" in df.columns and "RHOB_ns" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["NPHI_ns"], y=df["RHOB_ns"],
            mode="markers",
            name="Shale-corrected",
            marker=dict(size=4, color="#e91e63", opacity=0.6),
            hovertemplate="NPHI_ns=%{x:.3f}, RHOB_ns=%{y:.3f}<extra>Corrected</extra>",
        ))

    fig.update_layout(
        title="Neutron–Density Crossplot",
        xaxis=dict(title="NPHI (fraction)", range=[-0.05, 0.45]),
        yaxis=dict(title="RHOB (g/cc)", range=[3.0, 1.9]),
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Vsh method comparison
# ─────────────────────────────────────────────────────────────────────────────

def plotly_vsh_comparison(
    df: pd.DataFrame,
    depth_col: str = "DEPT",
) -> go.Figure:
    """
    Overlay the three Vsh methods on a single depth-track plot.

    Expected ranking (PRD §9 C.11): Larionov-T ≤ Steiber ≤ Larionov-O.

    Args:
        df (pd.DataFrame): DataFrame with Vsh_larionov_t, Vsh_larionov_o, Vsh_steiber.
        depth_col (str): Depth column name.

    Returns:
        plotly.graph_objects.Figure
    """
    depth = df[depth_col]
    fig = go.Figure()

    for col, color, label in [
        ("Vsh_larionov_t", "#ff5722", "Larionov-Tertiary"),
        ("Vsh_steiber",    "#607d8b", "Steiber"),
        ("Vsh_larionov_o", "#795548", "Larionov-Older"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[col], y=depth,
                mode="lines", name=label,
                line=dict(color=color, width=1.5),
            ))

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    fig.update_xaxes(range=[0, 1], title_text="Vsh (fraction)")
    fig.update_layout(
        title="Shale Volume Methods Comparison",
        height=500,
        hovermode="y unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Elbow and silhouette plots for facies k selection
# ─────────────────────────────────────────────────────────────────────────────

def plotly_elbow(
    elbow_df: pd.DataFrame,
    silhouette_df: pd.DataFrame = None,
) -> go.Figure:
    """
    Combined elbow + silhouette plot for facies cluster count selection.

    Args:
        elbow_df (pd.DataFrame): Output of facies.elbow_data() with cols ['k', 'inertia'].
        silhouette_df (pd.DataFrame): Output of facies.silhouette_data() with
            cols ['k', 'silhouette_score']. Optional.

    Returns:
        plotly.graph_objects.Figure
    """
    if silhouette_df is not None:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Elbow (Inertia)", "Silhouette Score"])
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Elbow (Inertia)"])

    fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"],
        mode="lines+markers", name="Inertia",
        line=dict(color="#4e79a7"),
    ), row=1, col=1)

    if silhouette_df is not None:
        fig.add_trace(go.Scatter(
            x=silhouette_df["k"], y=silhouette_df["silhouette_score"],
            mode="lines+markers", name="Silhouette",
            line=dict(color="#f28e2b"),
        ), row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)

    fig.update_xaxes(title_text="k (clusters)", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_layout(title="Facies Cluster Count Selection", height=380)
    return fig
