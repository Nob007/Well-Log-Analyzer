"""
petrophysics.py — Core petrophysical calculations for LOGVIP.

Implements, per depth sample:
  - Gamma-ray index (Igr)
  - Shale volume: Larionov-Tertiary, Larionov-Older, Steiber
  - Shale endpoint estimation (robust: top-5th-percentile mean)
  - Shale-corrected NPHI / RHOB
  - Density porosity (from shale-corrected RHOB)
  - Effective porosity (average of corrected NPHI + density porosity)
  - Water saturation (Simandoux)
  - Permeability (Timur empirical correlation)
  - Net-sand / net-reservoir / net-pay cutoff flags
  - Full pipeline orchestrator: run_petrophysics_pipeline()

Bugs fixed vs. source notebook / original sub-modules:
  - Average porosity uses (NPHI_ns + PHID_ns)/2, not minus.
  - Sw.clip() uses assignment form — no deprecated inplace=True on slice.
  - GR baselines use 5th/95th percentile, not raw min/max (spike-robust).
  - Shale endpoint uses top-5th-percentile mean of Vsh rows, not Vsh==1 exact match.
  - run_petrophysics_pipeline() enforces dependency order: no KeyError risk.

Units throughout:
  - Depth: metres (m)
  - GR: gAPI
  - NPHI, porosity, Vsh, Sw: fraction (dimensionless, 0–1)
  - RHOB: g/cc
  - RDEEP/Rsh/Rw: ohm·m
  - Permeability: millidarcies (mD)

Approved libraries: numpy, pandas, scipy only.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gamma-ray index
# ─────────────────────────────────────────────────────────────────────────────

def compute_igr(
    df: pd.DataFrame,
    gr_col: str = "GR",
    gr_sand: float = None,
    gr_shale: float = None,
    percentile_method: bool = True,
) -> pd.Series:
    """
    Compute the gamma-ray index (Igr), clipped to [0, 1].

    Formula:
        Igr = (GR - GR_sand) / (GR_shale - GR_sand)

    Args:
        df (pd.DataFrame): Well log DataFrame.
        gr_col (str): Name of the GR column.
        gr_sand (float): GR value representing clean sand. If None, estimated
            from the 5th percentile of the GR log (PRD §9 B.6).
        gr_shale (float): GR value representing pure shale. If None, estimated
            from the 95th percentile of the GR log (PRD §9 B.6).
        percentile_method (bool): If True and gr_sand/gr_shale are None, use
            5th/95th percentile. Ignored when explicit values are supplied.

    Returns:
        pd.Series: Igr values, clipped to [0, 1], with original index.
    """
    gr = df[gr_col].copy()
    if gr_sand is None:
        gr_sand = float(np.nanpercentile(gr.dropna(), 5))
        logger.info("GR_sand estimated from 5th percentile: %.2f gAPI", gr_sand)
    if gr_shale is None:
        gr_shale = float(np.nanpercentile(gr.dropna(), 95))
        logger.info("GR_shale estimated from 95th percentile: %.2f gAPI", gr_shale)

    if gr_shale == gr_sand:
        logger.warning("GR_shale == GR_sand; Igr set to 0.5 everywhere.")
        return pd.Series(0.5, index=df.index)

    igr = (gr - gr_sand) / (gr_shale - gr_sand)
    return igr.clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Shale volume models
# ─────────────────────────────────────────────────────────────────────────────

def compute_vsh_larionov_tertiary(igr: pd.Series) -> pd.Series:
    """
    Larionov (1969) shale volume for Tertiary / younger rocks.

    Formula:
        Vsh = 0.083 * (2^(3.7 * Igr) - 1)
    Clipped to [0, 1].

    Args:
        igr (pd.Series): Gamma-ray index, values in [0, 1].

    Returns:
        pd.Series: Shale volume fraction in [0, 1].
    """
    vsh = 0.083 * (2.0 ** (3.7 * igr) - 1.0)
    return vsh.clip(0.0, 1.0)


def compute_vsh_larionov_older(igr: pd.Series) -> pd.Series:
    """
    Larionov (1969) shale volume for pre-Tertiary / older rocks.

    Formula:
        Vsh = 0.33 * (2^(2 * Igr) - 1)
    Clipped to [0, 1].

    Args:
        igr (pd.Series): Gamma-ray index, values in [0, 1].

    Returns:
        pd.Series: Shale volume fraction in [0, 1].
    """
    vsh = 0.33 * (2.0 ** (2.0 * igr) - 1.0)
    return vsh.clip(0.0, 1.0)


def compute_vsh_steiber(igr: pd.Series) -> pd.Series:
    """
    Steiber (1973) shale volume for dispersed / laminated shales.

    Formula:
        Vsh = Igr / (3 - 2 * Igr)
    Clipped to [0, 1].

    Args:
        igr (pd.Series): Gamma-ray index, values in [0, 1].

    Returns:
        pd.Series: Shale volume fraction in [0, 1].
    """
    denom = 3.0 - 2.0 * igr
    vsh = igr / denom.replace(0.0, np.nan)
    return vsh.clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Shale endpoint estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shale_endpoints(
    df: pd.DataFrame,
    vsh_col: str = "Vsh",
    log_cols: list = None,
    percentile: float = 95.0,
) -> dict:
    """
    Estimate shale log values from the high-Vsh rows.

    Uses the top (100 - percentile)th fraction of rows sorted by Vsh
    (e.g., top 5% when percentile=95) and returns their mean log values.
    This is more robust than finding a single row where Vsh == 1.0 exactly
    (PRD §9 B.5).

    Args:
        df (pd.DataFrame): Well log DataFrame containing vsh_col and log_cols.
        vsh_col (str): Name of the shale volume column.
        log_cols (list): Log columns to compute shale endpoints for.
            Defaults to ['NPHI', 'RHOB', 'RDEEP'].
        percentile (float): Lower bound percentile for the "shale zone"
            (default 95 → top 5% of Vsh rows).

    Returns:
        dict: Mapping from log name to shale endpoint value (float).
            Example: {'NPHI': 0.33, 'RHOB': 2.68, 'RDEEP': 1.2}
    """
    if log_cols is None:
        log_cols = ["NPHI", "RHOB", "RDEEP"]

    threshold = np.nanpercentile(df[vsh_col].dropna(), percentile)
    shale_rows = df[df[vsh_col] >= threshold]

    n_shale = len(shale_rows)
    if n_shale == 0:
        logger.warning(
            "No rows found with Vsh >= %.2f; shale endpoints set to NaN.", threshold
        )

    endpoints = {}
    for col in log_cols:
        if col in df.columns:
            val = float(shale_rows[col].mean())
            endpoints[col] = val
            logger.info("Shale endpoint %s = %.4f (from %d rows)", col, val, n_shale)
        else:
            logger.warning("Column '%s' not found for shale endpoint.", col)

    return endpoints


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Shale-corrected logs
# ─────────────────────────────────────────────────────────────────────────────

def apply_shale_correction(
    df: pd.DataFrame,
    vsh_col: str = "Vsh",
    shale_endpoints: dict = None,
    log_cols: list = None,
) -> pd.DataFrame:
    """
    Subtract the shale volume contribution from each specified log.

    Formula (per log L):
        L_ns = L - Vsh * L_shale

    The corrected column is named with a '_ns' suffix (non-shale).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        vsh_col (str): Shale volume column name.
        shale_endpoints (dict): Dict of {log_name: shale_value} from
            compute_shale_endpoints(). Required.
        log_cols (list): Logs to correct. Defaults to ['NPHI', 'RHOB'].

    Returns:
        pd.DataFrame: df with new '<log>_ns' columns added.
    """
    if log_cols is None:
        log_cols = ["NPHI", "RHOB"]
    if shale_endpoints is None:
        raise ValueError("shale_endpoints must be provided (use compute_shale_endpoints).")

    for col in log_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not found; skipping shale correction.", col)
            continue
        val_sh = shale_endpoints.get(col, np.nan)
        df[f"{col}_ns"] = df[col] - df[vsh_col] * val_sh

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Porosity
# ─────────────────────────────────────────────────────────────────────────────

def compute_density_porosity(
    df: pd.DataFrame,
    rhob_ns_col: str = "RHOB_ns",
    rho_matrix: float = 2.65,
    rho_fluid: float = 1.0,
) -> pd.Series:
    """
    Density-derived porosity from the shale-corrected bulk density log.

    Formula:
        PHID = (rho_matrix - RHOB_ns) / (rho_matrix - rho_fluid)
    Clipped to [0, 0.47].

    Assumptions (confirm against mud report / lithology):
        rho_matrix = 2.65 g/cc  (clean sandstone; PRD §8)
        rho_fluid  = 1.0  g/cc  (fresh water; PRD §8)

    Args:
        df (pd.DataFrame): Well log DataFrame.
        rhob_ns_col (str): Shale-corrected RHOB column name.
        rho_matrix (float): Matrix density in g/cc.
        rho_fluid (float): Fluid density in g/cc.

    Returns:
        pd.Series: Density porosity, clipped to [0, 0.47].
    """
    denom = rho_matrix - rho_fluid
    if denom == 0:
        raise ValueError("rho_matrix and rho_fluid are equal; cannot compute PHID.")
    phid = (rho_matrix - df[rhob_ns_col]) / denom
    return phid.clip(0.0, 0.47)


def compute_effective_porosity(
    df: pd.DataFrame,
    nphi_ns_col: str = "NPHI_ns",
    phid_col: str = "PHID",
) -> pd.Series:
    """
    Effective porosity as the arithmetic mean of shale-corrected neutron
    and density porosities.

    Formula:
        phi_eff = (NPHI_ns + PHID) / 2
    Clipped to [0, 0.47].

    Args:
        df (pd.DataFrame): Well log DataFrame.
        nphi_ns_col (str): Shale-corrected neutron porosity column.
        phid_col (str): Density porosity column.

    Returns:
        pd.Series: Effective porosity in [0, 0.47].
    """
    phi_eff = (df[nphi_ns_col] + df[phid_col]) / 2.0
    return phi_eff.clip(0.0, 0.47)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Water saturation — Simandoux
# ─────────────────────────────────────────────────────────────────────────────

def compute_simandoux_sw(
    df: pd.DataFrame,
    porosity_col: str = "porosity",
    igr_col: str = "Igr",
    rdeep_col: str = "RDEEP",
    Rw: float = 0.135,
    Rsh: float = 4.0,
    C: float = 0.4,
) -> pd.Series:
    """
    Water saturation via the Simandoux equation.

    Formula:
        Sw = (C*Rw/phi^2) * (-Igr/Rsh + sqrt((5*phi^2/(Rw*Rt)) + (Igr/Rsh)^2))
    Clipped to [0, 1].

    NOTE: Rw = 0.135 ohm·m and C = 0.4 are notebook defaults (PRD §8).
          Confirm against a water sample / SP log before trusting Sw results.

    Args:
        df (pd.DataFrame): Well log DataFrame.
        porosity_col (str): Effective porosity column name.
        igr_col (str): Gamma-ray index column name.
        rdeep_col (str): Deep resistivity column name.
        Rw (float): Formation water resistivity (ohm·m).
        Rsh (float): Shale resistivity (ohm·m).
        C (float): Simandoux constant (dimensionless).

    Returns:
        pd.Series: Water saturation, clipped to [0, 1].
    """
    phi = df[porosity_col]
    igr = df[igr_col]
    rt = df[rdeep_col]

    # Avoid division by zero for zero-porosity rows
    phi_safe = phi.replace(0.0, np.nan)

    term_a = C * Rw / (phi_safe ** 2)
    term_b = -igr / Rsh
    term_c = np.sqrt(
        (5.0 * phi_safe ** 2 / (Rw * rt)) + (igr / Rsh) ** 2
    )
    sw = term_a * (term_b + term_c)
    # Assignment clip — avoids deprecated inplace=True on a column slice
    sw = sw.clip(0.0, 1.0)
    return sw


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Permeability — Timur empirical correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_permeability_timur(
    df: pd.DataFrame,
    porosity_col: str = "porosity",
    sw_col: str = "Sw",
    swirr: float = None,
    c: float = 0.136,
    d: float = 4.4,
    e: float = 2.0,
) -> pd.Series:
    """
    Permeability via the Timur (1968) empirical correlation.

    Formula:
        k = c * (phi^d) / (Swirr^e)      [mD]

    ⚠️ WARNING: The training points in the source notebook are SYNTHETIC
       placeholder numbers, NOT measured core or DST data for this well.
       All permeability output from this function is ILLUSTRATIVE ONLY and
       must NOT be used for volumetric decisions until calibrated against
       real core / DST permeability data (PRD §8 and §9 B.8).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        porosity_col (str): Effective porosity column.
        sw_col (str): Water saturation column (used to estimate Swirr if
            swirr is not provided).
        swirr (float): Irreducible water saturation. If None, uses the
            minimum non-zero Sw in the log.
        c (float): Timur constant (0.136 gives k in mD).
        d (float): Porosity exponent (default 4.4).
        e (float): Swirr exponent (default 2.0).

    Returns:
        pd.Series: Permeability in mD (values >= 0).
    """
    phi = df[porosity_col]
    if swirr is None:
        sw_vals = df[sw_col]
        positive_sw = sw_vals[sw_vals > 0.0]
        swirr = float(positive_sw.min()) if len(positive_sw) > 0 else 0.1
        logger.info("Swirr estimated from minimum Sw: %.4f", swirr)

    if swirr <= 0.0:
        swirr = 0.01
        logger.warning("Swirr <= 0; clamped to 0.01 to avoid division by zero.")

    k = c * (phi ** d) / (swirr ** e)
    k = k.clip(lower=0.0)

    # Flag suspiciously large values (>10,000 mD in shaly sand is unphysical)
    n_large = int((k > 10000.0).sum())
    if n_large > 0:
        logger.warning(
            "%d rows have permeability > 10,000 mD — check porosity or Swirr inputs.",
            n_large,
        )

    return k


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Net-sand / net-reservoir / net-pay flags
# ─────────────────────────────────────────────────────────────────────────────

def compute_cutoff_flags(
    df: pd.DataFrame,
    igr_col: str = "Igr",
    porosity_col: str = "porosity",
    sw_col: str = "Sw",
    igr_cut: float = 0.5,
    phi_cut: float = 0.15,
    sw_cut: float = 0.70,
) -> pd.DataFrame:
    """
    Compute boolean flags for net-sand, net-reservoir, and net-pay zones.

    Cutoffs (industry defaults — confirm for this field; PRD §8):
        net_sand       : Igr < igr_cut                          (shale indicator)
        net_reservoir  : net_sand AND porosity >= phi_cut        (flow capacity)
        net_pay        : net_reservoir AND Sw < sw_cut           (hydrocarbon bearing)

    The nested structure is asserted: pay ⊆ reservoir ⊆ sand (PRD §9 C.13).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        igr_col (str): Gamma-ray index column.
        porosity_col (str): Effective porosity column.
        sw_col (str): Water saturation column.
        igr_cut (float): Maximum Igr for net-sand (default 0.5).
        phi_cut (float): Minimum porosity for net-reservoir (default 0.15).
        sw_cut (float): Maximum Sw for net-pay (default 0.70).

    Returns:
        pd.DataFrame: Three boolean columns added to a copy of df:
            'net_sand', 'net_reservoir', 'net_pay'.
    """
    out = df.copy()
    out["net_sand"] = out[igr_col] < igr_cut
    out["net_reservoir"] = out["net_sand"] & (out[porosity_col] >= phi_cut)
    out["net_pay"] = out["net_reservoir"] & (out[sw_col] < sw_cut)

    # Sanity check: nesting should always hold
    assert (
        out["net_pay"].sum() <= out["net_reservoir"].sum() <= out["net_sand"].sum()
    ), "BUG: net-pay/reservoir/sand nesting violated — check cutoffs."

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Full pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_petrophysics_pipeline(
    df: pd.DataFrame,
    params: dict = None,
) -> pd.DataFrame:
    """
    Run the complete petrophysics pipeline on a well log DataFrame.

    Functions are called in strict dependency order so no KeyError can
    occur from out-of-order execution (PRD §9 A.2 / A.3).

    Derived columns added to the returned DataFrame:
        Igr, Vsh_larionov_t, Vsh_larionov_o, Vsh, Vsh_steiber,
        NPHI_ns, RHOB_ns, PHID, porosity, Sw, permeability,
        net_sand, net_reservoir, net_pay

    Args:
        df (pd.DataFrame): Well log DataFrame from io_utils.load_log_file().
        params (dict): Optional override dictionary. Supported keys and defaults::

            gr_col       = 'GR'
            nphi_col     = 'NPHI'
            rhob_col     = 'RHOB'
            rdeep_col    = 'RDEEP'
            gr_sand      = None   (→ 5th percentile)
            gr_shale     = None   (→ 95th percentile)
            rho_matrix   = 2.65   (g/cc; sandstone)
            rho_fluid    = 1.0    (g/cc; fresh water)
            Rw           = 0.135  (ohm·m; confirm from SP/water sample)
            Rsh          = 4.0    (ohm·m; from shale endpoint)
            C            = 0.4    (Simandoux constant)
            igr_cut      = 0.5
            phi_cut      = 0.15
            sw_cut       = 0.70
            vsh_method   = 'larionov_t'  ('larionov_t' | 'larionov_o' | 'steiber')

    Returns:
        pd.DataFrame: Input DataFrame with all derived petrophysical columns.
    """
    p = {
        "gr_col":     "GR",
        "nphi_col":   "NPHI",
        "rhob_col":   "RHOB",
        "rdeep_col":  "RDEEP",
        "gr_sand":    None,
        "gr_shale":   None,
        "rho_matrix": 2.65,
        "rho_fluid":  1.0,
        "Rw":         0.135,
        "Rsh":        4.0,
        "C":          0.4,
        "igr_cut":    0.5,
        "phi_cut":    0.15,
        "sw_cut":     0.70,
        "vsh_method": "larionov_t",
    }
    if params:
        p.update(params)

    df = df.copy()

    # Step 1: Igr
    df["Igr"] = compute_igr(
        df, gr_col=p["gr_col"],
        gr_sand=p["gr_sand"],
        gr_shale=p["gr_shale"],
    )

    # Step 2: Vsh (all three; primary selected by vsh_method)
    df["Vsh_larionov_t"] = compute_vsh_larionov_tertiary(df["Igr"])
    df["Vsh_larionov_o"] = compute_vsh_larionov_older(df["Igr"])
    df["Vsh_steiber"]    = compute_vsh_steiber(df["Igr"])

    vsh_map = {
        "larionov_t": "Vsh_larionov_t",
        "larionov_o": "Vsh_larionov_o",
        "steiber":    "Vsh_steiber",
    }
    primary_vsh_col = vsh_map.get(p["vsh_method"], "Vsh_larionov_t")
    df["Vsh"] = df[primary_vsh_col]

    # Step 3: Shale endpoints from top-5% Vsh rows
    endpoints = compute_shale_endpoints(
        df, vsh_col="Vsh",
        log_cols=[p["nphi_col"], p["rhob_col"], p["rdeep_col"]],
    )
    # Rsh from shale endpoint (use explicit param if provided, else derived)
    rsh = endpoints.get(p["rdeep_col"], p["Rsh"])

    # Step 4: Shale-corrected logs
    df = apply_shale_correction(
        df, vsh_col="Vsh",
        shale_endpoints=endpoints,
        log_cols=[p["nphi_col"], p["rhob_col"]],
    )
    nphi_ns_col = f"{p['nphi_col']}_ns"
    rhob_ns_col = f"{p['rhob_col']}_ns"

    # Step 5: Density porosity
    df["PHID"] = compute_density_porosity(
        df, rhob_ns_col=rhob_ns_col,
        rho_matrix=p["rho_matrix"],
        rho_fluid=p["rho_fluid"],
    )

    # Step 6: Effective porosity
    df["porosity"] = compute_effective_porosity(
        df, nphi_ns_col=nphi_ns_col, phid_col="PHID"
    )

    # Step 7: Water saturation (Simandoux)
    df["Sw"] = compute_simandoux_sw(
        df,
        porosity_col="porosity",
        igr_col="Igr",
        rdeep_col=p["rdeep_col"],
        Rw=p["Rw"],
        Rsh=rsh,
        C=p["C"],
    )

    # Step 8: Permeability (Timur)
    df["permeability"] = compute_permeability_timur(
        df, porosity_col="porosity", sw_col="Sw"
    )

    # Step 9: Cutoff flags
    df = compute_cutoff_flags(
        df,
        igr_col="Igr",
        porosity_col="porosity",
        sw_col="Sw",
        igr_cut=p["igr_cut"],
        phi_cut=p["phi_cut"],
        sw_cut=p["sw_cut"],
    )

    # Step 10: Sanity assertions (PRD §9 C.9 / C.10)
    for col in ["Vsh", "Vsh_larionov_t", "Vsh_larionov_o", "Vsh_steiber", "porosity", "Sw"]:
        vals = df[col].dropna()
        assert vals.between(0.0, 1.0).all(), (
            f"BUG: '{col}' has values outside [0, 1] after pipeline."
        )
    assert (df["permeability"].dropna() >= 0.0).all(), (
        "BUG: negative permeability detected."
    )

    return df
