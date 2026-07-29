"""
lithology.py — Lithology estimation for LOGVIP well log tool.

Provides:
  - Two-mineral (sand/shale) split from Igr.
  - Multi-mineral split (sand, shale, calcite, heavy mineral) via
    scipy.optimize.minimize (SLSQP), matching RHOB and NPHI to mineral
    end-members subject to volumes summing to 1.

Mineral end-member constants (from standard petrophysical references):
  Sand   → Quartz:          rho = 2.65 g/cc, nphi = -0.02 (fraction)
  Shale  → Average shale:   rho = 2.71 g/cc, nphi =  0.33
  Calcite → Pure calcite:   rho = 2.71 g/cc, nphi =  0.00
  Heavy  → Heavy minerals:  rho = 5.01 g/cc, nphi =  0.00

Approved libraries: numpy, pandas, scipy only.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mineral end-member constants
# Source: standard petrophysical references (Schlumberger Log Interpretation
# Charts, 2013; Ellis & Singer, 2007).
# ─────────────────────────────────────────────────────────────────────────────
MINERAL_ENDMEMBERS = {
    # key: (rho_g_per_cc, nphi_fraction)
    "sand":    {"rho": 2.65, "nphi": -0.02},  # Quartz
    "shale":   {"rho": 2.71, "nphi":  0.33},  # Average shale
    "calcite": {"rho": 2.71, "nphi":  0.00},  # Calcite / limestone
    "heavy":   {"rho": 5.01, "nphi":  0.00},  # Heavy minerals (e.g., pyrite)
}


def two_mineral_split(
    df: pd.DataFrame,
    igr_col: str = "Igr",
) -> pd.DataFrame:
    """
    Compute a two-mineral (sand / shale) volume split from the gamma-ray index.

    Formula:
        V_shale = Igr          (already clipped to [0, 1])
        V_sand  = 1 - V_shale

    This is the simplest lithology estimator; it is equivalent to the linear
    shale model applied to Igr.

    Args:
        df (pd.DataFrame): Well log DataFrame containing igr_col.
        igr_col (str): Name of the gamma-ray index column (values in [0, 1]).

    Returns:
        pd.DataFrame: df with two new columns: 'lith_sand', 'lith_shale'.
    """
    df = df.copy()
    df["lith_shale"] = df[igr_col].clip(0.0, 1.0)
    df["lith_sand"]  = 1.0 - df["lith_shale"]
    return df


def _solve_one_row(
    rhob: float,
    nphi: float,
    minerals: list,
    endmembers: dict,
) -> tuple:
    """
    Solve the four-mineral volume fractions for a single depth sample.

    Minimises the squared residual between observed (rhob, nphi) and the
    mixture model, subject to:
        sum(volumes) == 1
        each volume in [0, 1]

    Args:
        rhob (float): Observed bulk density (g/cc).
        nphi (float): Observed neutron porosity (fraction).
        minerals (list): Ordered list of mineral names (keys of endmembers).
        endmembers (dict): Mineral end-member dict (MINERAL_ENDMEMBERS).

    Returns:
        tuple: (volumes_array, success_bool)
    """
    rho_em = np.array([endmembers[m]["rho"]  for m in minerals])
    phi_em = np.array([endmembers[m]["nphi"] for m in minerals])

    def objective(v):
        rhob_calc = np.dot(v, rho_em)
        nphi_calc = np.dot(v, phi_em)
        return (rhob_calc - rhob) ** 2 + (nphi_calc - nphi) ** 2

    constraints = [{"type": "eq", "fun": lambda v: np.sum(v) - 1.0}]
    bounds = [(0.0, 1.0)] * len(minerals)
    x0 = np.ones(len(minerals)) / len(minerals)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )
    return result.x, result.success


def multimineral_split(
    df: pd.DataFrame,
    rhob_col: str = "RHOB",
    nphi_col: str = "NPHI",
    minerals: list = None,
    endmembers: dict = None,
) -> pd.DataFrame:
    """
    Per-row multi-mineral volume split using SLSQP optimisation.

    Matches RHOB and NPHI to a mixture of mineral end-members subject to
    volumes summing to 1 and each volume in [0, 1].

    Rows where the optimiser fails (sol.success == False) are left as NaN
    in the output and logged with a count summary (PRD §7.4).

    Args:
        df (pd.DataFrame): Well log DataFrame containing rhob_col and nphi_col.
        rhob_col (str): Bulk density column name.
        nphi_col (str): Neutron porosity column name.
        minerals (list): Ordered list of mineral names to include. Defaults to
            ['sand', 'shale', 'calcite', 'heavy'].
        endmembers (dict): End-member constants. Defaults to MINERAL_ENDMEMBERS.

    Returns:
        pd.DataFrame: df with new columns 'mm_<mineral>' for each mineral,
            plus 'mm_converged' (bool) flagging successful optimisation rows.
    """
    if minerals is None:
        minerals = ["sand", "shale", "calcite", "heavy"]
    if endmembers is None:
        endmembers = MINERAL_ENDMEMBERS

    df = df.copy()
    result_cols = {m: [] for m in minerals}
    converged = []

    n_failed = 0
    for _, row in df.iterrows():
        rhob = row.get(rhob_col, np.nan)
        nphi = row.get(nphi_col, np.nan)

        if np.isnan(rhob) or np.isnan(nphi):
            for m in minerals:
                result_cols[m].append(np.nan)
            converged.append(False)
            continue

        vols, success = _solve_one_row(rhob, nphi, minerals, endmembers)
        for m, v in zip(minerals, vols):
            result_cols[m].append(v if success else np.nan)
        converged.append(success)
        if not success:
            n_failed += 1

    for m in minerals:
        df[f"mm_{m}"] = result_cols[m]
    df["mm_converged"] = converged

    if n_failed > 0:
        logger.warning(
            "Multi-mineral optimizer failed to converge on %d / %d rows "
            "(%d%%). Those rows have NaN mineral volumes.",
            n_failed,
            len(df),
            round(100 * n_failed / max(len(df), 1)),
        )
    else:
        logger.info("Multi-mineral optimizer converged on all %d rows.", len(df))

    return df
