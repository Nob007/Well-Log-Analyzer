"""
qc.py — Quality control for LOGVIP well log tool.

Provides curve statistics, physical range flagging, and null reporting.
All functions are non-destructive: they return flags/summaries rather than
silently modifying the DataFrame.

Approved libraries: numpy, pandas only.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Physical plausibility ranges for common well log curves.
# Source: standard petrophysical log quality control guidelines.
PHYSICAL_RANGES = {
    "GR":    (0.0,  300.0),   # gAPI
    "NPHI":  (0.0,  0.6),     # fraction (v/v)
    "RHOB":  (1.0,  3.2),     # g/cc
    "RDEEP": (0.2,  2000.0),  # ohm·m
    "DEPT":  (0.0,  10000.0), # metres (depth)
    "DEPTH": (0.0,  10000.0),
}


def describe_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table with min, max, mean, std, null count, and % nulls
    for every numeric column in df.

    Args:
        df (pd.DataFrame): Well log DataFrame (float64 columns).

    Returns:
        pd.DataFrame: Summary table indexed by curve name with columns
            [min, max, mean, std, null_count, null_pct].
    """
    stats = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        null_count = int(s.isna().sum())
        stats.append(
            {
                "curve": col,
                "min":   round(s.min(), 4),
                "max":   round(s.max(), 4),
                "mean":  round(s.mean(), 4),
                "std":   round(s.std(), 4),
                "null_count": null_count,
                "null_pct":   round(100.0 * null_count / max(len(s), 1), 2),
            }
        )
    return pd.DataFrame(stats).set_index("curve")


def flag_outliers(
    df: pd.DataFrame,
    ranges: dict = None,
) -> pd.DataFrame:
    """
    Return a boolean DataFrame where True marks rows outside physically
    plausible ranges for each curve.

    Rows are flagged, NOT dropped — the caller decides what to do.

    Args:
        df (pd.DataFrame): Well log DataFrame.
        ranges (dict): Optional override of physical ranges. Keys are curve
            names; values are (min, max) tuples. Falls back to PHYSICAL_RANGES
            for any curve not in the override dict.

    Returns:
        pd.DataFrame: Boolean DataFrame with same index as df. A True cell
            means that row × curve combination is outside the expected range.
            Columns only included for curves that have a known range.
    """
    effective_ranges = {**PHYSICAL_RANGES, **(ranges or {})}
    flag_df = pd.DataFrame(index=df.index)

    for col, (lo, hi) in effective_ranges.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        out_of_range = ~df[col].between(lo, hi) & df[col].notna()
        flag_df[col] = out_of_range
        n_flagged = int(out_of_range.sum())
        if n_flagged > 0:
            logger.warning(
                "Curve '%s': %d rows outside physical range [%g, %g].",
                col, n_flagged, lo, hi,
            )

    return flag_df


def outlier_summary(flag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the boolean outlier flag DataFrame as a count table.

    Args:
        flag_df (pd.DataFrame): Output of flag_outliers().

    Returns:
        pd.DataFrame: Table with columns [curve, n_flagged, pct_flagged].
    """
    rows = []
    n_total = len(flag_df)
    for col in flag_df.columns:
        n = int(flag_df[col].sum())
        rows.append(
            {
                "curve": col,
                "n_flagged": n,
                "pct_flagged": round(100.0 * n / max(n_total, 1), 2),
            }
        )
    return pd.DataFrame(rows).set_index("curve")


def apply_null_sentinel(
    df: pd.DataFrame,
    sentinel: float = -999.25,
) -> pd.DataFrame:
    """
    Replace all occurrences of the LAS null sentinel with NaN in-place.

    This should be called once, immediately after loading, before any
    statistics or plots are produced.

    Args:
        df (pd.DataFrame): Well log DataFrame.
        sentinel (float): Sentinel value to replace (default -999.25).

    Returns:
        pd.DataFrame: DataFrame with sentinel values replaced by NaN.
            (Returns the same object; modifies in-place for efficiency.)
    """
    mask = df == sentinel
    n = int(mask.sum().sum())
    if n > 0:
        df = df.replace(sentinel, np.nan)
        logger.info("Replaced %d sentinel values (%g) with NaN.", n, sentinel)
    return df
