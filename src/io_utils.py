"""
io_utils.py — Data ingestion for LOGVIP well log tool.

Handles Excel (.xlsx) and CSV files with optional units-row detection.
Replaces LAS null sentinels (-999.25) with NaN at load time.

Approved libraries: numpy, pandas only.
"""

import pandas as pd
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)

# Default columns expected in a well log file
DEFAULT_REQUIRED_COLS = ["DEPT", "GR", "NPHI", "RDEEP", "RHOB"]

# LAS null sentinel value
LAS_NULL = -999.25


def _is_units_row(row: pd.Series) -> bool:
    """
    Return True if the row looks like a units row (can't be cast to float).

    Args:
        row (pd.Series): The first data row of the DataFrame.

    Returns:
        bool: True if the row is a units row, False otherwise.
    """
    for val in row:
        try:
            float(val)
        except (ValueError, TypeError):
            return True
    return False


def load_log_file(
    path_or_buffer,
    required_cols: list = None,
    null_sentinel: float = LAS_NULL,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a well log file from an Excel (.xlsx) or CSV (.csv) source.

    Steps performed:
      1. Read the file (auto-detect format from extension or buffer type).
      2. If row 0 fails a float() cast on any value, drop it as a units row.
      3. Cast all columns to float64; report any that fail.
      4. Replace null_sentinel values with NaN.
      5. Check that required_cols are present.

    Args:
        path_or_buffer: File path (str) or file-like object (BytesIO from Streamlit).
        required_cols (list): Column names that must be present after loading.
            Defaults to DEFAULT_REQUIRED_COLS.
        null_sentinel (float): Value used for missing data in LAS exports (default -999.25).

    Returns:
        tuple:
            - df (pd.DataFrame): Cleaned DataFrame with float64 columns.
            - report (dict): Summary with keys 'units_row_dropped', 'cols_dropped',
              'rows_before', 'rows_after', 'nulls_replaced'.
    """
    if required_cols is None:
        required_cols = DEFAULT_REQUIRED_COLS

    report = {
        "units_row_dropped": False,
        "cols_dropped": [],
        "rows_before": 0,
        "rows_after": 0,
        "nulls_replaced": 0,
    }

    # ── 1. Read file ────────────────────────────────────────────────────────
    if isinstance(path_or_buffer, (str,)):
        ext = str(path_or_buffer).lower().split(".")[-1]
    elif hasattr(path_or_buffer, "name"):
        ext = path_or_buffer.name.lower().split(".")[-1]
    else:
        # Assume CSV for raw BytesIO without a name
        ext = "csv"

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(path_or_buffer, header=0, dtype=str)
    else:
        df = pd.read_csv(path_or_buffer, header=0, dtype=str)

    report["rows_before"] = len(df)

    # ── 2. Drop units row if present ────────────────────────────────────────
    if len(df) > 0 and _is_units_row(df.iloc[0]):
        df = df.iloc[1:].reset_index(drop=True)
        report["units_row_dropped"] = True
        logger.info("Units row detected and dropped.")

    # ── 3. Cast to float64 ──────────────────────────────────────────────────
    cols_to_drop = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        except Exception:
            cols_to_drop.append(col)
            logger.warning("Column '%s' could not be cast to float64 — dropping.", col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        report["cols_dropped"] = cols_to_drop

    # ── 4. Replace null sentinel ─────────────────────────────────────────────
    null_mask = df == null_sentinel
    report["nulls_replaced"] = int(null_mask.sum().sum())
    df = df.replace(null_sentinel, np.nan)

    report["rows_after"] = len(df)

    # ── 5. Check required columns ────────────────────────────────────────────
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Required columns missing from file: %s. "
            "Calculations that depend on them will fail.",
            missing,
        )

    logger.info(
        "Loaded %d rows, %d columns. Units row dropped: %s. "
        "Sentinel nulls replaced: %d.",
        report["rows_after"],
        len(df.columns),
        report["units_row_dropped"],
        report["nulls_replaced"],
    )

    return df, report
