"""
facies.py — KMeans facies clustering for LOGVIP well log tool.

Provides:
  - cluster_facies(): StandardScaler + KMeans on user-selected log features.
  - elbow_data(): inertia vs. k for elbow plot.
  - silhouette_data(): silhouette score vs. k.

k (number of clusters) is a parameter, not hardcoded (PRD §7.5).
Default features: [GR, RHOB_ns, NPHI_ns, RDEEP].

Approved libraries: numpy, pandas, scikit-learn only.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Default log features for facies classification (PRD §7.5)
DEFAULT_FEATURE_COLS = ["GR", "RHOB_ns", "NPHI_ns", "RDEEP"]


def cluster_facies(
    df: pd.DataFrame,
    feature_cols: list = None,
    k: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Classify well log depth samples into k facies using KMeans clustering.

    Steps:
      1. Drop rows with NaN in any feature column.
      2. StandardScaler-normalise the features.
      3. Fit KMeans with k clusters.
      4. Write facies labels (0 … k-1) back to the original DataFrame.

    Rows that had NaN in any feature are assigned label -1 (unclassified).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        feature_cols (list): Feature columns to cluster on. Defaults to
            DEFAULT_FEATURE_COLS (['GR', 'RHOB_ns', 'NPHI_ns', 'RDEEP']).
        k (int): Number of clusters (default 4). Expose in the Streamlit
            sidebar so the student can adjust and justify the choice.
        random_state (int): KMeans random seed for reproducibility.

    Returns:
        pd.DataFrame: df with a new 'facies' column (int, -1 = unclassified).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Facies feature columns not found in DataFrame: %s. "
            "Using available columns: %s.",
            missing, available,
        )
    if not available:
        raise ValueError("No valid feature columns available for facies clustering.")

    df = df.copy()
    df["facies"] = -1  # default: unclassified

    sub = df[available].dropna()
    if len(sub) < k:
        logger.warning(
            "Fewer valid rows (%d) than requested clusters (%d). "
            "Returning all facies as -1.",
            len(sub), k,
        )
        return df

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)

    df.loc[sub.index, "facies"] = labels
    logger.info(
        "KMeans facies clustering complete: k=%d, %d rows classified, %d unclassified.",
        k, len(sub), len(df) - len(sub),
    )
    return df


def elbow_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute KMeans inertia for a range of k values (elbow method).

    Args:
        df (pd.DataFrame): Well log DataFrame.
        feature_cols (list): Feature columns to use.
        k_range (range): Range of k values to evaluate.
        random_state (int): KMeans random seed.

    Returns:
        pd.DataFrame: Table with columns ['k', 'inertia'].
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    sub = df[available].dropna()

    if len(sub) < max(k_range):
        logger.warning("Not enough rows for full k_range; limiting to available data.")

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)

    rows = []
    for k in k_range:
        if k > len(sub):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        rows.append({"k": k, "inertia": km.inertia_})

    return pd.DataFrame(rows)


def silhouette_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute silhouette scores for a range of k values.

    Args:
        df (pd.DataFrame): Well log DataFrame.
        feature_cols (list): Feature columns to use.
        k_range (range): Range of k values to evaluate.
        random_state (int): KMeans random seed.

    Returns:
        pd.DataFrame: Table with columns ['k', 'silhouette_score'].
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    sub = df[available].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)

    rows = []
    for k in k_range:
        if k >= len(sub):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette_score": round(score, 4)})

    return pd.DataFrame(rows)
