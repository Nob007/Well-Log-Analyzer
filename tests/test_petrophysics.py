"""
test_petrophysics.py — Unit tests for src/petrophysics.py

Every test uses 2–5 hand-computable rows so a student can verify the
arithmetic with a calculator. Values are chosen so the expected outputs
are straightforward to derive.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import math
import pytest
import numpy as np
import pandas as pd

# Add src to path so tests work from any working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from petrophysics import (
    compute_igr,
    compute_vsh_larionov_tertiary,
    compute_vsh_larionov_older,
    compute_vsh_steiber,
    compute_shale_endpoints,
    apply_shale_correction,
    compute_density_porosity,
    compute_effective_porosity,
    compute_simandoux_sw,
    compute_permeability_timur,
    compute_cutoff_flags,
    run_petrophysics_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a tiny DataFrame from keyword-array pairs."""
    return pd.DataFrame(kwargs)


def _approx(a, b, tol=1e-6):
    """Element-wise approximate equality for Series/scalars."""
    if hasattr(a, "__iter__") and not isinstance(a, str):
        return all(abs(ai - bi) < tol for ai, bi in zip(a, b))
    return abs(a - b) < tol


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_igr
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIgr:

    def test_basic_formula(self):
        """Igr = (GR - GR_sand) / (GR_shale - GR_sand)"""
        df = _make_df(GR=[30.0, 70.0, 110.0])
        # GR_sand=30, GR_shale=110 → expected [0, 0.5, 1.0]
        igr = compute_igr(df, gr_sand=30.0, gr_shale=110.0)
        assert _approx(igr.tolist(), [0.0, 0.5, 1.0])

    def test_clips_to_zero_one(self):
        """Values below sand or above shale must be clipped to [0, 1]."""
        df = _make_df(GR=[10.0, 200.0])  # both outside [30, 110]
        igr = compute_igr(df, gr_sand=30.0, gr_shale=110.0)
        assert (igr >= 0.0).all() and (igr <= 1.0).all()

    def test_percentile_method(self):
        """With no explicit baselines, 5th/95th percentile should be used."""
        gr = list(range(10, 110))   # 100 values, 5th pct ≈ 14.95, 95th ≈ 104.05
        df = _make_df(GR=gr)
        igr = compute_igr(df)
        # Result should still be in [0, 1]
        assert (igr >= 0.0).all() and (igr <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vsh models — hand-computed spot checks at Igr = 0.5
# ─────────────────────────────────────────────────────────────────────────────

class TestVshModels:

    IGR05 = pd.Series([0.5])

    def test_larionov_tertiary_at_half(self):
        """Vsh_LT(0.5) = 0.083 * (2^(3.7*0.5) - 1) ≈ 0.2362"""
        expected = 0.083 * (2 ** (3.7 * 0.5) - 1)
        result = compute_vsh_larionov_tertiary(self.IGR05).iloc[0]
        assert abs(result - expected) < 1e-6

    def test_larionov_older_at_half(self):
        """Vsh_LO(0.5) = 0.33 * (2^(2*0.5) - 1) ≈ 0.33"""
        expected = 0.33 * (2 ** (2 * 0.5) - 1)
        result = compute_vsh_larionov_older(self.IGR05).iloc[0]
        assert abs(result - expected) < 1e-6

    def test_steiber_at_half(self):
        """Vsh_S(0.5) = 0.5 / (3 - 2*0.5) = 0.5 / 2.0 = 0.25"""
        expected = 0.5 / (3.0 - 2.0 * 0.5)
        result = compute_vsh_steiber(self.IGR05).iloc[0]
        assert abs(result - expected) < 1e-6

    def test_ranking_at_half(self):
        """PRD §9 C.11: Larionov-T ≤ Steiber ≤ Larionov-O for same Igr."""
        igr = pd.Series([0.3, 0.5, 0.7])
        lt = compute_vsh_larionov_tertiary(igr)
        lo = compute_vsh_larionov_older(igr)
        st = compute_vsh_steiber(igr)
        assert (lt <= st + 1e-9).all(), "Larionov-T should be ≤ Steiber"
        assert (st <= lo + 1e-9).all(), "Steiber should be ≤ Larionov-O"

    def test_all_clipped_zero_one(self):
        igr = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        for fn in [compute_vsh_larionov_tertiary, compute_vsh_larionov_older, compute_vsh_steiber]:
            result = fn(igr)
            assert (result >= 0.0).all() and (result <= 1.0).all(), \
                f"{fn.__name__} output outside [0, 1]"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Density porosity
# ─────────────────────────────────────────────────────────────────────────────

class TestDensityPorosity:

    def test_hand_computed(self):
        """PHID = (rho_ma - RHOB_ns) / (rho_ma - rho_f)
           With rho_ma=2.65, rho_f=1.0, RHOB_ns=2.15 → PHID = (2.65-2.15)/(2.65-1.0)
                                                                = 0.5/1.65 ≈ 0.3030"""
        df = _make_df(RHOB_ns=[2.15])
        phi = compute_density_porosity(df, rhob_ns_col="RHOB_ns", rho_matrix=2.65, rho_fluid=1.0)
        expected = (2.65 - 2.15) / (2.65 - 1.0)
        assert abs(phi.iloc[0] - expected) < 1e-6

    def test_clips_to_zero_047(self):
        """Very high or low density → PHID clamped to [0, 0.47]."""
        df = _make_df(RHOB_ns=[1.0, 3.5])
        phi = compute_density_porosity(df, rhob_ns_col="RHOB_ns")
        assert (phi >= 0.0).all() and (phi <= 0.47).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Effective porosity
# ─────────────────────────────────────────────────────────────────────────────

class TestEffectivePorosity:

    def test_is_average_not_difference(self):
        """phi_eff = (NPHI_ns + PHID) / 2  — NOT minus (bug fix from source notebook)."""
        df = _make_df(NPHI_ns=[0.20], PHID=[0.15])
        phi = compute_effective_porosity(df, nphi_ns_col="NPHI_ns", phid_col="PHID")
        # Correct: (0.20 + 0.15) / 2 = 0.175
        assert abs(phi.iloc[0] - 0.175) < 1e-9, \
            "Effective porosity should be the AVERAGE (sum/2), not difference."

    def test_clips_to_zero_047(self):
        df = _make_df(NPHI_ns=[0.5, -0.1], PHID=[0.5, -0.1])
        phi = compute_effective_porosity(df)
        assert (phi >= 0.0).all() and (phi <= 0.47).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simandoux Sw
# ─────────────────────────────────────────────────────────────────────────────

class TestSimandouxSw:

    def test_range(self):
        """Sw must always be in [0, 1] regardless of input extremes."""
        df = _make_df(
            porosity=[0.05, 0.20, 0.35],
            Igr=[0.1, 0.5, 0.9],
            RDEEP=[1000.0, 10.0, 0.5],
        )
        sw = compute_simandoux_sw(df, Rw=0.135, Rsh=4.0, C=0.4)
        assert (sw >= 0.0).all() and (sw <= 1.0).all()

    def test_high_resistivity_gives_low_sw(self):
        """Very high RDEEP → hydrocarbon-bearing → Sw close to 0."""
        df = _make_df(porosity=[0.20], Igr=[0.2], RDEEP=[2000.0])
        sw = compute_simandoux_sw(df, Rw=0.135, Rsh=4.0, C=0.4)
        assert sw.iloc[0] < 0.2, "High Rt should give low Sw"

    def test_zero_porosity_no_crash(self):
        """Zero porosity rows should not raise ZeroDivisionError."""
        df = _make_df(porosity=[0.0], Igr=[0.5], RDEEP=[10.0])
        sw = compute_simandoux_sw(df)
        # Just check it runs without crashing; NaN or clipped is acceptable
        assert len(sw) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cutoff nesting (PRD §9 C.13)
# ─────────────────────────────────────────────────────────────────────────────

class TestCutoffNesting:

    def _petro_df(self):
        """Small DataFrame that exercises all three zones."""
        return _make_df(
            Igr=[0.2, 0.3, 0.6, 0.8, 0.4],      # some > 0.5 (shale)
            porosity=[0.20, 0.18, 0.22, 0.05, 0.10],
            Sw=[0.30, 0.65, 0.80, 0.50, 0.40],
        )

    def test_pay_subset_of_reservoir_subset_of_sand(self):
        df = compute_cutoff_flags(self._petro_df(),
                                  igr_cut=0.5, phi_cut=0.15, sw_cut=0.70)
        # net_pay rows must also be net_reservoir rows
        assert (df.loc[df["net_pay"], "net_reservoir"]).all(), \
            "net_pay ⊄ net_reservoir"
        # net_reservoir rows must also be net_sand rows
        assert (df.loc[df["net_reservoir"], "net_sand"]).all(), \
            "net_reservoir ⊄ net_sand"

    def test_counts_are_descending(self):
        df = compute_cutoff_flags(self._petro_df(),
                                  igr_cut=0.5, phi_cut=0.15, sw_cut=0.70)
        assert df["net_pay"].sum() <= df["net_reservoir"].sum() <= df["net_sand"].sum()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline — no KeyError, all columns present (PRD §9 A.2)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:

    def _fresh_df(self, n=10):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "DEPT":  np.linspace(2000, 2050, n),
            "GR":    rng.uniform(20, 120, n),
            "NPHI":  rng.uniform(0.05, 0.40, n),
            "RHOB":  rng.uniform(2.0, 2.7, n),
            "RDEEP": rng.uniform(1.0, 200.0, n),
        })

    def test_no_keyerror(self):
        """Pipeline must complete without KeyError on a fresh DataFrame."""
        df = self._fresh_df()
        result = run_petrophysics_pipeline(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_derived_columns_present(self):
        df = self._fresh_df()
        result = run_petrophysics_pipeline(df)
        required = [
            "Igr", "Vsh_larionov_t", "Vsh_larionov_o", "Vsh_steiber", "Vsh",
            "NPHI_ns", "RHOB_ns", "PHID", "porosity", "Sw", "permeability",
            "net_sand", "net_reservoir", "net_pay",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_fractions_in_range(self):
        df = self._fresh_df(n=50)
        result = run_petrophysics_pipeline(df)
        for col in ["Vsh", "Vsh_larionov_t", "Vsh_larionov_o", "Vsh_steiber",
                    "porosity", "Sw"]:
            vals = result[col].dropna()
            assert vals.between(0.0, 1.0).all(), \
                f"Column '{col}' has values outside [0, 1]"

    def test_permeability_non_negative(self):
        df = self._fresh_df(n=50)
        result = run_petrophysics_pipeline(df)
        assert (result["permeability"].dropna() >= 0.0).all()

    def test_vsh_ranking_holds(self):
        """
        PRD §9 C.11: Larionov-T should read lower than Larionov-O for the same Igr.

        Mathematical note: The statement 'LT ≤ Steiber ≤ LO' only holds for
        Igr < ~0.88 (verified analytically). At very high Igr (>0.88), the
        exponential in LT overtakes Steiber. The key auditable check is
        LT ≤ LO, which holds for all practical Igr < ~0.99.

        This test checks the main invariant (LT ≤ LO) on the Igr values
        that are within the range where theory holds.
        """
        df = self._fresh_df(n=100)
        result = run_petrophysics_pipeline(df)
        lt = result["Vsh_larionov_t"]
        lo = result["Vsh_larionov_o"]
        igr = result["Igr"]

        # Check LT ≤ LO holds where Igr < 0.99 (beyond that both saturate to 1)
        mask = igr < 0.99
        assert (lt[mask] <= lo[mask] + 1e-9).all(), \
            "Larionov-T > Larionov-O for Igr < 0.99 — upstream Igr values are suspect."
