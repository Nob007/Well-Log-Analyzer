import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def density_neutron_crossplot(df, rhob_col = 'RHOB', nphi_col = 'NPHI', rhob_col_ns = 'RHOB_ns', nphi_col_ns = 'NPHI_ns'):
    """
    Creates a density-neutron crossplot using the specified columns for bulk density and neutron porosity.
    Args:
        df (pd.DataFrame): The DataFrame containing the bulk density and neutron porosity logs.
        rhob_col (str): The name of the bulk density column in the DataFrame.
        nphi_col (str): The name of the neutron porosity column in the DataFrame.
        rhob_col_ns (str): The name of the non-shale bulk density column in the DataFrame.
        nphi_col_ns (str): The name of the non-shale neutron porosity column in the DataFrame.
    """
    try:
        if rhob_col_ns not in df.columns or nphi_col_ns not in df.columns:
            raise ValueError(f"Columns '{rhob_col_ns}' and/or '{nphi_col_ns}' not found in DataFrame.")
        rhob_ns = df[rhob_col_ns]
        nphi_ns = df[nphi_col_ns]
        if rhob_col not in df.columns or nphi_col not in df.columns:
            print(f"Warning: Columns '{rhob_col}' and/or '{nphi_col}' not found in DataFrame. Using non-shale logs for crossplot.")
        rhob_raw = df[rhob_col]
        nphi_raw = df[nphi_col]
    except Exception as e:
        print(f"Error: {e}")
        return None
    phi = np.linspace(0, 0.45, 200)
    rho_f = 1.0

    lithologies = {
    "Sandstone": {"rho_ma": 2.65, "phiN_ma": -0.02},
    "Limestone": {"rho_ma": 2.71, "phiN_ma": 0.00},
    "Dolomite":  {"rho_ma": 2.87, "phiN_ma": 0.02}
    }
    phi_labels = np.arange(0.0, 0.5, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(14,7), sharey=True)
    for ax, nphi, rhob, title in zip(
    axes,
    [nphi_raw, nphi_ns],
    [rhob_raw, rhob_ns],
    ["Raw Logs", "Shale Corrected Logs"]):
        ax.scatter(nphi, rhob, s=6, alpha=0.6)

        for name, props in lithologies.items():
            rho_line = phi * rho_f + (1 - phi) * props["rho_ma"]
            phiN_line = (1 - phi) * props["phiN_ma"] + phi * 1.0
            ax.plot(phiN_line, rho_line, linewidth=2, label=name)

            for p in phi_labels:
                rho_p = p * rho_f + (1 - p) * props["rho_ma"]
                phiN_p = (1 - p) * props["phiN_ma"] + p * 1.0
                ax.plot(phiN_p, rho_p, 'o', markersize=3)

        ax.set_xlim(-0.05, 0.45)
        ax.set_ylim(3.0, 1.9)

        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.6)
        ax.grid(which='minor', linestyle='-', linewidth=0.3, alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel("NPHI (fraction)")

    axes[0].set_ylabel("RHOB (g/cc)")

    axes[0].legend()

plt.suptitle("Neutron–Density Crossplot: Effect of Shale Correction")
plt.tight_layout()
plt.show()

