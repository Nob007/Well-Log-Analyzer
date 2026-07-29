import numpy as np
import pandas as pd

def archies_sw(df, porosity_col = 'porosity', Rt_col = 'Rt', a = 1.0, m = 2.0, n = 2.0, Rw = 0.135):
    """
    Calculates water saturation using Archie's equation.

    The Archie's equation is an empirical formula used to estimate water saturation (Sw)
    from porosity (phi), water resistivity (Rw), and formation resistivity (Rt).
    The general form is: Sw = a * (phi^m) / (Rt/Rw)^(1/n)

    Args:
        df (pd.DataFrame): The DataFrame containing the necessary columns.
        porosity_col (str): The name of the porosity column in the DataFrame.
        Rt_col (str): The name of the formation resistivity column in the DataFrame.
        Rw (float): Water resistivity (ohm-m).
        a (float): Tortuosity factor (default is 1.0).
        m (float): Cementation exponent (default is 2.0).
        n (float): Saturation exponent (default is 2.0).

    Returns:
        pd.Series or float: The calculated water saturation.
    """
    Sw_values = a * (df[porosity_col] ** m) / ((df[Rt_col] / Rw) ** (1/n))
    Sw_values = Sw_values.clip(lower=0.0, upper=1.0, inplace=True)
    df['Sw'] = Sw_values
    return Sw_values

def simandoux_sw(df, porosity_col = 'porosity', Rw = 0.135, Rsh = 10, RDEEP = 100, Igr = 0.1, C = 0.4):
    """
    Calculates water saturation using the Simandoux equation.

    The Simandoux equation is an empirical formula used to estimate water saturation (Sw)
    from porosity (phi), water saturation (Sw), and irreducible water saturation (Swc).
    The general form is: Sw = a * (phi^b) / (Swc^b)

    Args:
        phi (pd.Series or float): Porosity (fraction).
        sw (pd.Series or float): Water saturation (fraction).
        swc (pd.Series or float): Irreducible water saturation (fraction).
        a (float): Constant for the Simandoux equation (default is 1.0).
        b (float): Exponent for the Simandoux equation (default is 1.0).

    Returns:
        pd.Series or float: The calculated water saturation.
    """
    Sw_values = (C*Rw/df[porosity_col]**2)*(-df['Igr']/Rsh + np.sqrt((5*df[porosity_col]**2/(Rw*df['RDEEP'])+(df['Igr']/Rsh)**2)))
    Sw_values = Sw_values.clip(lower=0.0, upper=1.0, inplace=True)
    df['Sw'] = Sw_values
    return Sw_values