
def permeability_timur(df, porosity_col = 'porosity', sw_col = 'Sw', swirr = None, c=0.136, d=4.4, e=2.0):
    """
    Calculates permeability using the Timur correlation.

    The Timur equation is an empirical formula used to estimate permeability (k)
    from effective porosity (phi_e) and irreducible water saturation (sw_i).
    The general form is: k = C * (phi_e^D) / (sw_i^E)

    Args:
        df (pd.DataFrame): The DataFrame containing the necessary columns.
        porosity_col (str): The name of the effective porosity column in the DataFrame.
        sw_col (str): The name of the irreducible water saturation column in the DataFrame.
        swirr (float): The irreducible water saturation value (default is None).
        c (float): Permeability constant (default is 0.136 for k in mD).
        d (float): Porosity exponent (default is 4.4).
        e (float): Irreducible saturation exponent (default is 2.0).

    Returns:
        pd.Series or float: The calculated permeability in millidarcies (mD).
    """
    phi_e = df[porosity_col]
    sw_i = swirr if swirr is not None else df[sw_col][df[sw_col] > 0].min()
    permeability = c * (phi_e ** d) / (sw_i ** e)
    df['permeability'] = permeability
    return permeability

def permeability_coates(df, porosity_col = 'porosity', sw_col = 'Sw',swirr = None, a=1000., b=4.0, c=2.0):
    """
    Calculates permeability using the Coates correlation.

    The Coates equation is an empirical formula used to estimate permeability (k)
    from effective porosity (phi_e) and irreducible water saturation (sw_i).
    The general form is: k = A * (phi_e^B) / (sw_i^C)

    Args:
        df (pd.DataFrame): The DataFrame containing the necessary columns.
        porosity_col (str): The name of the effective porosity column in the DataFrame.
        sw_col (str): The name of the irreducible water saturation column in the DataFrame.
        swirr (float): The irreducible water saturation value (default is None).
        a (float): Permeability constant (default is 0.62 for k in mD).
        b (float): Porosity exponent (default is 3.0).
        c (float): Irreducible saturation exponent (default is 2.0).
    Returns:
        pd.Series or float: The calculated permeability in millidarcies (mD).
    """
    phi_e = df[porosity_col]
    sw_i = swirr if swirr is not None else df[sw_col][df[sw_col] > 0].min()
    permeability = a * (phi_e ** b) * ((1-sw_i)/sw_i)**c
    df['permeability'] = permeability
    return permeability
