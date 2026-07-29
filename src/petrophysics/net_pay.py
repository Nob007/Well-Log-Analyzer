import numpy as np
import pandas as pd

def net_reservoir_thickness(df,depth_col = 'DEPTH',porosity_col = 'porosity', shale_col = 'Vshale', permeability_col = 'permeability', porosity_cutoff = 0.1, perm_cutoff = 1.0, shale_cutoff = 0.4):
    """
    Calculates the net reservoir thickness based on a porosity cutoff.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the necessary columns.
        porosity_col (str): The name of the porosity column in the DataFrame.
        shale_col (str): The name of the shale column in the DataFrame.
        permeability_col (str): The name of the permeability column in the DataFrame.
        porosity_cutoff (float): The porosity cutoff value to determine net reservoir thickness.
        perm_cutoff (float): The permeability cutoff value to determine net reservoir thickness.
        shale_cutoff (float): The shale cutoff value to determine net reservoir thickness.

    Returns:
        float: The calculated net reservoir thickness.
    """
    try:
        net_reservoir = df[depth_col].loc[
            (df[porosity_col] >= porosity_cutoff) &
            (df[permeability_col] >= perm_cutoff) &
            (df[shale_col] < shale_cutoff)
        ]
        net_thickness = net_reservoir.sum()
        df['net_reservoir'] = net_reservoir
        print(f"Net reservoir thickness calculated: {net_thickness}")
        return net_thickness
    except Exception as e:
        print(f"Error calculating net reservoir thickness: {e}")
        return None

def net_pay_thickness(df, depth_col = 'DEPTH',porosity_col = 'porosity', shale_col = 'Vshale', permeability_col = 'permeability',sw_col = 'Sw', porosity_cutoff = 0.1, perm_cutoff = 1.0, shale_cutoff = 0.4, sw_cutoff = 0.6):
    """
    Calculates the net pay thickness based on a porosity cutoff.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the necessary columns.
        porosity_col (str): The name of the porosity column in the DataFrame.
        shale_col (str): The name of the shale column in the DataFrame.
        permeability_col (str): The name of the permeability column in the DataFrame.
        porosity_cutoff (float): The porosity cutoff value to determine net pay thickness.
        perm_cutoff (float): The permeability cutoff value to determine net pay thickness.
        shale_cutoff (float): The shale cutoff value to determine net pay thickness.
        sw_cutoff (float): The water saturation cutoff value to determine net pay thickness.

    Returns:
        float: The calculated net pay thickness.
    """
    try:
        net_pay = df[depth_col].loc[
            (df[porosity_col] >= porosity_cutoff) &
            (df[permeability_col] >= perm_cutoff) &
            (df[shale_col] < shale_cutoff) &
            (df[sw_col] < sw_cutoff)
        ]
        net_thickness = net_pay.sum()
        df['net_pay'] = net_pay
        print(f"Net pay thickness calculated: {net_thickness}")
        return net_thickness
    except Exception as e:
        print(f"Error calculating net pay thickness: {e}")
        return None