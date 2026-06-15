import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gr_constant_selector(gr_values, constant_selector = None):
    """
    Function to select shale and sand constants based on the provided method.
    Args:
        gr_values (pd.Series): The gamma ray values.
        constant_selector (str): The method to select constants. Options are:
        - None: Use the maximum and minimum values of the gamma ray log.
        - 'averaged': Use the average of the three highest and three lowest values.
        - 'user_defined': Prompt the user to input shale and sand values.
    Returns:
        tuple: A tuple containing the shale and sand constants (gr_shale, gr_sand).
    """
    if constant_selector is None:
        try:
            gr_shale = np.max(gr_values)
            gr_sand = np.min(gr_values)         
        except Exception as e:
            print(f"Error calculating shale value: {e}")
            gr_shale = np.nan
            gr_sand = np.nan
    elif constant_selector == 'averaged':
        try:
            gr_shale = gr_values.nlargest(3).mean()
            gr_sand = gr_values.nsmallest(3).mean()
        except Exception as e:
            print(f"Error calculating averaged values: {e}")
            gr_shale = np.nan
            gr_sand = np.nan
    elif constant_selector == 'user_defined':
        try:
            gr_shale = float(input("Enter the shale value: "))
            gr_sand = float(input("Enter the sand value: "))
        except Exception as e:
            print(f"Error with user-defined values: {e}")
            gr_shale = np.nan
            gr_sand = np.nan
    else:
        print(f"Invalid constant_selector value: {constant_selector}. Using default method.")
        try:
            gr_shale = np.max(gr_values)
            gr_sand = np.min(gr_values)         
        except Exception as e:
            print(f"Error calculating shale value: {e}")
            gr_shale = np.nan
            gr_sand = np.nan

    return gr_shale, gr_sand

def linear_shale_model(gr_values, gr_shale, gr_sand):
    """
    Calulates the volume of shale using the linear shale model. For younger formations, and quick estimations.
    Args:
        gr_values (pd.Series): The gamma ray values.
        gr_shale (float): The gamma ray value for shale.
        gr_sand (float): The gamma ray value for sand.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    try:
        if gr_shale == gr_sand:
            raise ValueError("Shale and sand values are equal, cannot compute GR.")
        gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
        return gr_normalized
    except Exception as e:
        print(f"Error in linear shale model: {e}")
        return None
    
def larionov_shale_model(gr_values, gr_shale, gr_sand, age_rock):
    """
    Calculates the volume of shale using the Larionov shale model. Pre- tertiary accounts for older rocks. Tertiary accounts for non-linear increase in shale radioactivity. 
    Args:
        gr_values (pd.Series): The gamma ray values.
        gr_shale (float): The gamma ray value for shale.
        gr_sand (float): The gamma ray value for sand.
        age_rock (str): The age of the rock, either 'pre-tertiary' or 'tertiary'.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    if age_rock == 'pre-tertiary':
        gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
        V_shale = 0.33 * (2 ** (2 * gr_normalized) - 1)
    elif age_rock == 'tertiary':
        gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
        V_shale = 0.083 * (2 ** (3.7 * gr_normalized) - 1)
    else:
        raise ValueError("Invalid age_rock value. Must be 'pre-tertiary' or 'tertiary'.")
    return V_shale

def steiber_shale_model(gr_values, gr_shale, gr_sand):
    """
    Calculates the volume of shale using the Steiber shale model. Accounts for dispersed shales and laminated sands.
    Args:
        gr_values (pd.Series): The gamma ray values.
        gr_shale (float): The gamma ray value for shale.
        gr_sand (float): The gamma ray value for sand.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
    V_shale = gr_normalized / (3 - 2*gr_normalized)
    return V_shale

def clavier_shale_model(gr_values, gr_shale, gr_sand):
    """
    Calculates the volume of shale using the Clavier shale model. Accounts for highly radioactive shales and complex lithology.
    Args:
        gr_values (pd.Series): The gamma ray values.
        gr_shale (float): The gamma ray value for shale.
        gr_sand (float): The gamma ray value for sand.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
    V_shale = 1.7 - (3.38 - (gr_normalized + 0.7)**2)**0.5
    return V_shale

def dresser_atlas_shale_model(gr_values, gr_shale, gr_sand):
    """
    Calculates the volume of shale using the Dresser Atlas shale model. General purpose and smooth transition.
    Args:
        gr_values (pd.Series): The gamma ray values.
        gr_shale (float): The gamma ray value for shale.
        gr_sand (float): The gamma ray value for sand.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    gr_normalized = (gr_values - gr_sand) / (gr_shale - gr_sand)
    V_shale = gr_normalized / (0.6 + 0.4*gr_normalized)
    return V_shale

def calculate_shale_volume(df, gr_col = 'GR', depth_col = 'DEPTH', constant_selector = None, shale_model = None):
    """
    Calculates the volume of shale using the specified shale model and constants.
    Args:
        df (pd.DataFrame): The DataFrame containing the gamma ray and depth logs.
        gr_col (str): The name of the gamma ray column in the DataFrame.
        depth_col (str): The name of the depth column in the DataFrame.
        constant_selector (str): The method to select constants. Options are:
            - None: Use the maximum and minimum values of the gamma ray log.
            - 'averaged': Use the average of the three highest and three lowest values.
            - 'user_defined': Prompt the user to input shale and sand values.
        shale_model (str): The shale model to use. Options are:
            - 'linear': Use the linear shale model.
            - 'larionov': Use the Larionov shale model.
            - 'steiber': Use the Steiber shale model.
            - 'clavier': Use the Clavier shale model.
            - 'dresser_atlas': Use the Dresser Atlas shale model.
    Returns:
        pd.Series: The calculated volume of shale.
    """
    try:
        if gr_col not in df.columns or depth_col not in df.columns:
            raise ValueError(f"Columns '{gr_col}' and/or '{depth_col}' not found in DataFrame.")
        gr_values = df[gr_col]
    except Exception as e:
        print(f"Error: {e}")
        return None
    gr_shale, gr_sand = gr_constant_selector(gr_values, constant_selector)
    if shale_model == 'linear':
        return linear_shale_model(gr_values, gr_shale, gr_sand)
    elif shale_model == 'larionov':
        age_rock = input("Enter the age of the rock ('pre-tertiary' or 'tertiary'): ")
        return larionov_shale_model(gr_values, gr_shale, gr_sand, age_rock)
    elif shale_model == 'steiber':
        return steiber_shale_model(gr_values, gr_shale, gr_sand)
    elif shale_model == 'clavier':
        return clavier_shale_model(gr_values, gr_shale, gr_sand)
    elif shale_model == 'dresser_atlas':
        return dresser_atlas_shale_model(gr_values, gr_shale, gr_sand)
    else:
        return linear_shale_model(gr_values, gr_shale, gr_sand)


