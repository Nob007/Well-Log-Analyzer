import pandas as pd
import lasio
import os
import numpy as np

def cleaning_log(df, valid_logs = None)->pd.DataFrame:
    """
    Cleans the log data by removing rows with missing values and selecting valid logs.
    Args:
        df (pd.DataFrame): The input DataFrame containing the log data.
        valid_logs (list): A list of column names representing the valid logs to be included.
    Returns:
        pd.DataFrame: A cleaned DataFrame with only the valid logs and no missing values.
    """
    try:
        try:
            df = df[valid_logs]
        except Exception as e:
            print(f"Error: {e}\nProceeding with all available logs.")
        df = df.dropna(axis = 0)
        print("Printing description of cleaned dataset:\n" + str(df.describe()))
        return df
    except Exception as e:
        print(f"Error: {e}")
