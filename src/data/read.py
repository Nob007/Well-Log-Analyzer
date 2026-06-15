import pandas as pd
import lasio
import os
import numpy as np

def read_log(file_path, file_name)->pd.DataFrame:
    """
    Reads a log file in CSV, LAS, or Excel format and returns a pandas DataFrame.
    Args: 
        file_path (str): The path to the directory containing the log file.
        file_name (str): The name of the log file to be read.
    Returns:
        pd.DataFrame: A DataFrame containing the log data, with invalid values replaced by NaN.
    """
    full_path = os.path.join(file_path, file_name)
    _, ext = os.path.splitext(file_name.lower())
    try:
        if ext == ".csv":
            df = pd.read_csv(full_path)
        elif ext == ".las":
            lf = lasio.read(full_path)
            df = lf.df().reset_index()
        elif ext in ['.xls','.xlsx']:
            df = pd.read_excel(full_path)
        print("Printing description of dataset:\n" + str(df.describe()))
        print("Printing available logs:\n" + str(df.columns))
        df = df.replace([-999, -999.25, -999.5, -999.75], np.nan)
        return df
    except Exception as e:
        print(f"Unsupported File Format!\nError: {e}")
        return None
