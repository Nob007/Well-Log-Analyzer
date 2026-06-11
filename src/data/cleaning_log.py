import pandas as pd
import lasio
import os
import numpy as np

def cleaning_log(df, valid_logs = None)->pd.DataFrame:
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
