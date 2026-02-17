import pandas as pd
import numpy as np

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def numeric_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c == "Name":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def prepare_matrix(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X
