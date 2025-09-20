# src/utils/data_utils.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from the given path."""
    return pd.read_csv(path)

def save_sample(df: pd.DataFrame, path: str, n: int = 500):
    """Save a random sample of the dataset to a new CSV."""
    df.sample(n, random_state=42).to_csv(path, index=False)
