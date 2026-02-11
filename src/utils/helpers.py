import pandas as pd
import numpy as np
from pathlib import Path


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def remove_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone from DatetimeIndex if present."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet or CSV based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.parquet':
        df.to_parquet(path)
    elif path.suffix == '.csv':
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from parquet or CSV based on extension."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")