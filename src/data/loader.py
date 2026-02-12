import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_data_config
from src.utils.helpers import ensure_datetime_index, remove_timezone

logger = get_logger(__name__)


class DataLoader:
    """
    Load raw data files into DataFrames.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_data_config()
    
    def load_prices(self) -> pd.DataFrame:
        """Load crypto price data."""
        path = self.config['data_paths']['raw']['prices']
        logger.info(f"Loading prices from {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        df = remove_timezone(df)
        
        logger.info(f"Loaded prices: {len(df)} rows, {df.columns.tolist()}")
        return df
    
    def load_vix(self) -> pd.DataFrame:
        """Load VIX data."""
        path = self.config['data_paths']['raw']['vix']
        logger.info(f"Loading VIX from {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        df = remove_timezone(df)
        
        logger.info(f"Loaded VIX: {len(df)} rows")
        return df
    
    def load_hy_spreads(self) -> pd.DataFrame:
        """Load High-Yield spread data."""
        path = self.config['data_paths']['raw']['hy_spreads']
        logger.info(f"Loading HY spreads from {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        df = remove_timezone(df)
        
        logger.info(f"Loaded HY spreads: {len(df)} rows")
        return df
    
    def load_mvrv(self) -> pd.DataFrame:
        """Load MVRV data."""
        path = self.config['data_paths']['raw']['mvrv']
        logger.info(f"Loading MVRV from {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True, 
                 date_format='ISO8601')
        df = ensure_datetime_index(df)
        df = remove_timezone(df)
        
        logger.info(f"Loaded MVRV: {len(df)} rows")
        return df
    
    def load_volume(self) -> pd.DataFrame:
        """Load volume data."""
        path = self.config['data_paths']['raw']['volume']
        logger.info(f"Loading volume from {path}")
        
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = ensure_datetime_index(df)
        df = remove_timezone(df)
        
        logger.info(f"Loaded volume: {len(df)} rows")
        return df
    
    def load_all(self) -> dict:
        """Load all data sources."""
        logger.info("Loading all data sources...")
        
        data = {
            'prices': self.load_prices(),
            'vix': self._safe_load(self.load_vix, 'VIX'),
            'hy_spreads': self._safe_load(self.load_hy_spreads, 'HY spreads'),
            'mvrv': self._safe_load(self.load_mvrv, 'MVRV'),
            'volume': self._safe_load(self.load_volume, 'Volume'),
        }
        
        logger.info("All data sources loaded")
        return data
    
    def _safe_load(self, load_func, name: str):
        """Load data with error handling (returns None if file missing)."""
        try:
            return load_func()
        except FileNotFoundError:
            logger.warning(f"{name} data not found, skipping")
            return None