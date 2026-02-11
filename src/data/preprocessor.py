import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Clean and preprocess raw data.
    """
    
    def preprocess_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data."""
        logger.info("Preprocessing price data...")
        
        df = df.copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.replace(0, np.nan)
        df = df.dropna(subset=['bitcoin'])
        
        logger.info(f"Price data cleaned: {len(df)} rows remaining")
        return df
    
    def preprocess_macro(self, df: pd.DataFrame, 
                        daily_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Clean macro data (VIX, HY spreads).
        Forward fill weekends and holidays.
        """
        if df is None:
            return None
        
        logger.info("Preprocessing macro data...")
        
        df = df.copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Forward fill weekends and missing days
        df = df.reindex(daily_index, method='ffill')
        
        logger.info(f"Macro data preprocessed: {len(df)} rows")
        return df
    
    def preprocess_mvrv(self, df: pd.DataFrame,
                       daily_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Clean MVRV data."""
        if df is None:
            return None
        
        logger.info("Preprocessing MVRV data...")
        
        df = df.copy()
        
        # Handle date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(daily_index, method='ffill')
        
        logger.info(f"MVRV data preprocessed: {len(df)} rows")
        return df
    
    def preprocess_volume(self, df: pd.DataFrame,
                         daily_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Clean volume data."""
        if df is None:
            return None
        
        logger.info("Preprocessing volume data...")
        
        df = df.copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(daily_index, method='ffill')
        df = df.fillna(0)
        
        logger.info(f"Volume data preprocessed: {len(df)} rows")
        return df
    
    def preprocess_all(self, data: dict) -> dict:
        """Preprocess all data sources."""
        logger.info("Preprocessing all data sources...")
        
        prices = self.preprocess_prices(data['prices'])
        daily_index = prices.index
        
        return {
            'prices': prices,
            'vix': self.preprocess_macro(data.get('vix'), daily_index),
            'hy_spreads': self.preprocess_macro(data.get('hy_spreads'), daily_index),
            'mvrv': self.preprocess_mvrv(data.get('mvrv'), daily_index),
            'volume': self.preprocess_volume(data.get('volume'), daily_index),
        }