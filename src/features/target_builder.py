import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TargetBuilder:
    """
    Build forward return target variable.
    """
    
    def __init__(self, horizon_days: int = 30):
        self.horizon_days = horizon_days
    
    def build(self, btc: pd.Series) -> pd.Series:
        """
        Calculate N-day forward returns.
        
        Parameters:
        -----------
        btc : pd.Series, Bitcoin price series
        
        Returns:
        --------
        pd.Series with forward returns as target
        """
        logger.info(f"Building {self.horizon_days}-day forward return target...")
        
        target = pd.Series(np.nan, index=btc.index, name='target')
        
        for i in tqdm(range(len(btc)), 
                      desc="Forward returns", 
                      unit="days"):
            current_date = btc.index[i]
            future_date = current_date + pd.Timedelta(days=self.horizon_days)
            
            future_prices = btc[btc.index >= future_date]
            
            if len(future_prices) > 0:
                p0 = btc.iloc[i]
                p1 = future_prices.iloc[0]
                target.iloc[i] = (p1 / p0 - 1) * 100
        
        valid = target.notna().sum()
        logger.info(f"Target built: {valid} valid values out of {len(target)}")
        
        return target