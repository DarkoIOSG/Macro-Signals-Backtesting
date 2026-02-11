import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceFeatureBuilder:
    """
    Build price-based features from BTC price data.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.return_windows = self.config.get('return_windows', [1, 7, 14, 30, 60, 90, 180, 365])
        self.volatility_windows = self.config.get('volatility_windows', [7, 30, 90])
        self.ma_windows = self.config.get('ma_windows', [7, 30, 50, 200])
        self.rsi_window = self.config.get('rsi_window', 14)
    
    def build(self, btc: pd.Series) -> pd.DataFrame:
        """
        Build all price features.
        
        Parameters:
        -----------
        btc : pd.Series, Bitcoin price series
        
        Returns:
        --------
        pd.DataFrame with all price features
        """
        features = pd.DataFrame(index=btc.index)
        
        total = (len(self.return_windows) + 
                 len(self.volatility_windows) + 
                 len(self.ma_windows) * 3 + 2 + 1)
        
        with tqdm(total=total, desc="BTC features", unit="feature") as pbar:
            
            # Returns
            for w in self.return_windows:
                features[f'return_{w}d'] = btc.pct_change(w) * 100
                pbar.update(1)
            
            # Volatility
            returns_daily = btc.pct_change()
            for w in self.volatility_windows:
                features[f'volatility_{w}d'] = (
                    returns_daily.rolling(w).std() * np.sqrt(365) * 100
                )
                pbar.update(1)
            
            # Moving averages
            mas = {}
            for w in self.ma_windows:
                mas[w] = btc.rolling(w).mean()
                features[f'ma_{w}d'] = mas[w]
                pbar.update(1)
            
            # Price vs moving averages
            for w in self.ma_windows:
                features[f'price_vs_ma{w}'] = (btc / mas[w] - 1) * 100
                pbar.update(1)
            
            # MA crossovers
            if 7 in self.ma_windows and 30 in self.ma_windows:
                features['ma7_vs_ma30'] = (mas[7] / mas[30] - 1) * 100
                pbar.update(1)
            if 30 in self.ma_windows and 200 in self.ma_windows:
                features['ma30_vs_ma200'] = (mas[30] / mas[200] - 1) * 100
                pbar.update(1)
            
            # Rate of change
            features['roc_14d'] = (btc / btc.shift(14) - 1) * 100
            features['roc_30d'] = (btc / btc.shift(30) - 1) * 100
            pbar.update(1)
            
            # RSI
            delta = btc.diff()
            gain = (delta.where(delta > 0, 0)).rolling(self.rsi_window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_window).mean()
            rs = gain / loss
            features[f'rsi_{self.rsi_window}d'] = 100 - (100 / (1 + rs))
            pbar.update(1)
        
        logger.info(f"Built {len(features.columns)} price features")
        return features