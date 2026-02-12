import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OnchainFeatureBuilder:
    """
    Build on-chain features (MVRV, volume).
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def build_mvrv_features(self, df_mvrv: pd.DataFrame) -> pd.DataFrame:
        """Build MVRV-based features."""
        if df_mvrv is None or df_mvrv.empty:
            logger.warning("MVRV data not available, skipping")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df_mvrv.index)
        
        with tqdm(total=4, desc="MVRV features", unit="feature") as pbar:
            features['mvrv'] = df_mvrv['mvrv']
            pbar.update(1)
            features['mvrv_change'] = df_mvrv['mvrv'].diff()
            pbar.update(1)
            features['mvrv_ma_30d'] = df_mvrv['mvrv'].rolling(30).mean()
            pbar.update(1)
            features['mvrv_vs_ma30'] = (df_mvrv['mvrv'] / features['mvrv_ma_30d'] - 1) * 100
            pbar.update(1)
        
        logger.info(f"Built {len(features.columns)} MVRV features")
        return features
    
    def build_volume_features(self, df_volume: pd.DataFrame) -> pd.DataFrame:
        """Build volume-based features.
        Note: raw volume values are in the hundreds of billions so we
        log-transform before computing any features.
        """
        if df_volume is None or df_volume.empty:
            logger.warning("Volume data not available, skipping")
            return pd.DataFrame()
        
        total_volume = df_volume.sum(axis=1)
        
        # Log-transform to bring from billions scale to ~20-30 range
        # log1p handles zeros safely: log(1 + 0) = 0
        log_volume = np.log1p(total_volume)
        
        features = pd.DataFrame(index=df_volume.index)
        
        with tqdm(total=5, desc="Volume features", unit="feature") as pbar:
            features['log_volume'] = log_volume
            pbar.update(1)
            features['log_volume_ma_7d'] = log_volume.rolling(7).mean()
            pbar.update(1)
            features['log_volume_ma_30d'] = log_volume.rolling(30).mean()
            pbar.update(1)
            features['volume_vs_ma30'] = (log_volume / features['log_volume_ma_30d'] - 1) * 100
            pbar.update(1)
            features['volume_change_7d'] = log_volume.diff(7)  # log diff ≈ % change
            pbar.update(1)
        
        logger.info(f"Built {len(features.columns)} volume features")
        return features
    
    def build(self, df_mvrv: pd.DataFrame = None,
              df_volume: pd.DataFrame = None) -> pd.DataFrame:
        """Build all on-chain features."""
        mvrv_features = self.build_mvrv_features(df_mvrv)
        volume_features = self.build_volume_features(df_volume)
        
        dfs = [df for df in [mvrv_features, volume_features] if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, axis=1)