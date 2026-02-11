import pandas as pd
from tqdm.auto import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MacroFeatureBuilder:
    """
    Build macro-economic features (VIX, HY spreads).
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def build_vix_features(self, df_vix: pd.DataFrame) -> pd.DataFrame:
        """Build VIX-based features."""
        if df_vix is None or df_vix.empty:
            logger.warning("VIX data not available, skipping")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df_vix.index)
        
        with tqdm(total=6, desc="VIX features", unit="feature") as pbar:
            features['vix'] = df_vix['VIX']
            pbar.update(1)
            features['vix_change'] = df_vix['VIX'].diff()
            pbar.update(1)
            features['vix_change_7d'] = df_vix['VIX'].pct_change(7) * 100
            pbar.update(1)
            features['vix_ma_30d'] = df_vix['VIX'].rolling(30).mean()
            pbar.update(1)
            features['vix_vs_ma30'] = (df_vix['VIX'] / features['vix_ma_30d'] - 1) * 100
            pbar.update(1)
            features['vix_acceleration'] = df_vix['VIX'].diff().diff()
            pbar.update(1)
        
        logger.info(f"Built {len(features.columns)} VIX features")
        return features
    
    def build_hy_features(self, df_hy: pd.DataFrame) -> pd.DataFrame:
        """Build High-Yield spread features."""
        if df_hy is None or df_hy.empty:
            logger.warning("HY spread data not available, skipping")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df_hy.index)
        
        with tqdm(total=6, desc="HY features", unit="feature") as pbar:
            features['hy_spread'] = df_hy['HY_Spread']
            pbar.update(1)
            features['hy_change'] = df_hy['HY_Spread'].diff()
            pbar.update(1)
            features['hy_change_7d'] = df_hy['HY_Spread'].pct_change(7) * 100
            pbar.update(1)
            features['hy_ma_30d'] = df_hy['HY_Spread'].rolling(30).mean()
            pbar.update(1)
            features['hy_vs_ma30'] = (df_hy['HY_Spread'] / features['hy_ma_30d'] - 1) * 100
            pbar.update(1)
            features['hy_acceleration'] = df_hy['HY_Spread'].diff().diff()
            pbar.update(1)
        
        logger.info(f"Built {len(features.columns)} HY spread features")
        return features
    
    def build(self, df_vix: pd.DataFrame = None, 
              df_hy: pd.DataFrame = None) -> pd.DataFrame:
        """Build all macro features."""
        vix_features = self.build_vix_features(df_vix)
        hy_features = self.build_hy_features(df_hy)
        
        # Combine
        dfs = [df for df in [vix_features, hy_features] if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, axis=1)