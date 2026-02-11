import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    Combine and select features for model training.
    """
    
    def combine_features(self, *feature_dfs) -> pd.DataFrame:
        """
        Combine multiple feature DataFrames on common index.
        
        Parameters:
        -----------
        *feature_dfs : variable number of DataFrames
        
        Returns:
        --------
        Combined DataFrame
        """
        valid_dfs = [df for df in feature_dfs if df is not None and not df.empty]
        
        if not valid_dfs:
            raise ValueError("No valid feature DataFrames provided")
        
        combined = pd.concat(valid_dfs, axis=1)
        logger.info(f"Combined {len(valid_dfs)} feature sets: {combined.shape}")
        
        return combined
    
    def build_dataset(self, features: pd.DataFrame, 
                     target: pd.Series) -> pd.DataFrame:
        """
        Combine features with target and clean.
        
        Parameters:
        -----------
        features : pd.DataFrame with all features
        target : pd.Series with target variable
        
        Returns:
        --------
        Clean DataFrame ready for training
        """
        logger.info("Building final dataset...")
        
        dataset = features.copy()
        dataset['target'] = target
        
        initial_rows = len(dataset)
        dataset = dataset.dropna()
        final_rows = len(dataset)
        
        dropped = initial_rows - final_rows
        logger.info(f"Dataset built: {final_rows} rows ({dropped} dropped for NaN)")
        
        return dataset
    
    def split_features_target(self, dataset: pd.DataFrame):
        """Split dataset into X and y."""
        X = dataset.drop('target', axis=1)
        y = dataset['target']
        return X, y
    
    def temporal_split(self, dataset: pd.DataFrame, 
                      train_ratio: float = 0.7):
        """
        Split dataset temporally (no shuffling).
        
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        X, y = self.split_features_target(dataset)
        
        split_idx = int(len(dataset) * train_ratio)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)} rows "
                   f"({X_train.index[0].date()} to {X_train.index[-1].date()})")
        logger.info(f"Test: {len(X_test)} rows "
                   f"({X_test.index[0].date()} to {X_test.index[-1].date()})")
        
        return X_train, X_test, y_train, y_test