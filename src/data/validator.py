import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validate data quality before processing.
    """
    
    def validate_prices(self, df: pd.DataFrame) -> bool:
        logger.info("Validating price data...")
        errors = []
        
        if 'bitcoin' not in df.columns:
            errors.append("Missing 'bitcoin' column in prices")
        else:
            # Only check NaN if column exists
            if df['bitcoin'].isna().sum() > len(df) * 0.1:
                errors.append("More than 10% missing values in bitcoin prices")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Price index is not DatetimeIndex")
        
        if len(df) < 365:
            errors.append(f"Too few price rows: {len(df)} (need at least 365)")
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        logger.info("Price data validation passed")
        return True
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate final dataset before training."""
        logger.info("Validating final dataset...")
        errors = []
        
        if 'target' not in df.columns:
            errors.append("Missing 'target' column")
        
        if len(df) < 100:
            errors.append(f"Dataset too small: {len(df)} rows")
        
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            errors.append(f"NaN values found in columns: {nan_cols}")
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        logger.info(f"Dataset validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    def validate_all(self, data: dict) -> bool:
        """Validate all data sources."""
        return self.validate_prices(data['prices'])