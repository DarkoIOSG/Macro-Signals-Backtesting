import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RidgeModel(BaseModel):
    """Ridge Regression model with StandardScaler."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.best_alpha = None
    
    def fit(self, X_train, y_train, best_alpha: float = None):
        """
        Train Ridge model.
        
        Parameters:
        -----------
        X_train : array-like, training features
        y_train : array-like, training target
        best_alpha : float, alpha to use (from tuning)
        """
        self.feature_names = (list(X_train.columns) 
                             if hasattr(X_train, 'columns') else None)
        self.best_alpha = best_alpha or self.config.get('alpha', 100.0)
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = Ridge(alpha=self.best_alpha)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        
        logger.info(f"Ridge model fitted with alpha={self.best_alpha}")
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature coefficients as importance."""
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model not fitted")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
        return importance_df.sort_values('abs_coefficient', ascending=False)