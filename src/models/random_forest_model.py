import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest Regressor model."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
    
    def fit(self, X_train, y_train, best_params: dict = None):
        """Train Random Forest model."""
        self.feature_names = (list(X_train.columns) 
                             if hasattr(X_train, 'columns') else None)
        
        params = best_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.best_params = params
        
        logger.info(f"Random Forest fitted with {params['n_estimators']} trees")
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return Gini feature importances."""
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model not fitted")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)