import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regressor model."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
    
    def fit(self, X_train, y_train, best_params: dict = None):
        """Train Gradient Boosting model."""
        self.feature_names = (list(X_train.columns) 
                             if hasattr(X_train, 'columns') else None)
        
        params = best_params or {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42
        }
        
        self.model = GradientBoostingRegressor(**params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.best_params = params
        
        logger.info(f"Gradient Boosting fitted with {params['n_estimators']} estimators")
        return self
    
    def predict(self, X):
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances."""
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model not fitted")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)