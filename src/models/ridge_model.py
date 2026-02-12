import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RidgeModel(BaseModel):
    """Ridge Regression model with StandardScaler pipeline."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.best_alpha = None

    def fit(self, X_train, y_train, best_alpha: float = None):
        self.feature_names = (list(X_train.columns)
                              if hasattr(X_train, 'columns') else None)
        self.best_alpha = best_alpha or self.config.get('alpha', 100.0)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=self.best_alpha))
        ])
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        logger.info(f"Ridge model fitted with alpha={self.best_alpha}")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model not fitted")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.named_steps['ridge'].coef_
        })
        importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
        return importance_df.sort_values('abs_coefficient', ascending=False)
