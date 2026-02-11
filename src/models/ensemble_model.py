import pandas as pd
import numpy as np
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models (simple average).
    """
    
    def __init__(self, models: list, config: dict = None):
        super().__init__(config)
        self.models = models  # List of fitted BaseModel instances
    
    def fit(self, X_train, y_train, **kwargs):
        """Fit all models in ensemble."""
        for model in self.models:
            model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"Ensemble fitted with {len(self.models)} models")
        return self
    
    def predict(self, X):
        """Average predictions from all models."""
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = np.array([model.predict(X) for model in self.models])
        ensemble_pred = predictions.mean(axis=0)
        ensemble_std = predictions.std(axis=0)
        
        return ensemble_pred, ensemble_std