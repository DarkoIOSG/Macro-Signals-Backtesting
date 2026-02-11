from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models.
    All models must implement fit() and predict().
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Generate predictions."""
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance (override in subclasses)."""
        raise NotImplementedError("This model doesn't support feature importance")