from src.models.ridge_model import RidgeModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.utils.logger import get_logger
from src.utils.helpers import save_dataframe
import pandas as pd

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrate model training and tuning.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.tuner = HyperparameterTuner(
            cv_splits=self.config.get('cv_splits', 5)
        )
    
    def train_ridge(self, X_train, y_train) -> RidgeModel:
        """Tune and train Ridge model."""
        logger.info("Training Ridge model...")
        
        alphas = self.config.get('ridge', {}).get('alphas', None)
        tuning_results = self.tuner.tune_ridge(X_train, y_train, alphas)
        
        model = RidgeModel(config=self.config.get('ridge', {}))
        model.fit(X_train, y_train, 
                 best_alpha=tuning_results['best_params']['alpha'])
        
        logger.info("Ridge model training complete")
        return model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestModel:
        """Tune and train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        param_grid = self.config.get('random_forest', {}).get('param_grid', None)
        tuning_results = self.tuner.tune_random_forest(X_train, y_train, param_grid)
        
        model = RandomForestModel(config=self.config.get('random_forest', {}))
        model.fit(X_train, y_train, 
                 best_params=tuning_results['best_params'])
        
        logger.info("Random Forest training complete")
        return model
    
    def train_gradient_boosting(self, X_train, y_train) -> GradientBoostingModel:
        """Tune and train Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        
        param_grid = self.config.get('gradient_boosting', {}).get('param_grid', None)
        tuning_results = self.tuner.tune_gradient_boosting(X_train, y_train, param_grid)
        
        model = GradientBoostingModel(config=self.config.get('gradient_boosting', {}))
        model.fit(X_train, y_train, 
                 best_params=tuning_results['best_params'])
        
        logger.info("Gradient Boosting training complete")
        return model
    
    def train_all(self, X_train, y_train) -> dict:
        """Train all models."""
        logger.info("Training all models...")
        
        models = {
            'ridge': self.train_ridge(X_train, y_train),
            'random_forest': self.train_random_forest(X_train, y_train),
            'gradient_boosting': self.train_gradient_boosting(X_train, y_train),
        }
        
        logger.info("All models trained successfully")
        return models