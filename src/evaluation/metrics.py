import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Calculate and compare model performance metrics."""
    
    def calculate_metrics(self, y_true, y_pred, label: str = "") -> dict:
        """Calculate regression metrics."""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
        }
        
        if label:
            logger.info(f"{label} - R²: {metrics['r2']:.4f}, "
                       f"RMSE: {metrics['rmse']:.4f}%")
        
        return metrics
    
    def evaluate_model(self, model, X_train, y_train, 
                      X_test, y_test) -> dict:
        """Full train/test evaluation."""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_metrics = self.calculate_metrics(y_train, y_train_pred, "Train")
        test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'overfitting_gap': train_metrics['r2'] - test_metrics['r2'],
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    def compare_models(self, evaluation_results: dict) -> pd.DataFrame:
        """Create comparison table of all models."""
        rows = []
        for model_name, results in evaluation_results.items():
            rows.append({
                'model': model_name,
                'train_r2': results['train']['r2'],
                'test_r2': results['test']['r2'],
                'train_rmse': results['train']['rmse'],
                'test_rmse': results['test']['rmse'],
                'overfitting_gap': results['overfitting_gap']
            })
        
        comparison_df = pd.DataFrame(rows).set_index('model')
        comparison_df = comparison_df.sort_values('test_r2', ascending=False)
        
        return comparison_df