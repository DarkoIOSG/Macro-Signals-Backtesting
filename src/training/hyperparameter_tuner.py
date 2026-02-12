import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Tune hyperparameters using time series cross-validation.
    """
    
    def __init__(self, cv_splits: int = 5):
        self.cv_splits = cv_splits
        self.tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    def tune_ridge(self, X_train, y_train, alphas: list = None) -> dict:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        alphas = alphas or [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 
                            50.0, 100.0, 200.0, 500.0, 1000.0]
        
        logger.info(f"Tuning Ridge with {len(alphas)} alpha values...")
        
        # Wrap in pipeline so scaler is applied inside each CV fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'ridge__alpha': alphas},  # note: ridge__alpha not alpha
            cv=self.tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_alpha = grid_search.best_params_['ridge__alpha']
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"Best Ridge alpha: {best_alpha} (CV RMSE: {best_rmse:.4f}%)")
        
        return {
            'best_params': {'alpha': best_alpha},
            'best_rmse': best_rmse,
            'cv_results': grid_search.cv_results_
        }
    
    def tune_random_forest(self, X_train, y_train, param_grid: dict = None) -> dict:
        """Tune Random Forest hyperparameters."""
        param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        
        total = np.prod([len(v) for v in param_grid.values()])
        logger.info(f"Tuning Random Forest with {total} combinations...")
        
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=self.tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"Best RF params: {best_params} (CV RMSE: {best_rmse:.4f}%)")
        
        return {
            'best_params': best_params,
            'best_rmse': best_rmse,
            'cv_results': grid_search.cv_results_
        }
    
    def tune_gradient_boosting(self, X_train, y_train, param_grid: dict = None) -> dict:
        """Tune Gradient Boosting hyperparameters."""
        param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'subsample': [0.8, 0.9],
            'max_features': ['sqrt', 'log2']
        }
        
        total = np.prod([len(v) for v in param_grid.values()])
        logger.info(f"Tuning Gradient Boosting with {total} combinations...")
        
        grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=param_grid,
            cv=self.tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"Best GB params: {best_params} (CV RMSE: {best_rmse:.4f}%)")
        
        return {
            'best_params': best_params,
            'best_rmse': best_rmse,
            'cv_results': grid_search.cv_results_
        }