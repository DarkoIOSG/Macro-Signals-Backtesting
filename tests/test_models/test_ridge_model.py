import pytest
import numpy as np
import pandas as pd
from src.models.ridge_model import RidgeModel


def test_ridge_fit_predict(sample_dataset):
    """Test Ridge model fits and predicts."""
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']
    
    model = RidgeModel()
    model.fit(X, y, best_alpha=100.0)
    
    assert model.is_fitted
    
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_ridge_feature_importance(sample_dataset):
    """Test feature importance is returned correctly."""
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']
    
    model = RidgeModel()
    model.fit(X, y, best_alpha=100.0)
    
    importance = model.get_feature_importance()
    
    assert 'feature' in importance.columns
    assert 'coefficient' in importance.columns
    assert len(importance) == len(X.columns)


def test_ridge_save_load(tmp_path, sample_dataset):
    """Test model saves and loads correctly."""
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']
    
    model = RidgeModel()
    model.fit(X, y, best_alpha=100.0)
    
    save_path = tmp_path / "test_model.pkl"
    model.save(str(save_path))
    
    loaded_model = RidgeModel.load(str(save_path))
    
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)