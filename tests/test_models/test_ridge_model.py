import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from src.models.ridge_model import RidgeModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel


@pytest.fixture
def sample_data():
    """Small dataset for fast model testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n)
    X = pd.DataFrame(
        np.random.randn(n, 10),
        columns=[f'feature_{i}' for i in range(10)],
        index=dates
    )
    y = pd.Series(np.random.randn(n) * 10, index=dates, name='target')
    return X, y


class TestRidgeModel:

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y, best_alpha=100.0)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_is_fitted_after_fit(self, sample_data):
        X, y = sample_data
        model = RidgeModel()
        assert not model.is_fitted
        model.fit(X, y, best_alpha=100.0)
        assert model.is_fitted

    def test_predict_raises_before_fit(self, sample_data):
        X, _ = sample_data
        model = RidgeModel()
        with pytest.raises(ValueError):
            model.predict(X)

    def test_feature_importance_returns_dataframe(self, sample_data):
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y, best_alpha=100.0)
        importance = model.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
        assert len(importance) == X.shape[1]

    def test_save_load(self, tmp_path, sample_data):
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y, best_alpha=100.0)

        path = str(tmp_path / "ridge.pkl")
        model.save(path)
        loaded = RidgeModel.load(path)

        np.testing.assert_array_almost_equal(
            model.predict(X),
            loaded.predict(X)
        )

    def test_predictions_are_finite(self, sample_data):
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y, best_alpha=100.0)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))


class TestRandomForestModel:

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_feature_importance_sums_to_one(self, sample_data):
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert abs(importance['importance'].sum() - 1.0) < 0.001


class TestGradientBoostingModel:

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_feature_importance_sums_to_one(self, sample_data):
        X, y = sample_data
        model = GradientBoostingModel()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert abs(importance['importance'].sum() - 1.0) < 0.001