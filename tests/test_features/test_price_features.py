import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.features.price_features import PriceFeatureBuilder
from src.features.target_builder import TargetBuilder
from src.features.feature_selector import FeatureSelector


@pytest.fixture
def btc_series():
    """Realistic BTC price series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2018-01-01', periods=600, freq='D')
    prices = 10000 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, 600)))
    return pd.Series(prices, index=dates, name='bitcoin')


class TestPriceFeatureBuilder:

    def test_build_returns_dataframe(self, btc_series):
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        assert isinstance(features, pd.DataFrame)

    def test_build_same_length_as_input(self, btc_series):
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        assert len(features) == len(btc_series)

    def test_no_raw_ma_price_columns(self, btc_series):
        """Raw MA price columns must not exist - they cause scaling issues."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        for w in [7, 30, 50, 200]:
            assert f'ma_{w}d' not in features.columns, \
                f"Raw MA column ma_{w}d found - remove it to prevent scaling issues"

    def test_ratio_columns_exist(self, btc_series):
        """Scale-free ratio columns should exist."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        assert 'price_vs_ma7' in features.columns
        assert 'price_vs_ma30' in features.columns
        assert 'ma7_vs_ma30' in features.columns
        assert 'ma30_vs_ma200' in features.columns

    def test_return_columns_exist(self, btc_series):
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        for w in [1, 7, 14, 30]:
            assert f'return_{w}d' in features.columns

    def test_no_future_leak_in_returns(self, btc_series):
        """First row should have NaN for return_1d (no previous price)."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        assert pd.isna(features['return_1d'].iloc[0])

    def test_feature_values_reasonable(self, btc_series):
        """Features should not contain extreme values after raw MA removal."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        clean = features.dropna()
        assert clean.max().max() < 100000, "Extreme values detected in features"
        assert clean.min().min() > -100000, "Extreme values detected in features"

    def test_rsi_bounded(self, btc_series):
        """RSI must be between 0 and 100."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        rsi = features['rsi_14d'].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()


class TestTargetBuilder:

    def test_build_returns_series(self, btc_series):
        builder = TargetBuilder(horizon_days=30)
        target = builder.build(btc_series)
        assert isinstance(target, pd.Series)

    def test_build_same_length_as_input(self, btc_series):
        builder = TargetBuilder(horizon_days=30)
        target = builder.build(btc_series)
        assert len(target) == len(btc_series)

    def test_last_rows_are_nan(self, btc_series):
        """Last ~30 rows should be NaN (no future data available)."""
        builder = TargetBuilder(horizon_days=30)
        target = builder.build(btc_series)
        assert target.iloc[-1] is np.nan or pd.isna(target.iloc[-1])

    def test_target_values_are_percentages(self, btc_series):
        """Target should be in percentage terms, not raw prices."""
        builder = TargetBuilder(horizon_days=30)
        target = builder.build(btc_series)
        valid = target.dropna()
        # Should be percent returns, not price levels
        assert valid.abs().max() < 10000


class TestFeatureSelector:

    def test_temporal_split_no_overlap(self, btc_series):
        """Train and test sets must not overlap."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        target = TargetBuilder(30).build(btc_series)

        selector = FeatureSelector()
        dataset = selector.build_dataset(features, target)
        X_train, X_test, y_train, y_test = selector.temporal_split(dataset, 0.7)

        assert X_train.index[-1] < X_test.index[0]

    def test_temporal_split_ratio(self, btc_series):
        """Split should respect train_ratio."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        target = TargetBuilder(30).build(btc_series)

        selector = FeatureSelector()
        dataset = selector.build_dataset(features, target)
        X_train, X_test, _, _ = selector.temporal_split(dataset, 0.7)

        total = len(X_train) + len(X_test)
        actual_ratio = len(X_train) / total
        assert abs(actual_ratio - 0.7) < 0.01

    def test_no_nan_in_final_dataset(self, btc_series):
        """Final dataset should have no NaN values."""
        builder = PriceFeatureBuilder()
        features = builder.build(btc_series)
        target = TargetBuilder(30).build(btc_series)

        selector = FeatureSelector()
        dataset = selector.build_dataset(features, target)
        assert not dataset.isna().any().any()