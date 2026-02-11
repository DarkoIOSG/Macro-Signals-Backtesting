import pytest
import pandas as pd
import numpy as np
from src.features.price_features import PriceFeatureBuilder


def test_price_features_shape(sample_btc_prices):
    """Test that price features have correct shape."""
    builder = PriceFeatureBuilder()
    btc = sample_btc_prices['bitcoin']
    
    features = builder.build(btc)
    
    assert len(features) == len(btc)
    assert len(features.columns) > 0


def test_price_features_columns(sample_btc_prices):
    """Test that expected columns are created."""
    builder = PriceFeatureBuilder()
    btc = sample_btc_prices['bitcoin']
    
    features = builder.build(btc)
    
    assert 'return_1d' in features.columns
    assert 'return_30d' in features.columns
    assert 'volatility_30d' in features.columns
    assert 'rsi_14d' in features.columns


def test_price_features_no_future_leak(sample_btc_prices):
    """Test that features don't use future data."""
    builder = PriceFeatureBuilder()
    btc = sample_btc_prices['bitcoin']
    
    features = builder.build(btc)
    
    # First 365 rows should have NaN for long-window features
    assert features['return_365d'].iloc[:364].isna().all()