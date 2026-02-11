import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_btc_prices():
    """Sample Bitcoin price data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    prices = pd.Series(
        10000 * np.exp(np.cumsum(np.random.normal(0, 0.02, 500))),
        index=dates,
        name='bitcoin'
    )
    return pd.DataFrame({'bitcoin': prices})


@pytest.fixture
def sample_vix():
    """Sample VIX data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    return pd.DataFrame({
        'VIX': np.random.uniform(15, 40, 500)
    }, index=dates)


@pytest.fixture
def sample_dataset():
    """Sample complete dataset for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    
    data = {f'feature_{i}': np.random.randn(n) for i in range(10)}
    data['target'] = np.random.randn(n) * 10
    
    return pd.DataFrame(data, index=dates)