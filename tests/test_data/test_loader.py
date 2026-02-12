import pytest
import pandas as pd
import sys
sys.path.append('.')

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor


class TestDataLoader:

    def test_load_prices_returns_dataframe(self):
        loader = DataLoader()
        df = loader.load_prices()
        assert isinstance(df, pd.DataFrame)

    def test_load_prices_has_bitcoin_column(self):
        loader = DataLoader()
        df = loader.load_prices()
        assert 'bitcoin' in df.columns

    def test_load_prices_has_datetime_index(self):
        loader = DataLoader()
        df = loader.load_prices()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_prices_no_timezone(self):
        loader = DataLoader()
        df = loader.load_prices()
        assert df.index.tz is None

    def test_load_all_returns_dict(self):
        loader = DataLoader()
        data = loader.load_all()
        assert isinstance(data, dict)
        assert 'prices' in data

    def test_load_all_prices_not_none(self):
        loader = DataLoader()
        data = loader.load_all()
        assert data['prices'] is not None

    def test_load_prices_sorted_ascending(self):
        loader = DataLoader()
        df = loader.load_prices()
        assert df.index.is_monotonic_increasing


class TestDataValidator:

    def test_validate_prices_passes_with_good_data(self):
        loader = DataLoader()
        data = loader.load_all()
        validator = DataValidator()
        assert validator.validate_prices(data['prices']) is True

    def test_validate_prices_fails_without_bitcoin_column(self):
        validator = DataValidator()
        bad_df = pd.DataFrame(
            {'other': [1, 2, 3]},
            index=pd.date_range('2020-01-01', periods=3)
        )
        assert validator.validate_prices(bad_df) is False

    def test_validate_prices_fails_with_too_few_rows(self):
        validator = DataValidator()
        small_df = pd.DataFrame(
            {'bitcoin': [100, 200]},
            index=pd.date_range('2020-01-01', periods=2)
        )
        assert validator.validate_prices(small_df) is False


class TestDataPreprocessor:

    def test_preprocess_prices_removes_duplicates(self):
        preprocessor = DataPreprocessor()
        # Create df with duplicate index
        idx = pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-02'])
        df = pd.DataFrame({'bitcoin': [100, 100, 200]}, index=idx)
        result = preprocessor.preprocess_prices(df)
        assert len(result) == 2

    def test_preprocess_prices_removes_zeros(self):
        preprocessor = DataPreprocessor()
        idx = pd.date_range('2020-01-01', periods=3)
        df = pd.DataFrame({'bitcoin': [100, 0, 200]}, index=idx)
        result = preprocessor.preprocess_prices(df)
        assert 0 not in result['bitcoin'].values

    def test_preprocess_all_aligns_to_price_index(self):
        loader = DataLoader()
        data = loader.load_all()
        preprocessor = DataPreprocessor()
        clean = preprocessor.preprocess_all(data)

        price_len = len(clean['prices'])
        if clean['vix'] is not None:
            assert len(clean['vix']) == price_len
        if clean['hy_spreads'] is not None:
            assert len(clean['hy_spreads']) == price_len