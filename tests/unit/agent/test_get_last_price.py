import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.common_sql_req = MagicMock()
    return manager


def test_get_last_price_happy_path(mock_data_manager):
    expected_price = 150.75
    mock_result = pd.DataFrame({"close": [expected_price]})
    mock_data_manager.common_sql_req.return_value = mock_result
    
    result = mock_data_manager.get_last_price("SBER", "hour")
    
    assert result == expected_price
    mock_data_manager.common_sql_req.assert_called_once_with(
        """
            SELECT
                close
            FROM candles_SBER_hour
            ORDER BY date DESC 
            LIMIT 1
            """
    )


def test_get_last_price_empty_result_raises_value_error(mock_data_manager):
    mock_data_manager.common_sql_req.return_value = pd.DataFrame()
    
    with pytest.raises(ValueError, match="No data returned for query:"):
        mock_data_manager.get_last_price("AAPL", "15min")


def test_get_last_price_none_result_raises_value_error(mock_data_manager):
    mock_data_manager.common_sql_req.return_value = None
    
    with pytest.raises(ValueError, match="No data returned for query:"):
        mock_data_manager.get_last_price("GOOGL", "hour")


def test_get_last_price_multiple_rows_returns_first(mock_data_manager):
    mock_result = pd.DataFrame({"close": [100.0, 101.5, 99.8]})
    mock_data_manager.common_sql_req.return_value = mock_result
    
    result = mock_data_manager.get_last_price("TSLA", "hour")
    
    assert result == 100.0


def test_get_last_price_with_different_ticker_and_resolution(mock_data_manager):
    test_cases = [
        ("AAPL", "15min", 175.25),
        ("MSFT", "hour", 350.50),
        ("BTC-USD", "hour", 45000.00),
    ]
    
    for ticker, resolution, expected_price in test_cases:
        mock_data_manager.common_sql_req.reset_mock()
        mock_result = pd.DataFrame({"close": [expected_price]})
        mock_data_manager.common_sql_req.return_value = mock_result
        
        result = mock_data_manager.get_last_price(ticker, resolution)
        
        assert result == expected_price
        expected_sql = f"""
            SELECT
                close
            FROM candles_{ticker}_{resolution}
            ORDER BY date DESC 
            LIMIT 1
            """
        mock_data_manager.common_sql_req.assert_called_once_with(expected_sql)
