import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_dm():
    dm = DataManager()
    dm.psql_main_creds = "postgresql://test:test@localhost/test"
    return dm


def test_get_tickers_from_datasets_happy_path(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": [
            "datasets_aapl_hour_cost_history",
            "datasets_aapl_hour_indicators",
            "datasets_googl_hour_cost_history",
            "datasets_googl_hour_indicators",
            "datasets_tsla_15min_cost_history",
            "other_table",
            "datasets_msft_hour_validation_indicators"
        ]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert sorted(result) == ["aapl", "googl", "msft"]
            assert len(result) == 3


def test_get_tickers_from_datasets_empty_result(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": [
            "datasets_aapl_15min_cost_history",
            "datasets_googl_15min_indicators",
            "other_table"
        ]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert result == []


def test_get_tickers_from_datasets_with_connection_param(mock_dm):
    mock_con = MagicMock()
    mock_tables_info = pd.DataFrame({
        "table": ["datasets_aapl_hour_cost_history"]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info) as mock_get_tables:
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(con=mock_con, resolution="hour")
            
            mock_get_tables.assert_called_once_with(con=mock_con)
            assert result == ["aapl"]


def test_get_tickers_from_datasets_no_datasets_tables(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": ["candles_aapl_hour", "analytics_googl", "other_table"]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert result == []


def test_get_tickers_from_datasets_empty_tables_info(mock_dm):
    mock_tables_info = pd.DataFrame(columns=["table"])
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert result == []


def test_get_tickers_from_datasets_complex_table_names(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": [
            "datasets_btc_usdt_hour_cost_history",
            "datasets_eth_usdt_hour_indicators",
            "datasets_btc_usdt_hour_validation_indicators"
        ]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert sorted(result) == ["btc_usdt", "eth_usdt"]
            assert len(result) == 2


def test_get_tickers_from_datasets_different_resolutions(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": [
            "datasets_aapl_hour_cost_history",
            "datasets_aapl_15min_cost_history",
            "datasets_googl_hour_indicators",
            "datasets_googl_30min_indicators",
            "datasets_tsla_daily_cost_history"
        ]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            
            result_hour = mock_dm.get_tickers_from_datasets(resolution="hour")
            assert sorted(result_hour) == ["aapl", "googl"]
            
            result_15min = mock_dm.get_tickers_from_datasets(resolution="15min")
            assert result_15min == ["aapl"]
            
            result_daily = mock_dm.get_tickers_from_datasets(resolution="daily")
            assert result_daily == ["tsla"]


def test_get_tickers_from_datasets_invalid_table_name_format(mock_dm):
    mock_tables_info = pd.DataFrame({
        "table": [
            "datasets_invalid_table",
            "datasets_",
            "datasets_aapl_hour"
        ]
    })
    
    with patch.object(mock_dm, 'get_tables_info', return_value=mock_tables_info):
        with patch.object(mock_dm, 'common_sql_req') as mock_sql_req:
            mock_sql_req.return_value = mock_tables_info
            result = mock_dm.get_tickers_from_datasets(resolution="hour")
            
            assert result == ["aapl"]
