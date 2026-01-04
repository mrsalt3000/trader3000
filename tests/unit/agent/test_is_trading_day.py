import pytest
from unittest.mock import patch, MagicMock
import datetime
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    return DataManager()


def test_is_trading_day_true(data_manager):
    schedule = "MOEX"
    today = str(datetime.datetime.now().date())
    
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_df) as mock_sql:
        result = data_manager.is_trading_day(schedule)
        
        expected_sql = f"""
        SELECT
            *
        FROM trading_schedules
        WHERE schedule = '{schedule.upper()}'
        AND date = '{today}'
                """
        mock_sql.assert_called_once_with(expected_sql)
        assert result is True


def test_is_trading_day_false(data_manager):
    schedule = "MOEX"
    today = str(datetime.datetime.now().date())
    
    mock_df = MagicMock()
    mock_df.__len__.return_value = 0
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_df) as mock_sql:
        result = data_manager.is_trading_day(schedule)
        
        expected_sql = f"""
        SELECT
            *
        FROM trading_schedules
        WHERE schedule = '{schedule.upper()}'
        AND date = '{today}'
                """
        mock_sql.assert_called_once_with(expected_sql)
        assert result is False


def test_is_trading_day_none_result(data_manager):
    schedule = "MOEX"
    
    with patch.object(data_manager, 'common_sql_req', return_value=None) as mock_sql:
        result = data_manager.is_trading_day(schedule)
        
        assert result is False


def test_is_trading_day_uppercase_conversion(data_manager):
    schedule = "moex"
    today = str(datetime.datetime.now().date())
    
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_df) as mock_sql:
        result = data_manager.is_trading_day(schedule)
        
        expected_sql = f"""
        SELECT
            *
        FROM trading_schedules
        WHERE schedule = '{schedule.upper()}'
        AND date = '{today}'
                """
        mock_sql.assert_called_once_with(expected_sql)
        assert result is True
