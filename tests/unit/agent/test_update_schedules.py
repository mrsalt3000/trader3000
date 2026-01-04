import datetime
from unittest.mock import MagicMock, patch, call
import pytest
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.common_sql_req = MagicMock()
    manager.get_schedules = MagicMock()
    return manager


def test_update_schedules_happy_path(mock_data_manager):
    mock_result = pd.DataFrame({"ans": ["2024-01-15 00:00:00"]})
    mock_data_manager.common_sql_req.return_value = mock_result
    
    with patch('src.trader3000.core.data_management.datetime') as mock_dt:
        mock_now = datetime.datetime(2024, 1, 20, 10, 30, 0)
        mock_dt.datetime.now.return_value = mock_now
        
        def strptime_side_effect(date_str, format_str):
            return datetime.datetime.strptime(date_str[:19], format_str)
        mock_dt.datetime.strptime.side_effect = strptime_side_effect
        
        mock_dt.timezone.utc = datetime.timezone.utc
        mock_dt.timedelta = datetime.timedelta
        
        mock_data_manager.update_schedules()
    
    mock_data_manager.common_sql_req.assert_called_once_with(
        "SELECT max(date) AS ans FROM trading.schedules"
    )
    
    assert mock_data_manager.get_schedules.call_count == 13
    
    calls = mock_data_manager.get_schedules.call_args_list
    assert len(calls) == 13
    
    for i, call_args in enumerate(calls):
        assert 'from_' in call_args.kwargs
        assert 'to' in call_args.kwargs


def test_update_schedules_no_existing_data(mock_data_manager):
    mock_result = pd.DataFrame({"ans": [None]})
    mock_data_manager.common_sql_req.return_value = mock_result
    
    with patch('src.trader3000.core.data_management.datetime') as mock_dt:
        mock_now = datetime.datetime(2024, 1, 20, 10, 30, 0)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timezone.utc = datetime.timezone.utc
        mock_dt.timedelta = datetime.timedelta
        
        mock_data_manager.update_schedules()
    
    assert mock_data_manager.get_schedules.call_count == 13


def test_update_schedules_exception_handling(mock_data_manager):
    mock_result = pd.DataFrame({"ans": ["2024-01-10 00:00:00"]})
    mock_data_manager.common_sql_req.return_value = mock_result
    
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise Exception("Test error")
        return None
    
    mock_data_manager.get_schedules.side_effect = side_effect
    
    with patch('src.trader3000.core.data_management.datetime') as mock_dt:
        mock_now = datetime.datetime(2024, 1, 20, 10, 30, 0)
        mock_dt.datetime.now.return_value = mock_now
        
        def strptime_side_effect(date_str, format_str):
            return datetime.datetime.strptime(date_str[:19], format_str)
        mock_dt.datetime.strptime.side_effect = strptime_side_effect
        
        mock_dt.timezone.utc = datetime.timezone.utc
        mock_dt.timedelta = datetime.timedelta
        
        with patch('builtins.print') as mock_print:
            mock_data_manager.update_schedules()
    
    assert mock_data_manager.get_schedules.call_count == 3
    mock_print.assert_called_with("Error: \nTest erro