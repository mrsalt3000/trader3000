import datetime
from unittest.mock import MagicMock
import pytest
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.common_sql_req = MagicMock()
    manager.get_timeline = MagicMock()
    return manager


def test_get_target_sell_dt_happy_path(mock_data_manager):
    schedule_name = "MOEX"
    buy_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
    resolution = "hour"
    steps = 3
    
    mock_schedule_df = pd.DataFrame({
        'schedule': ['MOEX'],
        'date': ['2024-01-15'],
        'work_seconds': [28800],
        'start_time': ['2024-01-15 10:00:00']
    })
    mock_data_manager.common_sql_req.return_value = mock_schedule_df
    
    timeline = [
        datetime.datetime(2024, 1, 15, 10, 0, 0),
        datetime.datetime(2024, 1, 15, 11, 0, 0),
        datetime.datetime(2024, 1, 15, 12, 0, 0),
        datetime.datetime(2024, 1, 15, 13, 0, 0),
        datetime.datetime(2024, 1, 15, 14, 0, 0),
        datetime.datetime(2024, 1, 15, 15, 0, 0),
    ]
    mock_data_manager.get_timeline.return_value = timeline
    
    result = mock_data_manager.get_target_sell_dt(
        schedule_name, buy_dt, resolution, steps
    )
    
    mock_data_manager.common_sql_req.assert_called_once()
    mock_data_manager.get_timeline.assert_called_once_with(mock_schedule_df, 3600)
    
    assert result == datetime.datetime(2024, 1, 15, 14, 0, 0)


def test_get_target_sell_dt_boundary_case_insufficient_steps(mock_data_manager):
    schedule_name = "MOEX"
    buy_dt = datetime.datetime(2024, 1, 15, 15, 0, 0)
    resolution = "hour"
    steps = 5
    
    mock_schedule_df = pd.DataFrame({
        'schedule': ['MOEX'],
        'date': ['2024-01-15'],
        'work_seconds': [28800],
        'start_time': ['2024-01-15 10:00:00']
    })
    mock_data_manager.common_sql_req.return_value = mock_schedule_df
    
    timeline = [
        datetime.datetime(2024, 1, 15, 10, 0, 0),
        datetime.datetime(2024, 1, 15, 11, 0, 0),
        datetime.datetime(2024, 1, 15, 12, 0, 0),
        datetime.datetime(2024, 1, 15, 13, 0, 0),
        datetime.datetime(2024, 1, 15, 14, 0, 0),
        datetime.datetime(2024, 1, 15, 15, 0, 0),
        datetime.datetime(2024, 1, 15, 16, 0, 0),
        datetime.datetime(2024, 1, 15, 17, 0, 0),
    ]
    mock_data_manager.get_timeline.return_value = timeline
    
    result = mock_data_manager.get_target_sell_dt(
        schedule_name, buy_dt, resolution, steps
    )
    
    assert result == datetime.datetime(2100, 1, 1, 1)


def test_get_target_sell_dt_error_case_no_schedule_data(mock_data_manager):
    schedule_name = "UNKNOWN_SCHEDULE"
    buy_dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
    resolution = "hour"
    steps = 5
    
    mock_data_manager.common_sql_req.return_value = None
    
    result = mock_data_manager.get_target_sell_dt(
        schedule_name, buy_dt, resolution, steps
    )
    
    mock_data_manager.common_sql_req.assert_called_once()
    assert result == datetime.datetime(2100, 1, 1, 1)


def test_get_target_sell_dt_with_15min_resolution(mock_data_manager):
    schedule_name = "MOEX"
    buy_dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
    resolution = "15min"
    steps = 4
    
    mock_schedule_df = pd.DataFrame({
        'schedule': ['MOEX'],
        'date': ['2024-01-15'],
        'work_seconds': [28800],
        'start_time': ['2024-01-15 10:00:00']
    })
    mock_data_manager.common_sql_req.return_value = mock_schedule_df
    
    timeline = [
        datetime.datetime(2024, 1, 15, 10, 0, 0),
        datetime.datetime(2024, 1, 15, 10, 15, 0),
        datetime.datetime(2024, 1, 15, 10, 30, 0),
        datetime.datetime(2024, 1, 15, 10, 45, 0),
        datetime.datetime(2024, 1, 15, 11, 0, 0),
        datetime.datetime(2024, 1, 15, 11, 15, 0),
    ]
    mock_data_manager.get_timeline.return_value = timeline
    
    result = mock_data_manager.get_target_sell_dt(
        schedule_name, buy_dt, resolution, steps
    )
    
    mock_data_manager.get_timeline.assert_called_once_with(mock_schedule_df, 900)
    assert result == datetime.datetime(2024, 1, 15, 11, 0, 0)
