import datetime
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_trading_manager():
    manager = MagicMock()
    return manager


@pytest.fixture
def data_manager(mock_trading_manager):
    dm = DataManager(trading_manager=mock_trading_manager)
    dm.psql_main_creds = "postgresql://test:test@localhost/test"
    return dm


def test_get_schedules_happy_path(data_manager, mock_trading_manager):
    from_dt = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    to_dt = datetime.datetime(2024, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_schedule = MagicMock()
    mock_schedule.exchange = "MOEX"
    
    mock_day = MagicMock()
    mock_day.is_trading_day = True
    mock_day.date = MagicMock()
    mock_day.date.date.return_value = datetime.date(2024, 1, 1)
    mock_day.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    mock_day.end_time = datetime.datetime(2024, 1, 1, 18, 0, 0, tzinfo=datetime.timezone.utc)
    mock_schedule.days = [mock_day]
    
    mock_trading_manager.get_trading_schedules.return_value = [mock_schedule]
    
    mock_connection = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        with patch.object(data_manager, 'seconds_between_dt', return_value=28800):
            with patch('sqlalchemy.create_engine') as mock_engine:
                mock_engine.return_value.connect.return_value = mock_connection
                data_manager.get_schedules(from_=from_dt, to=to_dt)
    
    mock_trading_manager.get_trading_schedules.assert_called_once_with(
        from_=from_dt, to=to_dt
    )
    
    mock_to_sql.assert_called_once()
    call_args = mock_to_sql.call_args
    assert call_args[1]['con'] == mock_connection
    assert call_args[1]['name'] == "schedules"
    assert call_args[1]['schema'] == "trading"
    assert call_args[1]['if_exists'] == "append"
    assert call_args[1]['index'] == False


def test_get_schedules_no_trading_days(data_manager, mock_trading_manager):
    from_dt = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    to_dt = datetime.datetime(2024, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_schedule = MagicMock()
    mock_schedule.exchange = "MOEX"
    
    mock_day = MagicMock()
    mock_day.is_trading_day = False
    mock_schedule.days = [mock_day]
    
    mock_trading_manager.get_trading_schedules.return_value = [mock_schedule]
    
    mock_connection = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_engine.return_value.connect.return_value = mock_connection
            data_manager.get_schedules(from_=from_dt, to=to_dt)
    
    mock_trading_manager.get_trading_schedules.assert_called_once_with(
        from_=from_dt, to=to_dt
    )
    
    mock_to_sql.assert_called_once()
    call_args = mock_to_sql.call_args
    assert call_args[1]['con'] == mock_connection
    assert call_args[1]['name'] == "schedules"
    assert call_args[1]['schema'] == "trading"
    assert call_args[1]['if_exists'] == "append"
    assert call_args[1]['index'] == False


def test_get_schedules_exception_in_trading_manager(data_manager, mock_trading_manager):
    from_dt = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    to_dt = datetime.datetime(2024, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_trading_manager.get_trading_schedules.side_effect = Exception("API error")
    
    mock_connection = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_engine.return_value.connect.return_value = mock_connection
            with pytest.raises(Exception, match="API error"):
                data_manager.get_schedules(from_=from_dt, to=to_dt)
    
    mock_trading_manager.get_trading_schedules.assert_called_once_with(
        from_=from_dt, to=to_dt
    )
    
    mock_to_sql.assert_not_called()


def test_get_schedules_multiple_schedules_and_days(data_manager, mock_trading_manager):
    from_dt = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    to_dt = datetime.datetime(2024, 1, 3, 0, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_schedule1 = MagicMock()
    mock_schedule1.exchange = "MOEX"
    
    mock_day1 = MagicMock()
    mock_day1.is_trading_day = True
    mock_day1.date = MagicMock()
    mock_day1.date.date.return_value = datetime.date(2024, 1, 1)
    mock_day1.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    mock_day1.end_time = datetime.datetime(2024, 1, 1, 18, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_day2 = MagicMock()
    mock_day2.is_trading_day = True
    mock_day2.date = MagicMock()
    mock_day2.date.date.return_value = datetime.date(2024, 1, 2)
    mock_day2.start_time = datetime.datetime(2024, 1, 2, 10, 0, 0, tzinfo=datetime.timezone.utc)
    mock_day2.end_time = datetime.datetime(2024, 1, 2, 18, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_schedule1.days = [mock_day1, mock_day2]
    
    mock_schedule2 = MagicMock()
    mock_schedule2.exchange = "SPBX"
    
    mock_day3 = MagicMock()
    mock_day3.is_trading_day = True
    mock_day3.date = MagicMock()
    mock_day3.date.date.return_value = datetime.date(2024, 1, 1)
    mock_day3.start_time = datetime.datetime(2024, 1, 1, 9, 30, 0, tzinfo=datetime.timezone.utc)
    mock_day3.end_time = datetime.datetime(2024, 1, 1, 16, 0, 0, tzinfo=datetime.timezone.utc)
    
    mock_schedule2.days = [mock_day3]
    
    mock_trading_manager.get_trading_schedules.return_value = [mock_schedule1, mock_schedule2]
    
    mock_connection = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        with patch.object(data_manager, 'seconds_between_dt', return_value=28800):
            with patch('sqlalchemy.create_engine') as mock_engine:
                mock_engine.return_value.connect.return_value = mock_connection
                data_manager.get_schedules(from_=from_dt, to=to_dt)
    
    mock_trading_manager.get_trading_schedules.assert_called_once_with(
        from_=from_dt, to=to_dt
    )
    
    assert mock_to_sql.call_count == 1
    call_args = mock_to_sql.call_args
    assert call_args[1]['con'] == mock_connection
    assert call_args[1]['name'] == "schedules"
    assert call_args[1]['schema'] == "trading"
    assert call_args[1]['if_exists'] == "append"
    assert call_args[1]['index'] == False
