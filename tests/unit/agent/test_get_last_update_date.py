import pytest
from unittest.mock import MagicMock, patch
import datetime

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_last_update_date_success(data_manager):
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (datetime.datetime(2024, 1, 15, 10, 30, 0),)
    mock_con.execute.return_value = mock_result
    
    result = data_manager.get_last_update_date("AAPL", mock_con)
    
    mock_con.execute.assert_called_once()
    assert result == datetime.datetime(2024, 1, 15, 10, 30, 0)


def test_get_last_update_date_no_data(data_manager):
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (None,)
    mock_con.execute.return_value = mock_result
    
    result = data_manager.get_last_update_date("GOOGL", mock_con)
    
    mock_con.execute.assert_called_once()
    assert result is None


def test_get_last_update_date_empty_result(data_manager):
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (None,)
    mock_con.execute.return_value = mock_result
    
    result = data_manager.get_last_update_date("MSFT", mock_con)
    
    mock_con.execute.assert_called_once()
    assert result is None


def test_get_last_update_date_with_none_connection(data_manager):
    with pytest.raises(ValueError, match="Database connection 'con' is required and cannot be None"):
        data_manager.get_last_update_date("TSLA", None)


def test_get_last_update_date_table_name_lowercase(data_manager):
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (datetime.datetime(2024, 2, 1, 12, 0, 0),)
    mock_con.execute.return_value = mock_result
    
    result = data_manager.get_last_update_date("SBER", mock_con)
    
    mock_con.execute.assert_called_once()
    assert result == datetime.datetime(2024, 2, 1, 12, 0, 0)
