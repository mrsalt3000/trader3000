import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, call

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_trading_log_happy_path(data_manager):
    expected_data = pd.DataFrame({
        'id': [1, 2, 3],
        'operation': ['buy', 'sell', 'buy'],
        'amount': [100.0, 50.0, 75.0]
    })
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_trading_log', autospec=True) as mock_method:
        mock_method.return_value = expected_data
        
        result = data_manager.get_trading_log(con=mock_con)
        
        pd.testing.assert_frame_equal(result, expected_data)


def test_get_trading_log_empty_result(data_manager):
    expected_data = pd.DataFrame()
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_trading_log', autospec=True) as mock_method:
        mock_method.return_value = expected_data
        
        result = data_manager.get_trading_log(con=mock_con)
        
        assert result.empty
        assert len(result) == 0
        pd.testing.assert_frame_equal(result, expected_data)


def test_get_trading_log_sql_exception(data_manager):
    mock_con = MagicMock()
    
    with patch('src.trader3000.core.data_management.pd.read_sql') as mock_read_sql:
        mock_read_sql.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            data_manager.get_trading_log(con=mock_con)
        
        assert "Database connection failed" in str(exc_info.value)
        mock_read_sql.assert_called_once()


def test_get_trading_log_without_connection_parameter(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    expected_data = pd.DataFrame({'test': [1, 2, 3]})
    
    with patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_read_sql:
        
        mock_create_engine.return_value = mock_engine
        mock_read_sql.return_value = expected_data
        
        result = data_manager.get_trading_log()
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_engine.connect.assert_called_once()
        mock_read_sql.assert_called_once()
        mock_con.commit.assert_called_once()
        mock_con.close.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_data)


def test_get_trading_log_rollback_on_exception(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_read_sql:
        
        mock_create_engine.return_value = mock_engine
        mock_read_sql.side_effect = Exception("Test error")
        
        with pytest.raises(Exception) as exc_info:
            data_manager.get_trading_log()
        
        assert "Test error" in str(exc_info.value)
        mock_con.rollback.assert_called_once()
        mock_con.commit.assert_not_called()
        mock_con.close.assert_called_once()
