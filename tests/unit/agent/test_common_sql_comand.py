import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_common_sql_comand_success(data_manager):
    test_sql = "SELECT * FROM test_table"
    mock_con = MagicMock()
    
    with patch('pandas.read_sql') as mock_read_sql:
        data_manager.common_sql_comand(test_sql, con=mock_con)
        
        mock_read_sql.assert_called_once_with(test_sql, mock_con)


def test_common_sql_comand_with_exception(data_manager):
    test_sql = "SELECT * FROM test_table"
    test_error = Exception("Database connection error")
    mock_con = MagicMock()
    
    with patch('pandas.read_sql', side_effect=test_error) as mock_read_sql, \
         patch('time.sleep') as mock_sleep:
        
        data_manager.common_sql_comand(test_sql, con=mock_con)
        
        mock_read_sql.assert_called_once_with(test_sql, mock_con)
        mock_sleep.assert_called_once_with(10)


def test_common_sql_comand_empty_sql(data_manager):
    empty_sql = ""
    mock_con = MagicMock()
    
    with patch('pandas.read_sql') as mock_read_sql:
        data_manager.common_sql_comand(empty_sql, con=mock_con)
        
        mock_read_sql.assert_called_once_with(empty_sql, mock_con)


def test_common_sql_comand_with_decorator_connection(data_manager):
    test_sql = "SELECT * FROM test_table"
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine), \
         patch('pandas.read_sql') as mock_read_sql:
        
        data_manager.common_sql_comand(test_sql)
        
        mock_engine.connect.assert_called_once()
        mock_read_sql.assert_called_once_with(test_sql, mock_con)
        mock_con.commit.assert_called_once()
        mock_con.close.assert_called_once()
