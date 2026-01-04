import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlalchemy

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_check_if_table_exists_table_exists(data_manager):
    table_name = "test_table"
    mock_df = pd.DataFrame({"is_exists": [1]})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
        result = data_manager.check_if_table_exists(table_name, con=mock_con)
        
        expected_sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table_name}'
            """
        mock_read_sql.assert_called_once_with(expected_sql, mock_con)
        assert result is True


def test_check_if_table_exists_table_not_exists(data_manager):
    table_name = "non_existent_table"
    mock_df = pd.DataFrame({"is_exists": [0]})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
        result = data_manager.check_if_table_exists(table_name, con=mock_con)
        
        expected_sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table_name}'
            """
        mock_read_sql.assert_called_once_with(expected_sql, mock_con)
        assert result is False


def test_check_if_table_exists_empty_result(data_manager):
    table_name = "test_table"
    mock_df = pd.DataFrame({"is_exists": []})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
        result = data_manager.check_if_table_exists(table_name, con=mock_con)
        
        expected_sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table_name}'
            """
        mock_read_sql.assert_called_once_with(expected_sql, mock_con)
        assert result is False


def test_check_if_table_exists_database_error(data_manager):
    table_name = "test_table"
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", side_effect=sqlalchemy.exc.SQLAlchemyError("DB error")) as mock_read_sql:
        with pytest.raises(sqlalchemy.exc.SQLAlchemyError):
            data_manager.check_if_table_exists(table_name, con=mock_con)
        
        expected_sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table_name}'
            """
        mock_read_sql.assert_called_once_with(expected_sql, mock_con)


def test_check_if_table_exists_without_connection_param(data_manager):
    table_name = "test_table"
    mock_df = pd.DataFrame({"is_exists": [1]})
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine) as mock_create_engine, \
         patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
        
        result = data_manager.check_if_table_exists(table_name)
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        expected_sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table_name}'
            """
        mock_read_sql.assert_called_once_with(expected_sql, mock_con)
        assert result is True
