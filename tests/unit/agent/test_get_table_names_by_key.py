import pytest
from unittest.mock import MagicMock, patch, create_autospec
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_table_names_by_key_happy_path(data_manager):
    mock_connection = MagicMock(spec=Connection)
    mock_result = MagicMock()
    mock_result.__iter__.return_value = [("table1",), ("table2",)]
    mock_connection.execute.return_value = mock_result
    
    result = data_manager.get_table_names_by_key("test_key", con=mock_connection)
    
    expected_sql = """
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name LIKE '%test_key%'
            """
    mock_connection.execute.assert_called_once_with(text(expected_sql))
    assert result == ["table1", "table2"]


def test_get_table_names_by_key_empty_result(data_manager):
    mock_connection = MagicMock(spec=Connection)
    mock_result = MagicMock()
    mock_result.__iter__.return_value = []
    mock_connection.execute.return_value = mock_result
    
    result = data_manager.get_table_names_by_key("nonexistent", con=mock_connection)
    
    assert result == []


def test_get_table_names_by_key_none_result(data_manager):
    mock_connection = MagicMock(spec=Connection)
    mock_connection.execute.return_value = None
    
    result = data_manager.get_table_names_by_key("test", con=mock_connection)
    
    assert result == []


def test_get_table_names_by_key_connection_none(data_manager):
    with pytest.raises(ValueError, match="Database connection 'con' is required and cannot be None"):
        data_manager.get_table_names_by_key("test", con=None)


def test_get_table_names_by_key_with_special_characters(data_manager):
    mock_connection = MagicMock(spec=Connection)
    mock_result = MagicMock()
    mock_result.__iter__.return_value = [("test_table_2024",)]
    mock_connection.execute.return_value = mock_result
    
    result = data_manager.get_table_names_by_key("2024", con=mock_connection)
    
    expected_sql = """
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name LIKE '%2024%'
            """
    mock_connection.execute.assert_called_once_with(text(expected_sql))
    assert result == ["test_table_2024"]


def test_get_table_names_by_key_decorator_integration(data_manager):
    with patch('sqlalchemy.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_connection = MagicMock(spec=Connection)
        mock_result = MagicMock()
        
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_result.__iter__.return_value = [("table1",), ("table2",)]
        mock_connection.execute.return_value = mock_result
        
        result = data_manager.get_table_names_by_key("test")
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        assert result == ["table1", "table2"]
