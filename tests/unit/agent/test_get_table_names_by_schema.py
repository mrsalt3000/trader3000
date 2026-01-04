import pytest
from unittest.mock import MagicMock, patch, create_autospec
import pandas as pd
from sqlalchemy import text

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    """Фикстура для создания экземпляра DataManager."""
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_table_names_by_schema_happy_path(data_manager):
    """Тест нормального поведения: возвращает список таблиц для существующей схемы."""
    # Arrange
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("table1",), ("table2",), ("table3",)]
    mock_con.execute.return_value = mock_result
    
    expected_tables = ["table1", "table2", "table3"]
    
    # Act
    result = data_manager.get_table_names_by_schema("candles", con=mock_con)
    
    # Assert
    assert result == expected_tables
    mock_con.execute.assert_called_once_with(
        text("""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_schema LIKE '%candles%'
            """)
    )


def test_get_table_names_by_schema_empty_result(data_manager):
    """Тест граничного случая: схема существует, но таблиц в ней нет."""
    # Arrange
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_con.execute.return_value = mock_result
    
    # Act
    result = data_manager.get_table_names_by_schema("empty_schema", con=mock_con)
    
    # Assert
    assert result == []
    mock_con.execute.assert_called_once()


def test_get_table_names_by_schema_none_connection(data_manager):
    """Тест ошибочной ситуации: передано None соединение."""
    # Act & Assert
    with pytest.raises(ValueError, match="Database connection 'con' is required and cannot be None"):
        data_manager.get_table_names_by_schema("candles", con=None)


def test_get_table_names_by_schema_execute_returns_none(data_manager):
    """Тест случая, когда execute возвращает None."""
    # Arrange
    mock_con = MagicMock()
    mock_con.execute.return_value = None
    
    # Act
    result = data_manager.get_table_names_by_schema("candles", con=mock_con)
    
    # Assert
    assert result == []
    mock_con.execute.assert_called_once()


def test_get_table_names_by_schema_partial_match(data_manager):
    """Тест поиска по частичному совпадению имени схемы."""
    # Arrange
    mock_con = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("candles_aapl",), ("candles_googl",)]
    mock_con.execute.return_value = mock_result
    
    expected_tables = ["candles_aapl", "candles_googl"]
    
    # Act
    result = data_manager.get_table_names_by_schema("candle", con=mock_con)
    
    # Assert
    assert result == expected_tables
    mock_con.execute.assert_called_once_with(
        text("""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_schema LIKE '%candle%'
            """)
    )
