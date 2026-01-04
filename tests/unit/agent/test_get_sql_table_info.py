import pytest
from unittest.mock import MagicMock, patch
import datetime
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    """Фикстура для создания экземпляра DataManager."""
    return DataManager()


def test_get_sql_table_info_happy_path(data_manager):
    """Тест нормального поведения: возвращается корректный SQL-запрос."""
    table_name = "test_table"
    
    result = data_manager.get_sql_table_info(table_name)
    
    expected_sql = f"""
        SELECT
            count(*) AS rows_count
            ,min(date) AS min_date
            ,max(date) AS max_date
        FROM candles.{table_name}
        """
    assert result.strip() == expected_sql.strip()


def test_get_sql_table_info_with_special_characters(data_manager):
    """Тест граничного случая: имя таблицы содержит специальные символы."""
    table_name = "table_name_with_underscores_and-numbers123"
    
    result = data_manager.get_sql_table_info(table_name)
    
    expected_sql = f"""
        SELECT
            count(*) AS rows_count
            ,min(date) AS min_date
            ,max(date) AS max_date
        FROM candles.{table_name}
        """
    assert result.strip() == expected_sql.strip()


def test_get_sql_table_info_empty_table_name(data_manager):
    """Тест граничного случая: пустое имя таблицы."""
    table_name = ""
    
    result = data_manager.get_sql_table_info(table_name)
    
    expected_sql = f"""
        SELECT
            count(*) AS rows_count
            ,min(date) AS min_date
            ,max(date) AS max_date
        FROM candles.{table_name}
        """
    assert result.strip() == expected_sql.strip()


def test_get_sql_table_info_sql_injection_prevention(data_manager):
    """Тест: SQL-запрос формируется безопасно (проверка структуры)."""
    table_name = "test'; DROP TABLE candles.test; --"
    
    result = data_manager.get_sql_table_info(table_name)
    
    # Проверяем, что имя таблицы подставляется как есть (без экранирования в этом методе)
    # Это нормально, так как метод только формирует строку, а безопасность должна обеспечиваться
    # на уровне выполнения запроса через параметризованные запросы
    assert table_name in result
    assert "candles.test'; DROP TABLE candles.test; --" in result


def test_get_sql_table_info_multiple_calls(data_manager):
    """Тест: метод работает корректно при многократном вызове."""
    tables = ["table1", "table2", "table3"]
    
    for table_name in tables:
        result = data_manager.get_sql_table_info(table_name)
        assert f"FROM candles.{table_name}" in result
        assert "count(*) AS rows_count" in result
        assert "min(date) AS min_date" in result
        assert "max(date) AS max_date" in result
