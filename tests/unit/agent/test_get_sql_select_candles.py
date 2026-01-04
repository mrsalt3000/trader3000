import pytest
from unittest.mock import patch, MagicMock
import datetime
from src.trader3000.core.data_management import DataManager


def test_get_sql_select_candles_happy_path():
    """Тест нормального формирования SQL-запроса с различными параметрами."""
    manager = DataManager()
    
    # Тест 1: Базовый случай без нормализации
    sql = manager.get_sql_select_candles(
        ticker="AAPL",
        resolution="hour",
        dataset_type="train",
        part="cost_history",
        normalized=False
    )
    expected = """
        SELECT
            *
        FROM datasets_AAPL_hour_train_cost_history
        """
    assert sql.strip() == expected.strip()
    
    # Тест 2: С нормализацией
    sql = manager.get_sql_select_candles(
        ticker="GOOG",
        resolution="1min",
        dataset_type="validation",
        part="indicators",
        normalized=True
    )
    expected = """
        SELECT
            *
        FROM datasets_GOOG_1min_validation_indicators_normalized
        """
    assert sql.strip() == expected.strip()


def test_get_sql_select_candles_edge_cases():
    """Тест граничных случаев и специальных значений."""
    manager = DataManager()
    
    # Тест 1: Тикер в верхнем регистре (сохраняется как есть)
    sql = manager.get_sql_select_candles(
        ticker="SBER",
        resolution="15min",
        dataset_type="test",
        part="cost_history",
        normalized=False
    )
    assert "datasets_SBER_15min_test_cost_history" in sql
    
    # Тест 2: Тикер в нижнем регистре (сохраняется как есть)
    sql = manager.get_sql_select_candles(
        ticker="msft",
        resolution="hour",
        dataset_type="validation",
        part="indicators",
        normalized=True
    )
    assert "datasets_msft_hour_validation_indicators_normalized" in sql
    
    # Тест 3: Пустые строки в параметрах
    sql = manager.get_sql_select_candles(
        ticker="",
        resolution="",
        dataset_type="",
        part="",
        normalized=False
    )
    assert "datasets___" in sql


def test_get_sql_select_candles_special_characters():
    """Тест обработки тикеров со специальными символами."""
    manager = DataManager()
    
    # Тест 1: Тикер с дефисом
    sql = manager.get_sql_select_candles(
        ticker="BTC-USD",
        resolution="hour",
        dataset_type="train",
        part="cost_history",
        normalized=False
    )
    assert "datasets_BTC-USD_hour_train_cost_history" in sql
    
    # Тест 2: Тикер с точкой
    sql = manager.get_sql_select_candles(
        ticker="BRK.B",
        resolution="1min",
        dataset_type="test",
        part="indicators",
        normalized=True
    )
    assert "datasets_BRK.B_1min_test_indicators_normalized" in sql


def test_get_sql_select_candles_all_dataset_types():
    """Тест всех возможных типов датасетов."""
    manager = DataManager()
    
    dataset_types = ["train", "test", "validation", "total"]
    parts = ["cost_history", "indicators"]
    
    for dataset_type in dataset_types:
        for part in parts:
            sql = manager.get_sql_select_candles(
                ticker="AAPL",
                resolution="hour",
                dataset_type=dataset_type,
                part=part,
                normalized=False
            )
            assert f"datasets_AAPL_hour_{dataset_type}_{part}" in sql


def test_get_sql_select_candles_normalized_variants():
    """Тест вариантов с нормализацией и без."""
    manager = DataManager()
    
    # Без нормализации
    sql_not_normalized = manager.get_sql_select_candles(
        ticker="TSLA",
        resolution="hour",
        dataset_type="validation",
        part="indicators",
        normalized=False
    )
    assert "normalized" not in sql_not_normalized
    
    # С нормализацией
    sql_normalized = manager.get_sql_select_candles(
        ticker="TSLA",
        resolution="hour",
        dataset_type="validation",
        part="indicators",
        normalized=True
    )
    assert "normalized" in sql_normalized
    # Проверяем, что normalized добавляется в конец имени таблицы
    assert sql_normalized.strip().endswith("_normalized")


def test_get_sql_select_candles_sql_structure():
    """Тест структуры SQL-запроса."""
    manager = DataManager()
    
    sql = manager.get_sql_select_candles(
        ticker="NVDA",
        resolution="hour",
        dataset_type="train",
        part="cost_history",
        normalized=False
    )
    
    # Проверяем базовую структуру SQL (учитываем отступы)
    assert "SELECT" in sql
    assert "*" in sql
    assert "FROM" in sql
    # Проверяем, что нет WHERE или других условий
    assert "WHERE" not in sql
    assert "ORDER BY" not in sql
    assert "LIMIT" not in sql
    # Проверяем правильность формирования имени таблицы
    assert "datasets_NVDA_hour_train_cost_history" in sql
