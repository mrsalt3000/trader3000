import datetime
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    """Фикстура для создания экземпляра DataManager."""
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_candles_last_dt_happy_path(data_manager):
    """Тест нормального случая: данные есть, возвращается корректный datetime."""
    # Мокаем common_sql_req для возврата DataFrame с датой
    mock_result = pd.DataFrame({
        "ans": ["2024-01-15 14:30:00"]
    })
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        result = data_manager.get_candles_last_dt(
            ticker="SBER",
            time_step_resolution="hour"
        )
        
        # Проверяем вызов SQL
        expected_sql = """
        SELECT
            max(date) AS ans
        FROM candles.sber_hour
        """
        mock_sql.assert_called_once_with(expected_sql)
        
        # Проверяем результат
        expected_dt = datetime.datetime.strptime(
            "2024-01-15 14:30:00", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        assert result == expected_dt


def test_get_candles_last_dt_empty_result(data_manager):
    """Тест граничного случая: таблица существует, но данных нет."""
    # Мокаем common_sql_req для возврата пустого DataFrame
    mock_result = pd.DataFrame({
        "ans": [None]  # или pd.NA / np.nan
    })
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        with pytest.raises(ValueError) as exc_info:
            data_manager.get_candles_last_dt(
                ticker="AAPL",
                time_step_resolution="hour"
            )
        
        # Проверяем сообщение об ошибке
        assert "No candle data found for ticker AAPL with resolution hour" in str(exc_info.value)
        
        # Проверяем вызов SQL
        expected_sql = """
        SELECT
            max(date) AS ans
        FROM candles.aapl_hour
        """
        mock_sql.assert_called_once_with(expected_sql)


def test_get_candles_last_dt_none_result(data_manager):
    """Тест ошибочной ситуации: common_sql_req возвращает None."""
    with patch.object(data_manager, 'common_sql_req', return_value=None) as mock_sql:
        with pytest.raises(ValueError) as exc_info:
            data_manager.get_candles_last_dt(
                ticker="GOOGL",
                time_step_resolution="15min"
            )
        
        # Проверяем сообщение об ошибке
        assert "No candle data found for ticker GOOGL with resolution 15min" in str(exc_info.value)
        
        # Проверяем вызов SQL
        expected_sql = """
        SELECT
            max(date) AS ans
        FROM candles.googl_15min
        """
        mock_sql.assert_called_once_with(expected_sql)


def test_get_candles_last_dt_default_parameters(data_manager):
    """Тест с параметрами по умолчанию."""
    mock_result = pd.DataFrame({
        "ans": ["2024-01-15 10:00:00"]
    })
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        result = data_manager.get_candles_last_dt()
        
        # Проверяем вызов SQL с параметрами по умолчанию
        expected_sql = """
        SELECT
            max(date) AS ans
        FROM candles.sber_hour
        """
        mock_sql.assert_called_once_with(expected_sql)
        
        # Проверяем результат
        expected_dt = datetime.datetime.strptime(
            "2024-01-15 10:00:00", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        assert result == expected_dt


def test_get_candles_last_dt_different_resolution(data_manager):
    """Тест с разными разрешениями времени."""
    mock_result = pd.DataFrame({
        "ans": ["2024-01-15 09:45:00"]
    })
    
    with patch.object(data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        result = data_manager.get_candles_last_dt(
            ticker="TSLA",
            time_step_resolution="15min"
        )
        
        # Проверяем вызов SQL с другим разрешением
        expected_sql = """
        SELECT
            max(date) AS ans
        FROM candles.tsla_15min
        """
        mock_sql.assert_called_once_with(expected_sql)
        
        # Проверяем результат
        expected_dt = datetime.datetime.strptime(
            "2024-01-15 09:45:00", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        assert result == expected_dt
