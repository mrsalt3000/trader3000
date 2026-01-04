import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания DataManager с мокнутым trading_manager."""
    manager = DataManager(trading_manager=MagicMock())
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_orders_sell_order_posted_happy_path(mock_data_manager):
    """Тест нормального поведения: возвращается массив с sell_order_id."""
    # Подготавливаем мок для common_sql_req
    mock_result = pd.DataFrame({
        "sell_order_id": ["order_123", "order_456", "order_789"]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        result = mock_data_manager.get_orders_sell_order_posted()
        
        # Проверяем, что запрос был вызван с правильным SQL
        expected_sql = """
        SELECT
            sell_order_id
        FROM transactions 
        WHERE status = 'sell_order_posted'
            AND direction = 'ORDER_DIRECTION_BUY'
        """
        mock_sql.assert_called_once()
        call_args = mock_sql.call_args[0][0]
        # Нормализуем пробелы для сравнения
        assert call_args.strip().replace(" ", "").replace("\n", "") == \
               expected_sql.strip().replace(" ", "").replace("\n", "")
        
        # Проверяем результат
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        assert result[0][0] == "order_123"
        assert result[1][0] == "order_456"
        assert result[2][0] == "order_789"


def test_get_orders_sell_order_posted_empty_result(mock_data_manager):
    """Тест граничного случая: пустой результат запроса."""
    # Мок возвращает пустой DataFrame
    mock_result = pd.DataFrame(columns=["sell_order_id"])
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result):
        result = mock_data_manager.get_orders_sell_order_posted()
        
        # Проверяем, что возвращается пустой массив правильной формы
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 1)


def test_get_orders_sell_order_posted_none_result(mock_data_manager):
    """Тест ошибочной ситуации: common_sql_req возвращает None."""
    # Мок возвращает None (например, при ошибке выполнения запроса)
    with patch.object(mock_data_manager, 'common_sql_req', return_value=None):
        result = mock_data_manager.get_orders_sell_order_posted()
        
        # Проверяем, что возвращается пустой массив правильной формы
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 1)


def test_get_orders_sell_order_posted_with_specific_columns(mock_data_manager):
    """Тест с проверкой структуры возвращаемых данных."""
    # Создаем DataFrame с разными типами данных в sell_order_id
    mock_result = pd.DataFrame({
        "sell_order_id": [1001, 1002, 1003]  # числовые ID
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result):
        result = mock_data_manager.get_orders_sell_order_posted()
        
        # Проверяем структуру и типы данных
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        # Проверяем, что числовые значения сохранились
        assert result[0][0] == 1001
        assert result[1][0] == 1002
        assert result[2][0] == 1003


def test_get_orders_sell_order_posted_sql_injection_prevention(mock_data_manager):
    """Тест, что SQL-запрос формируется статически, без конкатенации пользовательских данных."""
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql:
        mock_data_manager.get_orders_sell_order_posted()
        
        # Получаем SQL-запрос, который был передан
        sql_query = mock_sql.call_args[0][0]
        
        # Проверяем, что запрос не содержит динамических частей
        # (в данном методе их не должно быть)
        assert "'sell_order_posted'" in sql_query or '"sell_order_posted"' in sql_query
        assert "'ORDER_DIRECTION_BUY'" in sql_query or '"ORDER_DIRECTION_BUY"' in sql_query
        
        # Проверяем, что нет подозрительных конструкций
        assert "%" not in sql_query
        assert ";" not in sql_query or sql_query.count(";") == 1  # только завершающая точка с запятой
