import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания экземпляра DataManager с моком trading_manager."""
    manager = DataManager()
    manager.trading_manager = None
    return manager


def test_get_orders_bought_happy_path(mock_data_manager):
    """Тест нормального поведения: возвращаются данные о купленных ордерах."""
    # Подготовка тестовых данных
    test_data = pd.DataFrame({
        'order_id': ['order_1', 'order_2', 'order_3'],
        'estimated_price_change': [0.05, -0.02, 0.10]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        # Вызов тестируемого метода
        result = mock_data_manager.get_orders_bought()
        
        # Проверки
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        assert result[0, 0] == 'order_1'
        assert result[0, 1] == 0.05
        assert result[1, 0] == 'order_2'
        assert result[1, 1] == -0.02
        
        # Проверка SQL-запроса
        expected_sql = """
        SELECT
            order_id,
            estimated_price_change
        FROM transactions
        WHERE execution_report_status IN ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'bought'
        """
        mock_sql_req.assert_called_once_with(expected_sql)


def test_get_orders_bought_empty_result(mock_data_manager):
    """Тест граничного случая: нет купленных ордеров."""
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        # Возвращаем пустой DataFrame
        mock_sql_req.return_value = pd.DataFrame(columns=['order_id', 'estimated_price_change'])
        
        result = mock_data_manager.get_orders_bought()
        
        # Проверяем, что возвращается пустой массив правильной формы
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
        assert len(result) == 0


def test_get_orders_bought_none_result(mock_data_manager):
    """Тест ошибочной ситуации: common_sql_req возвращает None."""
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = mock_data_manager.get_orders_bought()
        
        # Проверяем, что возвращается пустой массив при None
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
        assert len(result) == 0


def test_get_orders_bought_empty_dataframe(mock_data_manager):
    """Тест граничного случая: DataFrame существует, но пустой."""
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        # DataFrame с колонками, но без строк
        mock_sql_req.return_value = pd.DataFrame({'order_id': [], 'estimated_price_change': []})
        
        result = mock_data_manager.get_orders_bought()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 2)
        assert len(result) == 0


def test_get_orders_bought_with_mixed_data_types(mock_data_manager):
    """Тест с различными типами данных в колонках."""
    test_data = pd.DataFrame({
        'order_id': ['order_1', 12345, 'order_3'],  # смешанные типы
        'estimated_price_change': [0.05, '0.02', 0.10]  # строка среди чисел
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        result = mock_data_manager.get_orders_bought()
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        # Проверяем, что значения сохраняются как есть
        assert result[0, 0] == 'order_1'
        assert result[1, 0] == 12345
        assert result[1, 1] == '0.02'
