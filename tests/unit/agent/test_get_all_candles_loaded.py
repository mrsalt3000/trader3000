import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания экземпляра DataManager с моком trading_manager."""
    manager = DataManager()
    manager.trading_manager = MagicMock()
    return manager


def test_get_all_candles_loaded_happy_path(mock_data_manager):
    """Тест нормального поведения: возвращает список тикеров из таблиц."""
    # Мокаем common_sql_req для возврата DataFrame с данными
    mock_result = pd.DataFrame({
        'name': ['AAPL', 'GOOGL', 'MSFT']
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result) as mock_sql:
        result = mock_data_manager.get_all_candles_loaded()
        
        # Проверяем вызов SQL-запроса
        mock_sql.assert_called_once_with("""
        SELECT DISTINCT
            replace(replace(replace(table_name,'candles_',''),'_hour',''),'_15min','') AS name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
            AND table_schema IN ('candles','analytics')
        """)
        
        # Проверяем результат: список тикеров
        assert isinstance(result, list)
        assert len(result) == 3
        assert 'AAPL' in result
        assert 'GOOGL' in result
        assert 'MSFT' in result
        assert result == ['AAPL', 'GOOGL', 'MSFT']


def test_get_all_candles_loaded_empty_result(mock_data_manager):
    """Тест граничного случая: нет таблиц в схемах candles и analytics."""
    # Мокаем common_sql_req для возврата пустого DataFrame
    mock_result = pd.DataFrame(columns=['name'])
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result):
        result = mock_data_manager.get_all_candles_loaded()
        
        assert isinstance(result, list)
        assert len(result) == 0
        assert result == []


def test_get_all_candles_loaded_none_result(mock_data_manager):
    """Тест ошибочной ситуации: common_sql_req возвращает None."""
    # Мокаем common_sql_req для возврата None
    with patch.object(mock_data_manager, 'common_sql_req', return_value=None):
        result = mock_data_manager.get_all_candles_loaded()
        
        assert isinstance(result, list)
        assert len(result) == 0
        assert result == []


def test_get_all_candles_loaded_with_processed_names(mock_data_manager):
    """Тест обработки уже обработанных имен таблиц SQL-запросом."""
    # Мокаем common_sql_req для возврата DataFrame с обработанными именами
    mock_result = pd.DataFrame({
        'name': [
            'aapl',
            'googl',
            'msft',
            'other_tsla_daily',
            'nvda_hour_extra',
        ]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result):
        result = mock_data_manager.get_all_candles_loaded()
        
        assert isinstance(result, list)
        assert len(result) == 5
        assert 'aapl' in result
        assert 'googl' in result
        assert 'msft' in result
        assert 'other_tsla_daily' in result
        assert 'nvda_hour_extra' in result


def test_get_all_candles_loaded_with_duplicate_values(mock_data_manager):
    """Тест обработки дубликатов значений из SQL-запроса."""
    # Мокаем common_sql_req для возврата DataFrame с дубликатами
    mock_result = pd.DataFrame({
        'name': ['AAPL', 'GOOGL', 'AAPL', 'MSFT', 'GOOGL', 'AAPL']
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_result):
        result = mock_data_manager.get_all_candles_loaded()
        
        # tolist() возвращает все значения как есть, включая дубликаты
        assert isinstance(result, list)
        assert len(result) == 6
        assert result == ['AAPL', 'GOOGL', 'AAPL', 'MSFT', 'GOOGL', 'AAPL']
