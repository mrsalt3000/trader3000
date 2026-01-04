import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import datetime

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания DataManager с замоканными зависимостями."""
    manager = DataManager(trading_manager=None)
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_transactions_happy_path(mock_data_manager):
    """Тест нормального поведения: возвращает DataFrame с транзакциями."""
    # Подготавливаем тестовые данные
    test_data = pd.DataFrame({
        'order_id': ['order1', 'order2', 'order3'],
        'buy_dt': [
            datetime.datetime(2023, 1, 3),
            datetime.datetime(2023, 1, 2),
            datetime.datetime(2023, 1, 1)
        ],
        'status': ['filled', 'pending', 'filled'],
        'direction': ['BUY', 'SELL', 'BUY']
    })
    
    # Мокаем common_sql_req для возврата тестовых данных
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        # Вызываем тестируемый метод
        result = mock_data_manager.get_transactions()
        
        # Проверяем вызов SQL-запроса
        expected_sql = """
            SELECT
                *
            FROM transactions
            ORDER BY buy_dt DESC 
            """
        mock_sql_req.assert_called_once_with(expected_sql)
        
        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['order_id', 'buy_dt', 'status', 'direction']
        # Проверяем сортировку по убыванию даты
        assert result['buy_dt'].iloc[0] == datetime.datetime(2023, 1, 3)
        assert result['buy_dt'].iloc[2] == datetime.datetime(2023, 1, 1)


def test_get_transactions_empty_result(mock_data_manager):
    """Тест граничного случая: таблица транзакций пуста."""
    # Мокаем common_sql_req для возврата пустого DataFrame
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = pd.DataFrame()
        
        result = mock_data_manager.get_transactions()
        
        # Проверяем вызов
        expected_sql = """
            SELECT
                *
            FROM transactions
            ORDER BY buy_dt DESC 
            """
        mock_sql_req.assert_called_once_with(expected_sql)
        
        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert result.empty


def test_get_transactions_sql_error(mock_data_manager):
    """Тест ошибочной ситуации: SQL-запрос возвращает None."""
    # Мокаем common_sql_req для возврата None (симуляция ошибки в запросе)
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = mock_data_manager.get_transactions()
        
        # Проверяем вызов
        expected_sql = """
            SELECT
                *
            FROM transactions
            ORDER BY buy_dt DESC 
            """
        mock_sql_req.assert_called_once_with(expected_sql)
        
        # Проверяем результат - должен быть None
        assert result is None


def test_get_transactions_with_mocked_connection():
    """Тест с полным моком декоратора with_connection."""
    # Создаем мок для DataManager
    with patch('src.trader3000.core.data_management.sqlalchemy') as mock_sqlalchemy:
        # Настраиваем моки для соединения
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_sqlalchemy.create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        # Создаем DataManager
        manager = DataManager(trading_manager=None)
        manager.psql_main_creds = "postgresql://test:test@localhost/test"
        
        # Подготавливаем тестовые данные
        test_data = pd.DataFrame({
            'order_id': ['test_order'],
            'buy_dt': [datetime.datetime(2023, 1, 1)]
        })
        
        # Мокаем pd.read_sql внутри common_sql_req
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = test_data
            
            # Вызываем метод
            result = manager.get_transactions()
            
            # Проверяем создание соединения
            mock_sqlalchemy.create_engine.assert_called_once_with(manager.psql_main_creds)
            
            # Проверяем, что read_sql был вызван с правильным SQL
            # (common_sql_req вызывает pd.read_sql внутри)
            assert mock_read_sql.called
            sql_arg = mock_read_sql.call_args[0][0]
            assert "SELECT" in sql_arg
            assert "FROM transactions" in sql_arg
            assert "ORDER BY buy_dt DESC" in sql_arg
            
            # Проверяем результат
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert result['order_id'].iloc[0] == 'test_order'


def test_get_transactions_with_specific_columns(mock_data_manager):
    """Тест проверяет, что возвращаются все колонки из таблицы."""
    # Создаем тестовые данные с разными колонками
    test_data = pd.DataFrame({
        'id': [1, 2],
        'order_id': ['ord1', 'ord2'],
        'buy_dt': [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)],
        'sell_dt': [datetime.datetime(2023, 1, 10), None],
        'price': [100.5, 102.3],
        'quantity': [10, 20],
        'status': ['filled', 'pending']
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        result = mock_data_manager.get_transactions()
        
        # Проверяем, что все колонки присутствуют
        expected_columns = ['id', 'order_id', 'buy_dt', 'sell_dt', 'price', 'quantity', 'status']
        assert list(result.columns) == expected_columns
        assert len(result) == 2
