import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    """Фикстура для создания экземпляра DataManager."""
    return DataManager()


def test_get_tickers_from_scope_table_exists(data_manager):
    """Тест нормального поведения: таблица существует и возвращает список тикеров."""
    # Мокируем check_if_table_exists чтобы вернуть True
    with patch.object(data_manager, 'check_if_table_exists') as mock_check:
        mock_check.return_value = True
        
        # Мокируем common_sql_req чтобы вернуть DataFrame с тикерами
        mock_df = pd.DataFrame({'ticker': ['AAPL', 'GOOGL', 'MSFT']})
        with patch.object(data_manager, 'common_sql_req') as mock_sql:
            mock_sql.return_value = mock_df
            
            # Вызываем тестируемый метод
            result = data_manager.get_tickers_from_scope()
            
            # Проверяем вызовы
            mock_check.assert_called_once_with("scope")
            mock_sql.assert_called_once_with(
                """
            SELECT
                ticker
            FROM tickers.scope 
            """
            )
            
            # Проверяем результат
            assert result == ['AAPL', 'GOOGL', 'MSFT']


def test_get_tickers_from_scope_table_not_exists(data_manager):
    """Тест граничного случая: таблица не существует, создается и затем возвращается список."""
    # Первый вызов check_if_table_exists возвращает False
    # Второй вызов (после создания) возвращает True
    with patch.object(data_manager, 'check_if_table_exists') as mock_check:
        mock_check.side_effect = [False, True]
        
        # Мокируем create_tickers_scope
        with patch.object(data_manager, 'create_tickers_scope') as mock_create:
            mock_create.return_value = pd.DataFrame({'ticker': ['TSLA', 'NVDA']})
            
            # Мокируем второй вызов common_sql_req
            with patch.object(data_manager, 'common_sql_req') as mock_sql:
                mock_sql.return_value = pd.DataFrame({'ticker': ['TSLA', 'NVDA']})
                
                # Вызываем тестируемый метод
                result = data_manager.get_tickers_from_scope()
                
                # Проверяем вызовы
                assert mock_check.call_count == 2
                mock_check.assert_has_calls([call("scope"), call("scope")])
                mock_create.assert_called_once()
                mock_sql.assert_called_once_with(
                    """
            SELECT
                ticker
            FROM tickers.scope 
            """
                )
                
                # Проверяем результат
                assert result == ['TSLA', 'NVDA']


def test_get_tickers_from_scope_sql_returns_none(data_manager):
    """Тест ошибочной ситуации: SQL-запрос возвращает None."""
    # Мокируем check_if_table_exists чтобы вернуть True
    with patch.object(data_manager, 'check_if_table_exists') as mock_check:
        mock_check.return_value = True
        
        # Мокируем common_sql_req чтобы вернуть None
        with patch.object(data_manager, 'common_sql_req') as mock_sql:
            mock_sql.return_value = None
            
            # Вызываем тестируемый метод
            result = data_manager.get_tickers_from_scope()
            
            # Проверяем вызовы
            mock_check.assert_called_once_with("scope")
            mock_sql.assert_called_once_with(
                """
            SELECT
                ticker
            FROM tickers.scope 
            """
            )
            
            # Проверяем результат - должен вернуться пустой список
            assert result == []
