import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, call

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    """Фикстура для создания экземпляра DataManager."""
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_create_tickers_scope_happy_path(data_manager):
    """Тест нормального поведения: создание таблицы scope с тикерами."""
    mock_table_names = ["candles_aapl_hour", "candles_googl_hour", "candles_tsla_hour"]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_schema') as mock_get_tables:
        mock_get_tables.return_value = mock_table_names
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            with patch('builtins.print') as mock_print:
                result = data_manager.create_tickers_scope(con=mock_con)
                
                mock_get_tables.assert_called_once_with("candles", con=mock_con)
                
                expected_tickers = ['aapl', 'googl', 'tsla']
                pd.testing.assert_frame_equal(
                    result,
                    pd.DataFrame(expected_tickers, columns=['ticker'])
                )
                
                mock_to_sql.assert_called_once_with(
                    con=mock_con,
                    schema='tickers',
                    name='scope',
                    if_exists='replace',
                    index=False
                )
                
                mock_print.assert_called_once_with("tickers scope created")


def test_create_tickers_scope_empty_tables(data_manager):
    """Тест граничного случая: нет таблиц в схеме candles."""
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_schema') as mock_get_tables:
        mock_get_tables.return_value = []
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            with patch('builtins.print') as mock_print:
                result = data_manager.create_tickers_scope(con=mock_con)
                
                mock_get_tables.assert_called_once_with("candles", con=mock_con)
                
                pd.testing.assert_frame_equal(
                    result,
                    pd.DataFrame([], columns=['ticker'])
                )
                
                mock_to_sql.assert_called_once()
                mock_print.assert_called_once_with("tickers scope created")


def test_create_tickers_scope_duplicate_tickers(data_manager):
    """Тест обработки дублирующихся тикеров (должны быть удалены через set)."""
    mock_table_names = [
        "candles_aapl_hour",
        "candles_googl_hour", 
        "candles_aapl_hour",
        "candles_tsla_hour",
        "candles_googl_hour"
    ]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_schema') as mock_get_tables:
        mock_get_tables.return_value = mock_table_names
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            with patch('builtins.print') as mock_print:
                result = data_manager.create_tickers_scope(con=mock_con)
                
                expected_tickers = {'aapl', 'googl', 'tsla'}
                actual_tickers = set(result['ticker'].tolist())
                assert actual_tickers == expected_tickers
                assert len(result) == 3
                
                mock_to_sql.assert_called_once()
                mock_print.assert_called_once_with("tickers scope created")


def test_create_tickers_scope_different_table_formats(data_manager):
    """Тест обработки таблиц с разными форматами имен."""
    mock_table_names = [
        "candles_aapl_hour",
        "candles_googl_15min",
        "candles_tsla_hour",
        "analytics_aapl",
        "some_other_table"
    ]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_schema') as mock_get_tables:
        mock_get_tables.return_value = mock_table_names
        
        with patch('pandas.DataFrame.to_sql') as mock_to_sql:
            with patch('builtins.print') as mock_print:
                result = data_manager.create_tickers_scope(con=mock_con)
                
                actual_tickers = set(result['ticker'].tolist())
                assert 'aapl' in actual_tickers
                assert 'tsla' in actual_tickers
                assert 'googl' not in actual_tickers
                assert len(result) == 2
                
                mock_to_sql.assert_called_once()
                mock_print.assert_called_once_with("tickers scope created")


def test_create_tickers_scope_exception_in_to_sql(data_manager):
    """Тест обработки исключения при сохранении в БД."""
    mock_table_names = ["candles_aapl_hour"]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_schema') as mock_get_tables:
        mock_get_tables.return_value = mock_table_names
        
        with patch('pandas.DataFrame.to_sql', side_effect=Exception("DB error")) as mock_to_sql:
            with patch('builtins.print') as mock_print:
                result = data_manager.create_tickers_scope(con=mock_con)
                
                mock_to_sql.assert_called_once()
                mock_print.assert_called_once_with("tickers scope created")
                
                assert isinstance(result, pd.DataFrame)
                assert list(result.columns) == ['ticker']
                assert result['ticker'].tolist() == ['aapl']
