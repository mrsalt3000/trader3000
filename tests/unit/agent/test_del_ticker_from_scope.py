import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_del_ticker_from_scope_happy_path(data_manager):
    ticker_to_delete = "AAPL"
    existing_tickers = ["AAPL", "GOOGL", "MSFT"]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers:
        mock_get_tickers.return_value = existing_tickers
        
        data_manager.del_ticker_from_scope(ticker_to_delete, con=mock_con)
        
        mock_get_tickers.assert_called_once()
        
        call_args = mock_con.to_sql.call_args
        assert call_args is not None
        
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert list(df_arg['ticker']) == ["GOOGL", "MSFT"]
        
        assert call_args[1]['schema'] == 'tickers'
        assert call_args[1]['name'] == 'scope'
        assert call_args[1]['if_exists'] == 'replace'
        assert call_args[1]['index'] is False


def test_del_ticker_from_scope_ticker_not_found(data_manager):
    ticker_to_delete = "TSLA"
    existing_tickers = ["AAPL", "GOOGL", "MSFT"]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers:
        mock_get_tickers.return_value = existing_tickers
        
        data_manager.del_ticker_from_scope(ticker_to_delete, con=mock_con)
        
        mock_get_tickers.assert_called_once()
        mock_con.to_sql.assert_not_called()


def test_del_ticker_from_scope_last_ticker(data_manager):
    ticker_to_delete = "AAPL"
    existing_tickers = ["AAPL"]
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers:
        mock_get_tickers.return_value = existing_tickers
        
        data_manager.del_ticker_from_scope(ticker_to_delete, con=mock_con)
        
        mock_get_tickers.assert_called_once()
        
        call_args = mock_con.to_sql.call_args
        assert call_args is not None
        
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 0
        assert list(df_arg.columns) == ['ticker']


def test_del_ticker_from_scope_with_connection_decorator(data_manager):
    ticker_to_delete = "AAPL"
    existing_tickers = ["AAPL", "GOOGL"]
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers, \
         patch('sqlalchemy.create_engine') as mock_create_engine, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tickers.return_value = existing_tickers
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        data_manager.del_ticker_from_scope(ticker_to_delete)
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_get_tickers.assert_called_once()
        mock_to_sql.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


def test_del_ticker_from_scope_empty_tickers_list(data_manager):
    ticker_to_delete = "AAPL"
    
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        data_manager.del_ticker_from_scope(ticker_to_delete, con=mock_con)
        
        mock_get_tickers.assert_called_once()
        mock_con.to_sql.assert_not_called()


def test_del_ticker_from_scope_exception_in_decorator(data_manager):
    ticker_to_delete = "AAPL"
    existing_tickers = ["AAPL", "GOOGL"]
    
    with patch.object(data_manager, 'get_tickers_from_scope') as mock_get_tickers, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tickers.return_value = existing_tickers
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        mock_conn.to_sql.side_effect = Exception("DB error")
        
        with pytest.raises(Exception, match="DB error"):
            data_manager.del_ticker_from_scope(ticker_to_delete)
        
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()
