import datetime
from unittest.mock import MagicMock, patch, call
import pytest
import pandas as pd
import sqlalchemy


@pytest.fixture
def mock_trading_manager():
    manager = MagicMock()
    manager.get_market_trading_history = MagicMock()
    return manager


@pytest.fixture
def data_manager(mock_trading_manager):
    from src.trader3000.core.data_management import DataManager
    manager = DataManager(trading_manager=mock_trading_manager)
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_training_datasets_download_happy_path(data_manager):
    tickers = ["AAPL", "GOOGL"]
    time_step_resolutions = ["1min", "1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    mock_data = pd.DataFrame({
        "date": [datetime.datetime(2023, 1, 1, 10, 0)],
        "open": [150.0],
        "high": [152.0],
        "low": [149.0],
        "close": [151.0],
        "volume": [1000]
    })
    
    data_manager.trading_manager.get_market_trading_history.return_value = mock_data
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch.object(data_manager, 'del_ticker_from_scope') as mock_del_ticker, \
         patch.object(data_manager, 'drop_tables_by_ticker') as mock_drop_tables, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('time.sleep') as mock_sleep, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        mock_get_last_update_date.return_value = datetime.datetime(2023, 1, 1)
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    expected_calls = [
        call(
            ticker="AAPL",
            start_date=start_date,
            finish_date=finish_date,
            time_step_resolution="1min"
        ),
        call(
            ticker="AAPL",
            start_date=start_date,
            finish_date=finish_date,
            time_step_resolution="1h"
        ),
        call(
            ticker="GOOGL",
            start_date=start_date,
            finish_date=finish_date,
            time_step_resolution="1min"
        ),
        call(
            ticker="GOOGL",
            start_date=start_date,
            finish_date=finish_date,
            time_step_resolution="1h"
        )
    ]
    data_manager.trading_manager.get_market_trading_history.assert_has_calls(expected_calls, any_order=True)
    
    assert mock_to_sql.call_count == 4
    mock_del_ticker.assert_not_called()
    mock_drop_tables.assert_not_called()
    mock_con.commit.assert_called()
    mock_con.close.assert_called()


def test_training_datasets_download_no_data_delete_old(data_manager):
    tickers = ["OBSOLETE"]
    time_step_resolutions = ["1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    data_manager.trading_manager.get_market_trading_history.return_value = None
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch.object(data_manager, 'del_ticker_from_scope') as mock_del_ticker, \
         patch.object(data_manager, 'drop_tables_by_ticker') as mock_drop_tables, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        old_date = datetime.datetime.now() - datetime.timedelta(days=35)
        mock_get_last_update_date.return_value = old_date
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    data_manager.trading_manager.get_market_trading_history.assert_called_once_with(
        ticker="OBSOLETE",
        start_date=start_date,
        finish_date=finish_date,
        time_step_resolution="1h"
    )
    
    mock_del_ticker.assert_called_once_with("OBSOLETE")
    mock_drop_tables.assert_called_once_with("obsolete")
    mock_to_sql.assert_not_called()


def test_training_datasets_download_no_data_recent_update(data_manager):
    tickers = ["RECENT"]
    time_step_resolutions = ["1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    data_manager.trading_manager.get_market_trading_history.return_value = None
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch.object(data_manager, 'del_ticker_from_scope') as mock_del_ticker, \
         patch.object(data_manager, 'drop_tables_by_ticker') as mock_drop_tables, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        recent_date = datetime.datetime.now() - datetime.timedelta(days=15)
        mock_get_last_update_date.return_value = recent_date
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    data_manager.trading_manager.get_market_trading_history.assert_called_once_with(
        ticker="RECENT",
        start_date=start_date,
        finish_date=finish_date,
        time_step_resolution="1h"
    )
    
    mock_del_ticker.assert_not_called()
    mock_drop_tables.assert_not_called()
    mock_to_sql.assert_not_called()


def test_training_datasets_download_empty_dataframe(data_manager):
    tickers = ["EMPTY"]
    time_step_resolutions = ["1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    data_manager.trading_manager.get_market_trading_history.return_value = pd.DataFrame()
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch.object(data_manager, 'del_ticker_from_scope') as mock_del_ticker, \
         patch.object(data_manager, 'drop_tables_by_ticker') as mock_drop_tables, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        mock_get_last_update_date.return_value = None
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    data_manager.trading_manager.get_market_trading_history.assert_called_once()
    mock_del_ticker.assert_called_once_with("EMPTY")
    mock_drop_tables.assert_called_once_with("empty")
    mock_to_sql.assert_not_called()


def test_training_datasets_download_with_sleep(data_manager):
    tickers = ["AAPL", "GOOGL", "MSFT"]
    time_step_resolutions = ["1min", "1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    mock_data = pd.DataFrame({
        "date": [datetime.datetime(2023, 1, 1, 10, 0)],
        "open": [150.0],
        "high": [152.0],
        "low": [149.0],
        "close": [151.0],
        "volume": [1000]
    })
    
    data_manager.trading_manager.get_market_trading_history.return_value = mock_data
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch('time.sleep') as mock_sleep, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        mock_get_last_update_date.return_value = datetime.datetime(2023, 1, 1)
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    assert mock_sleep.call_count == 6
    mock_sleep.assert_has_calls([call(60)] * 6)


def test_training_datasets_download_exception_in_get_last_update_date(data_manager):
    tickers = ["ERROR"]
    time_step_resolutions = ["1h"]
    start_date = datetime.datetime(2023, 1, 1)
    finish_date = datetime.datetime(2023, 1, 2)
    save_mode = "replace"
    
    data_manager.trading_manager.get_market_trading_history.return_value = None
    
    with patch.object(data_manager, 'get_last_update_date') as mock_get_last_update_date, \
         patch.object(data_manager, 'del_ticker_from_scope') as mock_del_ticker, \
         patch.object(data_manager, 'drop_tables_by_ticker') as mock_drop_tables, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql, \
         patch('sqlalchemy.create_engine') as mock_create_engine:
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_engine.connect.return_value = mock_con
        mock_create_engine.return_value = mock_engine
        
        mock_get_last_update_date.side_effect = Exception("DB error")
        
        data_manager.training_datasets_download(
            tickers=tickers,
            time_step_resolutions=time_step_resolutions,
            start_date=start_date,
            finish_date=finish_date,
            save_mode=save_mode
        )
    
    data_manager.trading_manager.get_market_trading_history.assert_called_once()
    mock_del_ticker.assert_called_once_with("ERROR")
    mock_drop_tables.assert_called_once_with("error")
    mock_to_sql.assert_not_called()
