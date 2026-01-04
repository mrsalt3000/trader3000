import pytest
from unittest.mock import patch, MagicMock
import datetime
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_trading_manager():
    return MagicMock()


@pytest.fixture
def data_manager(mock_trading_manager):
    with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://test'}):
        dm = DataManager(trading_manager=mock_trading_manager)
    return dm


def test_load_all_candles_add_mode_happy_path(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = [
        'AAPL', 'GOOGL', 'TSLA', 'MSFT'
    ]
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = ['aapl', 'googl']
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                data_manager.load_all_candles(mode='add')
                
                mock_trading_manager.get_all_instruments_tickers.assert_called_once()
                mock_get_loaded.assert_called_once()
                
                expected_tickers = ['TSLA', 'MSFT']
                mock_download.assert_called_once_with(
                    tickers=expected_tickers,
                    time_step_resolutions=['hour'],
                    start_date=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
                    finish_date=datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc),
                    save_mode='replace'
                )


def test_load_all_candles_reload_mode(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = [
        'AAPL', 'GOOGL', 'TSLA'
    ]
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = ['aapl']
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                data_manager.load_all_candles(mode='reload')
                
                mock_trading_manager.get_all_instruments_tickers.assert_called_once()
                mock_get_loaded.assert_called_once()
                
                expected_tickers = ['AAPL', 'GOOGL', 'TSLA']
                mock_download.assert_called_once_with(
                    tickers=expected_tickers,
                    time_step_resolutions=['hour'],
                    start_date=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
                    finish_date=datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc),
                    save_mode='replace'
                )


def test_load_all_candles_no_new_tickers(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = [
        'AAPL', 'GOOGL'
    ]
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = ['aapl', 'googl']
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                data_manager.load_all_candles(mode='add')
                
                mock_trading_manager.get_all_instruments_tickers.assert_called_once()
                mock_get_loaded.assert_called_once()
                
                mock_download.assert_called_once_with(
                    tickers=[],
                    time_step_resolutions=['hour'],
                    start_date=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
                    finish_date=datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc),
                    save_mode='replace'
                )


def test_load_all_candles_invalid_mode_raises_unbound_local_error(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = ['AAPL', 'GOOGL']
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = ['googl']
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                with pytest.raises(UnboundLocalError):
                    data_manager.load_all_candles(mode='invalid_mode')


def test_load_all_candles_empty_instruments(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = []
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = []
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                data_manager.load_all_candles(mode='add')
                
                mock_trading_manager.get_all_instruments_tickers.assert_called_once()
                mock_get_loaded.assert_called_once()
                
                mock_download.assert_called_once_with(
                    tickers=[],
                    time_step_resolutions=['hour'],
                    start_date=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
                    finish_date=datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc),
                    save_mode='replace'
                )


def test_load_all_candles_exception_in_get_all_candles_loaded(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = ['AAPL', 'GOOGL']
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.side_effect = Exception("Database error")
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                with pytest.raises(Exception, match="Database error"):
                    data_manager.load_all_candles(mode='add')
                
                mock_get_loaded.assert_called_once()
                mock_download.assert_not_called()


def test_load_all_candles_all_tickers_already_loaded_in_reload_mode(data_manager, mock_trading_manager):
    mock_trading_manager.get_all_instruments_tickers.return_value = ['AAPL', 'GOOGL']
    
    with patch.object(data_manager, 'get_all_candles_loaded') as mock_get_loaded:
        mock_get_loaded.return_value = ['aapl', 'googl']
        
        with patch.object(data_manager, 'training_datasets_download') as mock_download:
            with patch('sqlalchemy.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_connection = MagicMock()
                mock_engine.connect.return_value = mock_connection
                mock_create_engine.return_value = mock_engine
                
                data_manager.load_all_candles(mode='reload')
                
                mock_trading_manager.get_all_instruments_tickers.assert_called_once()
                mock_get_loaded.assert_called_once()
                
                mock_download.assert_called_once_with(
                    tickers=['AAPL', 'GOOGL'],
                    time_step_resolutions=['hour'],
                    start_date=datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc),
                    finish_date=datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc),
                    save_mode='replace'
                )
