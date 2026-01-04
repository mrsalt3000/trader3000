import datetime
from unittest.mock import MagicMock, patch, call
import pytest
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_trading_manager():
    manager = MagicMock()
    manager.get_market_trading_history.return_value = pd.DataFrame({
        'date': [datetime.datetime(2024, 1, 2, 10, 0, 0)],
        'open': [100.0],
        'high': [105.0],
        'low': [99.0],
        'close': [102.0],
        'volume': [1000]
    })
    return manager


@pytest.fixture
def data_manager(mock_trading_manager):
    dm = DataManager(trading_manager=mock_trading_manager)
    dm.psql_main_creds = "postgresql://test:test@localhost/test"
    return dm


def test_download_candles_delta_happy_path(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download:
        
        mock_get_last_dt.return_value = datetime.datetime(2024, 1, 1, 10, 0, 0, 
                                                          tzinfo=datetime.timezone.utc)
        
        data_manager.download_candles_delta(
            tickers=["SBER", "GAZP"],
            time_step_resolutions=["hour", "15min"]
        )
        
        assert mock_get_last_dt.call_count == 2
        expected_calls = [
            call("SBER", "hour"),
            call("GAZP", "hour"),
        ]
        mock_get_last_dt.assert_has_calls(expected_calls, any_order=False)
        
        assert mock_training_download.call_count == 2
        
        first_call = mock_training_download.call_args_list[0]
        assert first_call[1]['tickers'] == ["SBER"]
        assert first_call[1]['time_step_resolutions'] == ["hour"]
        assert first_call[1]['save_mode'] == "append"
        
        second_call = mock_training_download.call_args_list[1]
        assert second_call[1]['tickers'] == ["GAZP"]
        assert second_call[1]['time_step_resolutions'] == ["hour"]


def test_download_candles_delta_resource_exhausted_retry(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download, \
         patch('time.sleep') as mock_sleep:
        
        mock_get_last_dt.side_effect = [
            datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
        ]
        
        mock_training_download.side_effect = [
            Exception("RESOURCE_EXHAUSTED: Quota exceeded"),
            None
        ]
        
        data_manager.download_candles_delta(
            tickers=["SBER"],
            time_step_resolutions=["hour", "15min"]
        )
        
        mock_sleep.assert_called_once_with(30)
        assert mock_training_download.call_count == 2
        assert mock_get_last_dt.call_count == 2


def test_download_candles_delta_other_exception_propagates(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download:
        
        mock_get_last_dt.return_value = datetime.datetime(2024, 1, 1, 10, 0, 0,
                                                          tzinfo=datetime.timezone.utc)
        
        test_exception = ValueError("Some other error")
        mock_training_download.side_effect = test_exception
        
        with pytest.raises(ValueError, match="Some other error"):
            data_manager.download_candles_delta(
                tickers=["SBER"],
                time_step_resolutions=["hour"]
            )
        
        assert mock_training_download.call_count == 1


def test_download_candles_delta_empty_lists(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download:
        
        data_manager.download_candles_delta(
            tickers=[],
            time_step_resolutions=[]
        )
        
        mock_get_last_dt.assert_not_called()
        mock_training_download.assert_not_called()


def test_download_candles_delta_time_delta_calculation(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download:
        
        last_dt = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_get_last_dt.return_value = last_dt
        
        data_manager.download_candles_delta(
            tickers=["SBER"],
            time_step_resolutions=["hour", "15min"]
        )
        
        assert mock_training_download.call_count == 1
        
        call_kwargs = mock_training_download.call_args[1]
        assert call_kwargs['start_date'] == last_dt + datetime.timedelta(hours=1)
        assert call_kwargs['tickers'] == ["SBER"]
        assert call_kwargs['time_step_resolutions'] == ["hour"]


def test_download_candles_delta_default_parameters(data_manager):
    with patch.object(data_manager, 'get_candles_last_dt') as mock_get_last_dt, \
         patch.object(data_manager, 'training_datasets_download') as mock_training_download:
        
        mock_get_last_dt.return_value = datetime.datetime(2024, 1, 1, 10, 0, 0,
                                                          tzinfo=datetime.timezone.utc)
        
        data_manager.download_candles_delta()
        
        mock_get_last_dt.assert_called_once_with("SBER", "hour")
        mock_training_download.assert_called_once()
        
        call_kwargs = mock_training_download.call_args[1]
        assert call_kwargs['tickers'] == ["SBER"]
        assert call_kwargs['time_step_resolutions'] == ["hour"]
        assert call_kwargs['save_mode'] == "append"
        assert call_kwargs['finish_date'].tzinfo == datetime.timezone.utc
