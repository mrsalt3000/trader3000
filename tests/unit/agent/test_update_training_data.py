import datetime
from unittest.mock import patch, MagicMock, call
import pytest
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания DataManager с замоканными зависимостями."""
    with patch('sqlalchemy.create_engine') as mock_engine, \
         patch('sqlalchemy.engine.Connection') as mock_conn_class:
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value = mock_conn
        mock_conn_class.return_value = mock_conn
        
        manager = DataManager()
        manager.psql_main_creds = "postgresql://test:test@localhost/test"
        manager.trading_manager = MagicMock()
        yield manager


def test_update_training_data_happy_path_lag_gt_2(mock_data_manager):
    """Тест нормального поведения при лаге > 2 дней."""
    # Arrange
    test_tickers = ["AAPL", "GOOGL", "MSFT"]
    mock_random_choice = MagicMock(return_value="AAPL")
    
    # Мок для common_sql_req с результатом даты
    mock_result_df = pd.DataFrame({"max(date)": ["2024-01-01 00:00:00"]})
    
    # Мок для get_all_candles_loaded
    mock_data_manager.get_all_candles_loaded = MagicMock(
        return_value=["aapl", "googl", "msft"]
    )
    
    # Мок для download_candles_delta
    mock_data_manager.download_candles_delta = MagicMock()
    
    with patch("random.choice", mock_random_choice), \
         patch.object(mock_data_manager, "common_sql_req", 
                      return_value=mock_result_df) as mock_sql_req:
        
        # Act
        mock_data_manager.update_training_data(test_tickers)
        
        # Assert
        # Проверяем вызов common_sql_req для получения максимальной даты
        mock_sql_req.assert_called_once_with(
            "SELECT max(date) FROM candles.AAPL_hour"
        )
        
        # Проверяем вызов get_all_candles_loaded
        mock_data_manager.get_all_candles_loaded.assert_called_once()
        
        # Проверяем вызов download_candles_delta с отфильтрованными тикерами
        mock_data_manager.download_candles_delta.assert_called_once_with(
            tickers=["AAPL", "GOOGL", "MSFT"]
        )


def test_update_training_data_lag_le_2(mock_data_manager):
    """Тест граничного случая при лаге <= 2 дней."""
    # Arrange
    test_tickers = ["AAPL", "GOOGL"]
    mock_random_choice = MagicMock(return_value="GOOGL")
    
    # Устанавливаем дату близкую к текущей (лаг <= 2 дней)
    recent_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    mock_result_df = pd.DataFrame({"max(date)": [recent_date]})
    
    mock_data_manager.get_all_candles_loaded = MagicMock(
        return_value=["aapl", "googl"]
    )
    mock_data_manager.download_candles_delta = MagicMock()
    
    with patch("random.choice", mock_random_choice), \
         patch.object(mock_data_manager, "common_sql_req", 
                      return_value=mock_result_df) as mock_sql_req:
        
        # Act
        mock_data_manager.update_training_data(test_tickers)
        
        # Assert
        mock_sql_req.assert_called_once_with(
            "SELECT max(date) FROM candles.GOOGL_hour"
        )
        mock_data_manager.get_all_candles_loaded.assert_called_once()
        # При лаге <= 2 download_candles_delta НЕ должен вызываться
        mock_data_manager.download_candles_delta.assert_not_called()


def test_update_training_data_no_data_returned(mock_data_manager):
    """Тест ошибочной ситуации, когда запрос возвращает None."""
    # Arrange
    test_tickers = ["AAPL"]
    mock_random_choice = MagicMock(return_value="AAPL")
    
    with patch("random.choice", mock_random_choice), \
         patch.object(mock_data_manager, "common_sql_req", 
                      return_value=None) as mock_sql_req:
        
        # Act & Assert
        with pytest.raises(ValueError, match="No data returned for ticker AAPL"):
            mock_data_manager.update_training_data(test_tickers)
        
        mock_sql_req.assert_called_once_with(
            "SELECT max(date) FROM candles.AAPL_hour"
        )


def test_update_training_data_no_max_date_found(mock_data_manager):
    """Тест ошибочной ситуации, когда max(date) равен None."""
    # Arrange
    test_tickers = ["TSLA"]
    mock_random_choice = MagicMock(return_value="TSLA")
    
    mock_result_df = pd.DataFrame({"max(date)": [None]})
    
    with patch("random.choice", mock_random_choice), \
         patch.object(mock_data_manager, "common_sql_req", 
                      return_value=mock_result_df) as mock_sql_req:
        
        # Act & Assert
        with pytest.raises(ValueError, match="No max date found for ticker TSLA"):
            mock_data_manager.update_training_data(test_tickers)
        
        mock_sql_req.assert_called_once_with(
            "SELECT max(date) FROM candles.TSLA_hour"
        )


def test_update_training_data_filters_tickers_correctly(mock_data_manager):
    """Тест корректной фильтрации тикеров по наличию в загруженных свечах."""
    # Arrange
    test_tickers = ["AAPL", "GOOGL", "MSFT", "NONEXISTENT"]
    mock_random_choice = MagicMock(return_value="AAPL")
    
    mock_result_df = pd.DataFrame({"max(date)": ["2023-12-01 00:00:00"]})
    
    # Только 3 из 4 тикеров есть в загруженных свечах
    mock_data_manager.get_all_candles_loaded = MagicMock(
        return_value=["aapl", "googl", "msft"]
    )
    
    mock_data_manager.download_candles_delta = MagicMock()
    
    with patch("random.choice", mock_random_choice), \
         patch.object(mock_data_manager, "common_sql_req", 
                      return_value=mock_result_df):
        
        # Act
        mock_data_manager.update_training_data(test_tickers)
        
        # Assert
        # download_candles_delta должен быть вызван только с тикерами, которые есть в загруженных свечах
        mock_data_manager.download_candles_delta.assert_called_once_with(
            tickers=["AAPL", "GOOGL", "MSFT"]
        )
