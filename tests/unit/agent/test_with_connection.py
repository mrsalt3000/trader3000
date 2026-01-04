import pytest
from unittest.mock import MagicMock, patch, call
import datetime
import pandas as pd
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError

from src.trader3000.core.data_management import with_connection, DataManager


@pytest.fixture
def mock_engine():
    return MagicMock()


@pytest.fixture
def mock_connection():
    conn = MagicMock()
    conn.commit = MagicMock()
    conn.rollback = MagicMock()
    conn.close = MagicMock()
    return conn


@pytest.fixture
def mock_data_manager():
    dm = MagicMock(spec=DataManager)
    dm.psql_main_creds = "postgresql://user:pass@localhost/db"
    return dm


def test_with_connection_happy_path(mock_data_manager, mock_engine, mock_connection):
    """Тест нормального выполнения функции с успешным коммитом."""
    test_result = "test_result"
    
    @with_connection
    def sample_method(self, arg1, con=None):
        assert con is not None
        assert arg1 == "test_arg"
        return test_result
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine) as mock_create_engine:
        mock_engine.connect.return_value = mock_connection
        
        result = sample_method(mock_data_manager, "test_arg")
        
        mock_create_engine.assert_called_once_with(mock_data_manager.psql_main_creds)
        mock_engine.connect.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_connection.close.assert_called_once()
        assert result == test_result


def test_with_connection_exception_rollback(mock_data_manager, mock_engine, mock_connection):
    """Тест обработки исключения с откатом транзакции."""
    test_exception = ValueError("Test error")
    
    @with_connection
    def failing_method(self, con=None):
        raise test_exception
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine):
        mock_engine.connect.return_value = mock_connection
        
        with pytest.raises(ValueError) as exc_info:
            failing_method(mock_data_manager)
        
        assert exc_info.value == test_exception
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()
        mock_connection.close.assert_called_once()


def test_with_connection_sqlalchemy_error(mock_data_manager, mock_engine, mock_connection):
    """Тест обработки SQLAlchemyError при создании соединения."""
    sql_error = SQLAlchemyError("Connection failed")
    
    @with_connection
    def sample_method(self, con=None):
        return "should_not_reach_here"
    
    with patch('sqlalchemy.create_engine', side_effect=sql_error):
        with pytest.raises(SQLAlchemyError) as exc_info:
            sample_method(mock_data_manager)
        
        assert exc_info.value == sql_error


def test_with_connection_connection_close_on_success(mock_data_manager, mock_engine, mock_connection):
    """Тест гарантированного закрытия соединения даже при успешном выполнении."""
    call_count = {"close_calls": 0}
    
    original_close = mock_connection.close
    def counting_close():
        call_count["close_calls"] += 1
        original_close()
    
    mock_connection.close = counting_close
    
    @with_connection
    def sample_method(self, con=None):
        return "result"
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine):
        mock_engine.connect.return_value = mock_connection
        
        result = sample_method(mock_data_manager)
        
        assert result == "result"
        assert call_count["close_calls"] == 1
        mock_connection.commit.assert_called_once()


def test_with_connection_connection_close_on_exception(mock_data_manager, mock_engine, mock_connection):
    """Тест гарантированного закрытия соединения даже при исключении."""
    call_count = {"close_calls": 0}
    
    original_close = mock_connection.close
    def counting_close():
        call_count["close_calls"] += 1
        original_close()
    
    mock_connection.close = counting_close
    
    @with_connection
    def failing_method(self, con=None):
        raise RuntimeError("Operation failed")
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine):
        mock_engine.connect.return_value = mock_connection
        
        with pytest.raises(RuntimeError):
            failing_method(mock_data_manager)
        
        assert call_count["close_calls"] == 1
        mock_connection.rollback.assert_called_once()
        mock_connection.commit.assert_not_called()


def test_with_connection_multiple_args_kwargs(mock_data_manager, mock_engine, mock_connection):
    """Тест передачи множества аргументов и ключевых слов в оборачиваемую функцию."""
    @with_connection
    def complex_method(self, arg1, arg2, kwarg1=None, kwarg2=None, con=None):
        assert con is not None
        return f"{arg1}_{arg2}_{kwarg1}_{kwarg2}"
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine):
        mock_engine.connect.return_value = mock_connection
        
        result = complex_method(
            mock_data_manager, 
            "arg1_val", 
            "arg2_val", 
            kwarg1="kw1", 
            kwarg2="kw2"
        )
        
        assert result == "arg1_val_arg2_val_kw1_kw2"
        mock_connection.commit.assert_called_once()
        mock_connection.close.assert_called_once()
