import pytest
from unittest.mock import patch, MagicMock, call
import datetime
import sqlalchemy
import pandas as pd
import numpy as np
from src.trader3000.core.data_management import with_connection, DataManager


@pytest.fixture
def mock_engine():
    mock_conn = MagicMock(spec=sqlalchemy.engine.Connection)
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn
    return mock_engine, mock_conn


@pytest.fixture
def data_manager():
    with patch.dict('os.environ', {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
        return DataManager()


def test_wrapper_happy_path(mock_engine, data_manager):
    mock_engine_obj, mock_conn = mock_engine
    
    @with_connection
    def test_func(self, con=None, extra_arg=None):
        assert con is mock_conn
        assert extra_arg == "test_value"
        return "success"
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine_obj) as mock_create_engine:
        result = test_func(data_manager, extra_arg="test_value")
        
        assert result == "success"
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_engine_obj.connect.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


def test_wrapper_exception_rollback(mock_engine, data_manager):
    mock_engine_obj, mock_conn = mock_engine
    
    @with_connection
    def test_func(self, con=None):
        raise ValueError("Test error")
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine_obj):
        with pytest.raises(ValueError, match="Test error"):
            test_func(data_manager)
        
        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()
        mock_conn.close.assert_called_once()


def test_wrapper_with_additional_args_kwargs(mock_engine, data_manager):
    mock_engine_obj, mock_conn = mock_engine
    
    @with_connection
    def test_func(self, arg1, arg2, con=None, kwarg1=None, kwarg2=None):
        return f"{arg1}_{arg2}_{kwarg1}_{kwarg2}"
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine_obj):
        result = test_func(data_manager, "a", "b", kwarg1="c", kwarg2="d")
        
        assert result == "a_b_c_d"
        mock_engine_obj.connect.assert_called_once()


def test_wrapper_connection_close_on_success(mock_engine, data_manager):
    mock_engine_obj, mock_conn = mock_engine
    
    @with_connection
    def test_func(self, con=None):
        return "ok"
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine_obj):
        test_func(data_manager)
        
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
        calls = mock_conn.mock_calls
        assert calls.index(call.commit()) < calls.index(call.close())


def test_wrapper_connection_close_on_exception(mock_engine, data_manager):
    mock_engine_obj, mock_conn = mock_engine
    
    @with_connection
    def test_func(self, con=None):
        raise RuntimeError("Error")
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine_obj):
        with pytest.raises(RuntimeError, match="Error"):
            test_func(data_manager)
        
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()
        calls = mock_conn.mock_calls
        assert calls.index(call.rollback()) < calls.index(call.close())
