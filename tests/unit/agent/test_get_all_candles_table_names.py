import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_all_candles_table_names_happy_path(data_manager):
    expected_tables = ["table1", "table2", "table3"]
    mock_df = pd.DataFrame({"name": expected_tables})
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine) as mock_create_engine, \
         patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
        
        result = data_manager.get_all_candles_table_names()
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_engine.connect.assert_called_once()
        mock_read_sql.assert_called_once()
        
        sql_arg = mock_read_sql.call_args[0][0]
        assert "information_schema.tables" in sql_arg
        assert "table_type = 'BASE TABLE'" in sql_arg
        assert "table_schema = 'candles'" in sql_arg
        
        call_args = mock_read_sql.call_args
        assert len(call_args[0]) == 2
        assert call_args[0][1] == mock_con
        
        mock_con.close.assert_called_once()
        
        assert isinstance(result, np.ndarray)
        assert list(result) == expected_tables


def test_get_all_candles_table_names_empty_result(data_manager):
    mock_df = pd.DataFrame({"name": []})
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine), \
         patch("pandas.read_sql", return_value=mock_df):
        
        result = data_manager.get_all_candles_table_names()
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


def test_get_all_candles_table_names_sql_exception(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine), \
         patch("pandas.read_sql", side_effect=Exception("Database error")):
        
        with pytest.raises(Exception, match="Database error"):
            data_manager.get_all_candles_table_names()


def test_get_all_candles_table_names_with_explicit_connection(data_manager):
    expected_tables = ["table1", "table2"]
    mock_df = pd.DataFrame({"name": expected_tables})
    
    mock_con = MagicMock()
    
    with patch("src.trader3000.core.data_management.pd.read_sql", return_value=mock_df) as mock_read_sql:
        with patch.object(data_manager, 'get_all_candles_table_names') as mock_method:
            mock_method.__wrapped__ = lambda self, con: mock_df["name"].values
            result = data_manager.get_all_candles_table_names(con=mock_con)
            
            mock_read_sql.assert_called_once()
            call_args = mock_read_sql.call_args
            assert call_args[0][1] == mock_con
            
            assert isinstance(result, np.ndarray)
            assert list(result) == expected_tables


def test_get_all_candles_table_names_sorted_result(data_manager):
    unsorted_tables = ["z_table", "a_table", "m_table"]
    sorted_tables = sorted(unsorted_tables)
    mock_df = pd.DataFrame({"name": sorted_tables})
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine), \
         patch("pandas.read_sql", return_value=mock_df):
        
        result = data_manager.get_all_candles_table_names()
        
        assert isinstance(result, np.ndarray)
        result_list = list(result)
        assert result_list == sorted_tables


def test_get_all_candles_table_names_none_values(data_manager):
    mock_df = pd.DataFrame({"name": ["table1", None, "table3", None]})
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch("sqlalchemy.create_engine", return_value=mock_engine), \
         patch("pandas.read_sql", return_value=mock_df):
        
        result = data_manager.get_all_candles_table_names()
        
        assert isinstance(result, np.ndarray)
        result_list = list(result)
        assert result_list == ["table1", None, "table3", None]
        assert len(result_list) == 4
