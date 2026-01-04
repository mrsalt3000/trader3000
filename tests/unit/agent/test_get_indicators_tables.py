import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_indicators_tables_happy_path(mock_data_manager):
    mock_result = pd.DataFrame({
        "name": ["table1", "table2", "table3"]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_result
        
        result = mock_data_manager.get_indicators_tables()
        
        mock_sql_req.assert_called_once()
        call_args = mock_sql_req.call_args[0][0]
        assert "information_schema.tables" in call_args
        assert "table_schema = 'indicators'" in call_args
        
        assert list(result) == ["table1", "table2", "table3"]


def test_get_indicators_tables_empty_result(mock_data_manager):
    mock_result = pd.DataFrame(columns=["name"])
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_result
        
        result = mock_data_manager.get_indicators_tables()
        
        assert result == []


def test_get_indicators_tables_none_result(mock_data_manager):
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = mock_data_manager.get_indicators_tables()
        
        assert result == []


def test_get_indicators_tables_empty_dataframe(mock_data_manager):
    mock_result = pd.DataFrame()
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_result
        
        result = mock_data_manager.get_indicators_tables()
        
        assert result == []
