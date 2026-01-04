import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, call

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_tables_info_happy_path(data_manager):
    mock_table_names_df = pd.DataFrame({"table_name": ["table1", "table2", "table3"]})
    mock_info_df1 = pd.DataFrame({
        "rows_count": [100],
        "min_date": ["2023-01-01"],
        "max_date": ["2023-12-31"]
    })
    mock_info_df2 = pd.DataFrame({
        "rows_count": [200],
        "min_date": ["2023-02-01"],
        "max_date": ["2023-11-30"]
    })
    mock_info_df3 = pd.DataFrame({
        "rows_count": [150],
        "min_date": ["2023-03-01"],
        "max_date": ["2023-10-31"]
    })
    
    mock_con = MagicMock()
    
    with patch.object(pd, 'read_sql') as mock_read_sql, \
         patch.object(data_manager, 'get_sql_table_info') as mock_get_sql:
        
        mock_read_sql.side_effect = [
            mock_table_names_df,
            mock_info_df1,
            mock_info_df2,
            mock_info_df3
        ]
        
        mock_get_sql.side_effect = [
            "SELECT count(*) AS rows_count, min(date) AS min_date, max(date) AS max_date FROM candles.table1",
            "SELECT count(*) AS rows_count, min(date) AS min_date, max(date) AS max_date FROM candles.table2",
            "SELECT count(*) AS rows_count, min(date) AS min_date, max(date) AS max_date FROM candles.table3"
        ]
        
        result = data_manager.get_tables_info(con=mock_con)
        
        assert mock_read_sql.call_count == 4
        
        first_call = mock_read_sql.call_args_list[0]
        assert "information_schema.tables" in first_call[0][0]
        assert first_call[1]["con"] == mock_con
        
        for i, table_name in enumerate(["table1", "table2", "table3"], 1):
            call_args = mock_read_sql.call_args_list[i]
            assert "candles." + table_name in call_args[0][0]
            assert call_args[1]["con"] == mock_con
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "rows_count" in result.columns
        assert "min_date" in result.columns
        assert "max_date" in result.columns
        assert result["rows_count"].tolist() == [100, 200, 150]


def test_get_tables_info_empty_schema(data_manager):
    mock_empty_df = pd.DataFrame({"table_name": []})
    mock_con = MagicMock()
    
    with patch.object(pd, 'read_sql', return_value=mock_empty_df) as mock_read_sql:
        result = data_manager.get_tables_info(con=mock_con)
        
        mock_read_sql.assert_called_once()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def test_get_tables_info_sql_error(data_manager):
    mock_con = MagicMock()
    
    with patch.object(pd, 'read_sql', side_effect=Exception("Database connection failed")):
        with pytest.raises(Exception) as exc_info:
            data_manager.get_tables_info(con=mock_con)
        
        assert "Database connection failed" in str(exc_info.value)


def test_get_tables_info_with_decorator(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    mock_table_names_df = pd.DataFrame({"table_name": ["test_table"]})
    mock_info_df = pd.DataFrame({
        "rows_count": [50],
        "min_date": ["2023-01-01"],
        "max_date": ["2023-01-31"]
    })
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine) as mock_create_engine, \
         patch.object(pd, 'read_sql') as mock_read_sql, \
         patch.object(data_manager, 'get_sql_table_info') as mock_get_sql:
        
        mock_read_sql.side_effect = [mock_table_names_df, mock_info_df]
        mock_get_sql.return_value = "SELECT count(*) AS rows_count, min(date) AS min_date, max(date) AS max_date FROM candles.test_table"
        
        result = data_manager.get_tables_info()
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_engine.connect.assert_called_once()
        assert mock_read_sql.call_count == 2
        mock_con.commit.assert_called_once()
        mock_con.close.assert_called_once()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["rows_count"].iloc[0] == 50


def test_get_tables_info_rollback_on_error(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine), \
         patch.object(pd, 'read_sql', side_effect=Exception("Test error")):
        
        with pytest.raises(Exception, match="Test error"):
            data_manager.get_tables_info()
        
        mock_con.rollback.assert_called_once()
        mock_con.commit.assert_not_called()
        mock_con.close.assert_called_once()
