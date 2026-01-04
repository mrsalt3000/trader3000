import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_dm():
    dm = DataManager()
    dm.psql_main_creds = "postgresql://test:test@localhost/test"
    return dm


def test_update_normalized_datasets_happy_path(mock_dm):
    mock_tables = ["indicators_table1", "indicators_table2"]
    
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = mock_tables
        
        test_data_1 = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'feature1': [100.0, -50.0],
            'feature2': [30.0, 60.0],
            'feature3': [0.0, 25.0]
        })
        
        test_data_2 = pd.DataFrame({
            'date': ['2023-01-01'],
            'feature1': [-200.0],
            'feature2': [0.0],
            'feature3': [150.0]
        })
        
        mock_sql_req.side_effect = [test_data_1, test_data_2]
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        assert mock_get_tables.call_count == 1
        
        expected_sql_calls = [
            call("SELECT * FROM indicators_table1"),
            call("SELECT * FROM indicators_table2")
        ]
        mock_sql_req.assert_has_calls(expected_sql_calls, any_order=False)
        
        assert mock_to_sql.call_count == 2
        
        call_args_1 = mock_to_sql.call_args_list[0]
        assert call_args_1[1]['con'] == mock_con
        assert call_args_1[1]['schema'] == 'datasets'
        assert call_args_1[1]['name'] == 'indicators_table1_normalized'
        assert call_args_1[1]['if_exists'] == 'replace'
        assert call_args_1[1]['index'] is False
        
        df_to_save_1 = call_args_1[0][0]
        assert df_to_save_1['feature1'].tolist() == [1.0, -0.5]
        assert df_to_save_1['feature2'].tolist() == [0.5, 1.0]
        assert df_to_save_1['feature3'].tolist() == [0.0, 1.0]
        
        call_args_2 = mock_to_sql.call_args_list[1]
        df_to_save_2 = call_args_2[0][0]
        assert df_to_save_2['feature1'].tolist() == [-1.0]
        assert df_to_save_2['feature2'].tolist() == [0.0]
        assert df_to_save_2['feature3'].tolist() == [1.0]


def test_update_normalized_datasets_empty_tables(mock_dm):
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = []
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_not_called()
        mock_to_sql.assert_not_called()


def test_update_normalized_datasets_with_empty_dataframe(mock_dm):
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = ['empty_table']
        mock_sql_req.return_value = pd.DataFrame()
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once_with("SELECT * FROM empty_table")
        mock_to_sql.assert_not_called()


def test_update_normalized_datasets_sql_error(mock_dm):
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = ['error_table']
        mock_sql_req.return_value = None
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once_with("SELECT * FROM error_table")
        mock_to_sql.assert_not_called()


def test_update_normalized_datasets_with_single_column(mock_dm):
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = ['single_col_table']
        
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02']
        })
        
        mock_sql_req.return_value = test_data
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once_with("SELECT * FROM single_col_table")
        
        mock_to_sql.assert_called_once()
        call_args = mock_to_sql.call_args
        df_to_save = call_args[0][0]
        
        pd.testing.assert_frame_equal(df_to_save, test_data)
        assert call_args[1]['name'] == 'single_col_table_normalized'


def test_update_normalized_datasets_with_zero_max_value(mock_dm):
    with patch.object(mock_dm, 'get_indicators_tables') as mock_get_tables, \
         patch.object(mock_dm, 'common_sql_req') as mock_sql_req, \
         patch('pandas.DataFrame.to_sql') as mock_to_sql:
        
        mock_get_tables.return_value = ['zero_table']
        
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'feature1': [0.0, 0.0],
            'feature2': [0.0, 0.0]
        })
        
        mock_sql_req.return_value = test_data
        
        mock_con = MagicMock()
        
        with patch.object(mock_dm, 'update_normalized_datasets.__wrapped__') as mock_wrapped:
            mock_dm.update_normalized_datasets(con=mock_con)
        
        call_args = mock_to_sql.call_args
        df_to_save = call_args[0][0]
        
        assert df_to_save['feature1'].tolist() == [0.0, 0.0]
        assert df_to_save['feature2'].tolist() == [0.0, 0.0]
