import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_common_sql_req_success(data_manager):
    sql = "SELECT * FROM test_table"
    expected_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = data_manager.common_sql_req.__wrapped__(data_manager, sql, con=mock_con, n_attempts=4)
        
        mock_read_sql.assert_called_once_with(sql, mock_con)
        pd.testing.assert_frame_equal(result, expected_df)


def test_common_sql_req_retry_success(data_manager):
    sql = "SELECT * FROM test_table"
    expected_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql") as mock_read_sql, \
         patch("time.sleep") as mock_sleep:
        
        mock_read_sql.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            expected_df
        ]
        
        result = data_manager.common_sql_req.__wrapped__(data_manager, sql, con=mock_con, n_attempts=4)
        
        assert mock_read_sql.call_count == 3
        mock_read_sql.assert_has_calls([
            call(sql, mock_con),
            call(sql, mock_con),
            call(sql, mock_con)
        ])
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(5), call(5)])
        pd.testing.assert_frame_equal(result, expected_df)


def test_common_sql_req_all_attempts_fail(data_manager):
    sql = "SELECT * FROM test_table"
    mock_con = MagicMock()
    
    with patch("pandas.read_sql") as mock_read_sql, \
         patch("time.sleep") as mock_sleep, \
         patch("builtins.print") as mock_print:
        
        mock_read_sql.side_effect = Exception("Persistent error")
        
        result = data_manager.common_sql_req.__wrapped__(data_manager, sql, con=mock_con, n_attempts=3)
        
        assert mock_read_sql.call_count == 3
        assert mock_sleep.call_count == 2
        assert result is None


def test_common_sql_req_zero_attempts(data_manager):
    sql = "SELECT * FROM test_table"
    mock_con = MagicMock()
    
    with patch("pandas.read_sql") as mock_read_sql:
        result = data_manager.common_sql_req.__wrapped__(data_manager, sql, con=mock_con, n_attempts=0)
        
        mock_read_sql.assert_not_called()
        assert result is None


def test_common_sql_req_single_attempt(data_manager):
    sql = "SELECT * FROM test_table"
    expected_df = pd.DataFrame({"col1": [1]})
    mock_con = MagicMock()
    
    with patch("pandas.read_sql", return_value=expected_df) as mock_read_sql:
        result = data_manager.common_sql_req.__wrapped__(data_manager, sql, con=mock_con, n_attempts=1)
        
        mock_read_sql.assert_called_once_with(sql, mock_con)
        pd.testing.assert_frame_equal(result, expected_df)
