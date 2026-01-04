import pytest
import numpy as np
from unittest.mock import patch
import datetime
import pandas as pd

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_orders_to_sell_by_exp_time_happy_path(mock_data_manager):
    test_data = pd.DataFrame({
        'order_id': ['order_1', 'order_2'],
        'estimated_price_change': [1.5, 2.3]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        result = mock_data_manager.get_orders_to_sell_by_exp_time(days_treshold=5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result[0][0] == 'order_1'
        assert result[0][1] == 1.5
        assert result[1][0] == 'order_2'
        assert result[1][1] == 2.3
        
        treshold_date = str((datetime.datetime.now() - datetime.timedelta(days=5)).date())
        expected_sql = f"""
        SELECT
            order_id
            ,estimated_price_change
        FROM transactions 
        WHERE execution_report_status in ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'take_profit_posted'
            AND buy_dt <= '{treshold_date}'
        """
        
        actual_sql = mock_sql_req.call_args[0][0]
        actual_sql_normalized = ' '.join(actual_sql.split())
        expected_sql_normalized = ' '.join(expected_sql.split())
        assert actual_sql_normalized == expected_sql_normalized


def test_get_orders_to_sell_by_exp_time_no_data(mock_data_manager):
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = mock_data_manager.get_orders_to_sell_by_exp_time(days_treshold=10)
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        assert result.shape == (0,)


def test_get_orders_to_sell_by_exp_time_empty_dataframe(mock_data_manager):
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = pd.DataFrame()
        
        result = mock_data_manager.get_orders_to_sell_by_exp_time(days_treshold=3)
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        assert result.shape == (0,)


def test_get_orders_to_sell_by_exp_time_zero_threshold(mock_data_manager):
    test_data = pd.DataFrame({
        'order_id': ['order_1'],
        'estimated_price_change': [1.0]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = test_data
        
        result = mock_data_manager.get_orders_to_sell_by_exp_time(days_treshold=0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)
        
        treshold_date = str(datetime.datetime.now().date())
        actual_sql = mock_sql_req.call_args[0][0]
        assert treshold_date in actual_sql
