import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import datetime
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_orders_rancid_happy_path(mock_data_manager):
    test_date = datetime.datetime.now().date() - datetime.timedelta(days=2)
    expected_array = np.array([
        ["order_123", str(test_date), 10],
        ["order_456", str(test_date), 5]
    ])
    
    mock_df = MagicMock()
    mock_df.values = expected_array
    mock_df.__len__ = lambda self: 2
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_df
        
        result = mock_data_manager.get_orders_rancid(days_treshold=1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0][0] == "order_123"
        assert result[1][0] == "order_456"
        mock_sql_req.assert_called_once()


def test_get_orders_rancid_empty_result(mock_data_manager):
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = mock_data_manager.get_orders_rancid(days_treshold=0)
        
        assert result == []
        mock_sql_req.assert_called_once()


def test_get_orders_rancid_empty_dataframe(mock_data_manager):
    import pandas as pd
    
    mock_df = pd.DataFrame(columns=['order_id', 'buy_dt', 'lots_requested'])
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_df
        
        result = mock_data_manager.get_orders_rancid(days_treshold=5)
        
        assert result == []
        mock_sql_req.assert_called_once()


def test_get_orders_rancid_with_dataframe_result(mock_data_manager):
    import pandas as pd
    
    test_date = datetime.datetime.now().date() - datetime.timedelta(days=3)
    test_data = {
        'order_id': ['order_789', 'order_999'],
        'buy_dt': [str(test_date), str(test_date)],
        'lots_requested': [15, 20]
    }
    mock_df = pd.DataFrame(test_data)
    
    with patch.object(mock_data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_df
        
        result = mock_data_manager.get_orders_rancid(days_treshold=2)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0][0] == "order_789"
        assert result[1][2] == 20
        mock_sql_req.assert_called_once()
