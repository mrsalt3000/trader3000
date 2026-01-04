import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import datetime
import numpy as np
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


@pytest.fixture
def mock_candles_data():
    dates = pd.date_range(start="2020-01-01", periods=100, freq="h")
    data = pd.DataFrame({
        "date": dates,
        "open": np.random.randn(100),
        "high": np.random.randn(100),
        "low": np.random.randn(100),
        "close": np.random.randn(100),
        "volume": np.random.randn(100)
    })
    return data


def test_recalculate_indicators_total_mode_happy_path(data_manager, mock_candles_data):
    with patch.object(data_manager, 'get_all_candles_table_names') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_req') as mock_sql_req, \
         patch('src.trader3000.core.data_management.IndicatorsCalculator') as mock_calculator, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_pd_read_sql, \
         patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tables.return_value = ["table1_hour"]
        mock_sql_req.return_value = mock_candles_data
        
        mock_calculator_instance = MagicMock()
        mock_calculator.return_value = mock_calculator_instance
        
        dates = pd.date_range(start="2020-01-01", periods=100, freq="h")
        calculated_data = pd.DataFrame(
            np.random.randn(100, 20),
            index=dates,
            columns=[f"col_{i}" for i in range(20)]
        )
        mock_calculator_instance.calculate_indicators.return_value = calculated_data
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_con
        
        mock_pd_read_sql.return_value = pd.DataFrame({"table_name": ["table1_hour"]})
        
        data_manager.recalculate_indicators(mode="total")
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once()
        mock_calculator.assert_called_once()
        mock_calculator_instance.calculate_indicators.assert_called_once()


def test_recalculate_indicators_update_mode_happy_path(data_manager, mock_candles_data):
    with patch.object(data_manager, 'get_all_candles_table_names') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_req') as mock_sql_req, \
         patch('src.trader3000.core.data_management.IndicatorsCalculator') as mock_calculator, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_pd_read_sql, \
         patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tables.return_value = ["table1_hour"]
        
        old_dates = pd.date_range(start="2019-12-01", periods=50, freq="h")
        old_cost_history_df = pd.DataFrame({
            "date": old_dates,
            "col_0": np.random.randn(50),
            "col_1": np.random.randn(50),
            "col_2": np.random.randn(50),
            "col_3": np.random.randn(50),
            "col_4": np.random.randn(50)
        })
        
        old_indicators_df = pd.DataFrame({
            "date": old_dates,
            "col_5": np.random.randn(50),
            "col_6": np.random.randn(50)
        })
        
        mock_sql_req.side_effect = [
            mock_candles_data,
            old_cost_history_df,
            old_indicators_df,
        ]
        
        mock_calculator_instance = MagicMock()
        mock_calculator.return_value = mock_calculator_instance
        
        dates = pd.date_range(start="2020-01-01", periods=100, freq="h")
        calculated_data = pd.DataFrame(
            np.random.randn(100, 20),
            index=dates,
            columns=[f"col_{i}" for i in range(20)]
        )
        mock_calculator_instance.calculate_indicators.return_value = calculated_data
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_con
        
        mock_pd_read_sql.return_value = pd.DataFrame({"table_name": ["table1_hour"]})
        
        data_manager.recalculate_indicators(mode="update")
        
        mock_get_tables.assert_called_once()
        assert mock_sql_req.call_count == 3
        mock_calculator.assert_called_once()
        mock_calculator_instance.calculate_indicators.assert_called_once()


def test_recalculate_indicators_with_none_candles_data(data_manager):
    with patch.object(data_manager, 'get_all_candles_table_names') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_req') as mock_sql_req, \
         patch('src.trader3000.core.data_management.IndicatorsCalculator') as mock_calculator, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_pd_read_sql, \
         patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tables.return_value = ["table1_hour"]
        mock_sql_req.return_value = None
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_con
        
        mock_pd_read_sql.return_value = pd.DataFrame({"table_name": ["table1_hour"]})
        
        data_manager.recalculate_indicators(mode="total")
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once()
        mock_calculator.assert_not_called()


def test_recalculate_indicators_with_empty_candles_data(data_manager):
    with patch.object(data_manager, 'get_all_candles_table_names') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_req') as mock_sql_req, \
         patch('src.trader3000.core.data_management.IndicatorsCalculator') as mock_calculator, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_pd_read_sql, \
         patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tables.return_value = ["table1_hour"]
        mock_sql_req.return_value = pd.DataFrame()
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_con
        
        mock_pd_read_sql.return_value = pd.DataFrame({"table_name": ["table1_hour"]})
        
        data_manager.recalculate_indicators(mode="total")
        
        mock_get_tables.assert_called_once()
        mock_sql_req.assert_called_once()
        mock_calculator.assert_not_called()


def test_recalculate_indicators_invalid_mode(data_manager):
    with patch.object(data_manager, 'get_all_candles_table_names') as mock_get_tables, \
         patch('src.trader3000.core.data_management.pd.read_sql') as mock_pd_read_sql, \
         patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine:
        
        mock_get_tables.return_value = []
        
        mock_engine = MagicMock()
        mock_con = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_con
        
        mock_pd_read_sql.return_value = pd.DataFrame({"table_name": []})
        
        data_manager.recalculate_indicators(mode="invalid_mode")
        
        mock_get_tables.assert_called_once()
