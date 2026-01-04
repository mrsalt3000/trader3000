import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, create_autospec
import datetime
import sqlalchemy

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_datasets_test_mode_single_ticker(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    mock_cost_history = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'close': [100.0, 101.0]
    })
    
    mock_indicators = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'rsi': [50.0, 55.0]
    })
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            def read_sql_side_effect(sql, con):
                if 'cost_history' in sql:
                    return mock_cost_history.copy()
                elif 'indicators' in sql:
                    return mock_indicators.copy()
                return pd.DataFrame()
            
            mock_read_sql.side_effect = read_sql_side_effect
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                mock_get_sql.return_value = "SELECT * FROM test_table"
                
                result = data_manager.get_datasets(
                    ticker="AAPL",
                    resolution="hour",
                    con=mock_con,
                    mode="test",
                    normalized=False
                )
    
    assert isinstance(result, dict)
    expected_keys = [
        'train_cost_history', 'train_indicators',
        'test_cost_history', 'test_indicators',
        'validation_cost_history', 'validation_indicators'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], pd.DataFrame)
    
    assert len(result['train_cost_history']) == 2
    assert len(result['train_indicators']) == 2


def test_get_datasets_prod_mode_multy_ticker(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    mock_tickers = ["AAPL", "GOOGL"]
    
    mock_cost_history = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'close': [100.0, 101.0]
    })
    
    mock_indicators = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'rsi': [50.0, 55.0]
    })
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = mock_tickers
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            def read_sql_side_effect(sql, con):
                if 'cost_history' in sql:
                    return mock_cost_history.copy()
                elif 'indicators' in sql:
                    return mock_indicators.copy()
                return pd.DataFrame()
            
            mock_read_sql.side_effect = read_sql_side_effect
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                mock_get_sql.return_value = "SELECT * FROM test_table"
                
                result = data_manager.get_datasets(
                    ticker="multy",
                    resolution="hour",
                    con=mock_con,
                    mode="prod",
                    normalized=True
                )
    
    assert isinstance(result, dict)
    expected_keys = [
        'train_cost_history', 'train_indicators',
        'validation_cost_history', 'validation_indicators'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], pd.DataFrame)
    
    assert len(result['train_cost_history']) == 4
    assert len(result['validation_cost_history']) == 4


def test_get_datasets_with_normalized_indicators(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    mock_cost_history = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'close': [100.0, 101.0]
    })
    
    mock_indicators = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02'],
        'rsi': [0.5, 0.55]
    })
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            def read_sql_side_effect(sql, con):
                if 'cost_history' in sql:
                    return mock_cost_history.copy()
                elif 'indicators' in sql:
                    return mock_indicators.copy()
                return pd.DataFrame()
            
            mock_read_sql.side_effect = read_sql_side_effect
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                def get_sql_side_effect(ticker, resolution, dataset_type, part, normalized=False):
                    if normalized and part == "indicators":
                        return f"SELECT * FROM datasets_{ticker}_{resolution}_{dataset_type}_indicators_normalized"
                    return f"SELECT * FROM datasets_{ticker}_{resolution}_{dataset_type}_{part}"
                
                mock_get_sql.side_effect = get_sql_side_effect
                
                result = data_manager.get_datasets(
                    ticker="AAPL",
                    resolution="hour",
                    con=mock_con,
                    mode="test",
                    normalized=True
                )
    
    calls = mock_get_sql.call_args_list
    indicator_calls = [c for c in calls if c[0][3] == "indicators"]
    for call_args in indicator_calls:
        assert call_args[1]['normalized'] == True


def test_get_datasets_empty_result(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                mock_get_sql.return_value = "SELECT * FROM test_table"
                
                result = data_manager.get_datasets(
                    ticker="UNKNOWN",
                    resolution="hour",
                    con=mock_con,
                    mode="test",
                    normalized=False
                )
    
    assert isinstance(result, dict)
    for key in result:
        assert isinstance(result[key], pd.DataFrame)
        assert len(result[key]) == 0


def test_get_datasets_invalid_mode_returns_prod_structure(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({'date': [], 'close': []})
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                mock_get_sql.return_value = "SELECT * FROM test_table"
                
                result = data_manager.get_datasets(
                    ticker="AAPL",
                    resolution="hour",
                    con=mock_con,
                    mode="invalid_mode",
                    normalized=False
                )
    
    assert isinstance(result, dict)
    expected_keys = [
        'train_cost_history', 'train_indicators',
        'validation_cost_history', 'validation_indicators'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], pd.DataFrame)


def test_get_datasets_exception_handling(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    with patch.object(data_manager, 'get_tickers_from_datasets') as mock_get_tickers:
        mock_get_tickers.return_value = []
        
        with patch.object(pd, 'read_sql') as mock_read_sql:
            mock_read_sql.side_effect = Exception("Database connection error")
            
            with patch.object(data_manager, 'get_sql_select_candles') as mock_get_sql:
                mock_get_sql.return_value = "SELECT * FROM test_table"
                
                with pytest.raises(Exception):
                    data_manager.get_datasets(
                        ticker="AAPL",
                        resolution="hour",
                        con=mock_con,
                        mode="test",
                        normalized=False
                    )
