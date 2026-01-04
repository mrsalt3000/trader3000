import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    manager = DataManager()
    manager.trading_manager = MagicMock()
    return manager


def test_get_all_features_names_happy_path(mock_data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'feature1': [1.0],
        'set_calendar': [2.0],
        'feature2': [3.0],
        'set_weather': [4.0],
        'feature3': [5.0]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_data):
        result = mock_data_manager.get_all_features_names()
        
        assert result == ['feature1', 'feature2', 'feature3']
        assert 'set_calendar' not in result
        assert 'set_weather' not in result


def test_get_all_features_names_empty_result(mock_data_manager):
    with patch.object(mock_data_manager, 'common_sql_req', return_value=None):
        result = mock_data_manager.get_all_features_names()
        assert result == []
    
    empty_df = pd.DataFrame()
    with patch.object(mock_data_manager, 'common_sql_req', return_value=empty_df):
        result = mock_data_manager.get_all_features_names()
        assert result == []
    
    df_only_date = pd.DataFrame({'date': ['2023-01-01']})
    with patch.object(mock_data_manager, 'common_sql_req', return_value=df_only_date):
        result = mock_data_manager.get_all_features_names()
        assert result == []


def test_get_all_features_names_only_set_features(mock_data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'set_calendar_day': [1],
        'set_calendar_month': [2],
        'set_weather_temp': [20.5]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_data):
        result = mock_data_manager.get_all_features_names()
        assert result == []


def test_get_all_features_names_with_mixed_columns(mock_data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'set_config': [1],
        'volume': [1000],
        'set_group': [2],
        'price': [50.5],
        'another': [3]
    })
    
    with patch.object(mock_data_manager, 'common_sql_req', return_value=mock_data):
        result = mock_data_manager.get_all_features_names()
        
        assert result == ['volume', 'price', 'another']
        assert 'set_config' not in result
        assert 'set_group' not in result
        assert 'date' not in result


def test_get_all_features_names_exception_handled(mock_data_manager):
    with patch.object(mock_data_manager, 'get_all_features_names_with_sets', return_value=[]):
        result = mock_data_manager.get_all_features_names()
        assert result == []
