import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_all_features_names_with_sets_happy_path(data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'feature1': [1.0],
        'feature2': [2.0],
        'set_calendar': [3.0],
        'feature3': [4.0]
    })
    
    with patch.object(data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_data
        
        result = data_manager.get_all_features_names_with_sets()
        
        mock_sql_req.assert_called_once_with("""
            SELECT
                *
            FROM datasets_sber_hour_validation_indicators
            LIMIT 1
            """)
        
        assert result == ['feature1', 'feature2', 'set_calendar', 'feature3']


def test_get_all_features_names_with_sets_empty_result(data_manager):
    with patch.object(data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = data_manager.get_all_features_names_with_sets()
        
        assert result == []


def test_get_all_features_names_with_sets_single_column(data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01']
    })
    
    with patch.object(data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_data
        
        result = data_manager.get_all_features_names_with_sets()
        
        assert result == []


def test_get_all_features_names_with_sets_exception_in_sql(data_manager):
    with patch.object(data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = None
        
        result = data_manager.get_all_features_names_with_sets()
        
        assert result == []
        mock_sql_req.assert_called_once()


def test_get_all_features_names_with_sets_various_column_names(data_manager):
    mock_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'feature_1': [1.0],
        'feature-2': [2.0],
        'feature.3': [3.0],
        'set_calendar_day': [4.0],
        'set_weather': [5.0],
        'normal_feature': [6.0]
    })
    
    with patch.object(data_manager, 'common_sql_req') as mock_sql_req:
        mock_sql_req.return_value = mock_data
        
        result = data_manager.get_all_features_names_with_sets()
        
        expected = ['feature_1', 'feature-2', 'feature.3', 
                   'set_calendar_day', 'set_weather', 'normal_feature']
        assert result == expected
