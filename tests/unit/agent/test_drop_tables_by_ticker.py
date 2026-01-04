import pytest
from unittest.mock import patch, MagicMock, call
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_drop_tables_by_ticker_happy_path(data_manager):
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'drop_table_by_name') as mock_drop_table:
        
        mock_get_tables.return_value = [
            'aapl_hour',
            'aapl_15min',
            'datasets_aapl_hour_validation_indicators'
        ]
        
        data_manager.drop_tables_by_ticker('AAPL', con=mock_con)
        
        mock_get_tables.assert_called_once_with('aapl', con=mock_con)
        mock_drop_table.assert_has_calls([
            call('aapl_hour', con=mock_con),
            call('aapl_15min', con=mock_con),
            call('datasets_aapl_hour_validation_indicators', con=mock_con)
        ], any_order=True)
        assert mock_drop_table.call_count == 3


def test_drop_tables_by_ticker_no_tables_found(data_manager):
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'drop_table_by_name') as mock_drop_table:
        
        mock_get_tables.return_value = []
        
        data_manager.drop_tables_by_ticker('NONEXISTENT', con=mock_con)
        
        mock_get_tables.assert_called_once_with('nonexistent', con=mock_con)
        mock_drop_table.assert_not_called()


def test_drop_tables_by_ticker_case_insensitive(data_manager):
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'drop_table_by_name') as mock_drop_table:
        
        mock_get_tables.return_value = ['ibm_hour']
        
        data_manager.drop_tables_by_ticker('IBM', con=mock_con)
        
        mock_get_tables.assert_called_once_with('ibm', con=mock_con)
        mock_drop_table.assert_called_once_with('ibm_hour', con=mock_con)


def test_drop_tables_by_ticker_empty_string(data_manager):
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'drop_table_by_name') as mock_drop_table:
        
        mock_get_tables.return_value = []
        
        data_manager.drop_tables_by_ticker('', con=mock_con)
        
        mock_get_tables.assert_called_once_with('', con=mock_con)
        mock_drop_table.assert_not_called()


def test_drop_tables_by_ticker_exception_in_get_tables(data_manager):
    mock_con = MagicMock()
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables:
        
        mock_get_tables.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            data_manager.drop_tables_by_ticker('TEST', con=mock_con)


def test_drop_tables_by_ticker_without_connection_parameter(data_manager):
    with patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_create_engine, \
         patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'drop_table_by_name') as mock_drop_table:
        
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        mock_get_tables.return_value = ['test_hour']
        
        data_manager.drop_tables_by_ticker('TEST')
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_engine.connect.assert_called_once()
        mock_get_tables.assert_called_once_with('test', con=mock_connection)
        mock_drop_table.assert_called_once_with('test_hour', con=mock_connection)
