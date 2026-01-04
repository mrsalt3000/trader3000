import pytest
from unittest.mock import MagicMock, patch
import sqlalchemy

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    dm = DataManager()
    dm.psql_main_creds = "postgresql://test:test@localhost/test"
    return dm


def test_drop_table_by_name_table_exists(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_comand') as mock_sql_comand:
        
        mock_get_tables.return_value = ["test_table"]
        
        data_manager.drop_table_by_name("test_table", con=mock_con)
        
        mock_get_tables.assert_called_once_with("test_table", con=mock_con)
        mock_sql_comand.assert_called_once_with(
            """DROP TABLE IF EXISTS candles.test_table""",
            con=mock_con
        )


def test_drop_table_by_name_table_not_exists(data_manager):
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    
    with patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_comand') as mock_sql_comand:
        
        mock_get_tables.return_value = []
        
        data_manager.drop_table_by_name("non_existent_table", con=mock_con)
        
        mock_get_tables.assert_called_once_with("non_existent_table", con=mock_con)
        mock_sql_comand.assert_not_called()


def test_drop_table_by_name_without_connection_param(data_manager):
    mock_engine = MagicMock()
    mock_con = MagicMock(spec=sqlalchemy.engine.Connection)
    mock_engine.connect.return_value = mock_con
    
    with patch('src.trader3000.core.data_management.sqlalchemy.create_engine', return_value=mock_engine) as mock_create_engine, \
         patch.object(data_manager, 'get_table_names_by_key') as mock_get_tables, \
         patch.object(data_manager, 'common_sql_comand') as mock_sql_comand:
        
        mock_get_tables.return_value = ["test_table"]
        
        data_manager.drop_table_by_name("test_table")
        
        mock_create_engine.assert_called_once_with(data_manager.psql_main_creds)
        mock_get_tables.assert_called_once_with("test_table", con=mock_con)
        mock_sql_comand.assert_called_once_with(
            """DROP TABLE IF EXISTS candles.test_table""",
            con=mock_con
        )
