import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def data_manager():
    dm = DataManager()
    return dm


def test_save_transaction_happy_path(data_manager):
    transaction = {
        "order_id": "12345",
        "buy_dt": "2024-01-15 10:30:00",
        "ticker": "SBER",
        "price": 300.50,
        "quantity": 10
    }
    
    mock_con = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        data_manager.save_transaction(transaction, con=mock_con)
        
        assert mock_to_sql.called
        call_args = mock_to_sql.call_args
        
        assert call_args[1]['con'] == mock_con
        assert call_args[1]['schema'] == "transaction"
        assert call_args[1]['name'] == "transactions"
        assert call_args[1]['if_exists'] == "append"
        assert call_args[1]['index'] is False
        
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert list(df_arg.columns) == list(transaction.keys())
        assert df_arg.iloc[0].to_dict() == transaction


def test_save_transaction_empty_dict(data_manager):
    transaction = {}
    
    mock_con = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        data_manager.save_transaction(transaction, con=mock_con)
        
        assert mock_to_sql.called
        call_args = mock_to_sql.call_args
        
        assert call_args[1]['con'] == mock_con
        assert call_args[1]['schema'] == "transaction"
        assert call_args[1]['name'] == "transactions"
        assert call_args[1]['if_exists'] == "append"
        assert call_args[1]['index'] is False
        
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert list(df_arg.columns) == []


def test_save_transaction_with_none_values(data_manager):
    transaction = {
        "order_id": None,
        "buy_dt": None,
        "ticker": "SBER",
        "price": None,
        "quantity": 10
    }
    
    mock_con = MagicMock()
    
    with patch('pandas.DataFrame.to_sql') as mock_to_sql:
        data_manager.save_transaction(transaction, con=mock_con)
        
        assert mock_to_sql.called
        call_args = mock_to_sql.call_args
        
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert list(df_arg.columns) == list(transaction.keys())
        
        row = df_arg.iloc[0]
        assert row['order_id'] is None
        assert row['buy_dt'] is None
        assert row['ticker'] == "SBER"
        assert row['price'] is None
        assert row['quantity'] == 10


def test_save_transaction_decorator_creates_connection():
    dm = DataManager()
    dm.psql_main_creds = "postgresql://user:pass@localhost/db"
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    mock_to_sql = MagicMock()
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine), \
         patch('pandas.DataFrame.to_sql', mock_to_sql):
        
        transaction = {"order_id": "123", "ticker": "SBER"}
        dm.save_transaction(transaction)
        
        assert mock_engine.connect.called
        assert mock_con.commit.called
        assert mock_con.close.called
        
        assert mock_to_sql.called
        call_args = mock_to_sql.call_args
        assert call_args[1]['con'] == mock_con


def test_save_transaction_exception_handling():
    dm = DataManager()
    dm.psql_main_creds = "postgresql://user:pass@localhost/db"
    
    mock_engine = MagicMock()
    mock_con = MagicMock()
    mock_engine.connect.return_value = mock_con
    
    mock_to_sql = MagicMock(side_effect=Exception("Database error"))
    
    with patch('sqlalchemy.create_engine', return_value=mock_engine), \
         patch('pandas.DataFrame.to_sql', mock_to_sql):
        
        transaction = {"order_id": "123", "ticker": "SBER"}
        
        with pytest.raises(Exception, match="Database error"):
            dm.save_transaction(transaction)
        
        assert mock_con.rollback.called
        assert mock_con.close.called
