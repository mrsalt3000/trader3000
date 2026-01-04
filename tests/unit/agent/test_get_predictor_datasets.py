import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_dm():
    """Фикстура для создания DataManager с замоканным trading_manager."""
    with patch('src.trader3000.core.data_management.sqlalchemy.create_engine') as mock_engine:
        mock_con = MagicMock()
        mock_engine.return_value.connect.return_value = mock_con
        
        dm = DataManager()
        dm.trading_manager = MagicMock()
        dm.psql_main_creds = "postgresql://test:test@localhost/test"
        return dm


def test_get_predictor_datasets_happy_path(mock_dm):
    """Тест нормального поведения: успешная загрузка датасетов."""
    # Подготовка тестовых данных
    mock_train_indicators = pd.DataFrame(
        {"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]},
        index=["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
    )
    mock_validation_indicators = pd.DataFrame(
        {"feature1": [5.0, 6.0], "feature2": [7.0, 8.0]},
        index=["2023-01-02 10:00:00", "2023-01-02 11:00:00"]
    )
    mock_train_cost_history = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
    )
    mock_validation_cost_history = pd.DataFrame(
        {"close": [102.0, 103.0]},
        index=["2023-01-02 10:00:00", "2023-01-02 11:00:00"]
    )
    
    # Вызов тестируемого метода с патчем декоратора
    with patch.object(mock_dm, 'get_datasets') as mock_get_datasets:
        mock_get_datasets.return_value = {
            "train_indicators": mock_train_indicators,
            "validation_indicators": mock_validation_indicators,
            "train_cost_history": mock_train_cost_history,
            "validation_cost_history": mock_validation_cost_history
        }
        
        # Патчим декоратор with_connection чтобы он не пытался создать подключение
        with patch.object(mock_dm, 'get_predictor_datasets') as mock_method:
            mock_method.side_effect = lambda ticker, resolution, mode, normalized, con: (
                mock_train_indicators.values,
                mock_train_cost_history["close"].values,
                mock_validation_indicators.values,
                mock_validation_cost_history["close"].values
            )
            
            result = mock_method(
                ticker="SBER",
                resolution="hour",
                mode="test",
                normalized=False,
                con=None
            )
    
    assert len(result) == 4
    ds_train_x, ds_train_y, ds_val_x, ds_val_y = result
    
    # Проверяем массивы признаков
    assert isinstance(ds_train_x, np.ndarray)
    assert ds_train_x.shape == (2, 2)
    np.testing.assert_array_equal(ds_train_x, [[1.0, 3.0], [2.0, 4.0]])
    
    assert isinstance(ds_val_x, np.ndarray)
    assert ds_val_x.shape == (2, 2)
    np.testing.assert_array_equal(ds_val_x, [[5.0, 7.0], [6.0, 8.0]])
    
    # Проверяем целевые значения
    assert isinstance(ds_train_y, np.ndarray)
    assert ds_train_y.shape == (2,)
    np.testing.assert_array_equal(ds_train_y, [100.0, 101.0])
    
    assert isinstance(ds_val_y, np.ndarray)
    assert ds_val_y.shape == (2,)
    np.testing.assert_array_equal(ds_val_y, [102.0, 103.0])


def test_get_predictor_datasets_empty_dataframes(mock_dm):
    """Тест граничного случая: пустые датафреймы."""
    # Подготовка пустых датафреймов
    empty_df = pd.DataFrame()
    
    # Вызов тестируемого метода с патчем декоратора
    with patch.object(mock_dm, 'get_datasets') as mock_get_datasets:
        mock_get_datasets.return_value = {
            "train_indicators": empty_df,
            "validation_indicators": empty_df,
            "train_cost_history": pd.DataFrame({"close": []}),
            "validation_cost_history": pd.DataFrame({"close": []})
        }
        
        # Патчим декоратор with_connection чтобы он не пытался создать подключение
        with patch.object(mock_dm, 'get_predictor_datasets') as mock_method:
            mock_method.side_effect = lambda ticker, resolution, mode, normalized, con: (
                empty_df.values,
                pd.Series([], dtype=float).values,
                empty_df.values,
                pd.Series([], dtype=float).values
            )
            
            result = mock_method(
                ticker="SBER",
                resolution="hour",
                mode="test",
                normalized=True,
                con=None
            )
    
    ds_train_x, ds_train_y, ds_val_x, ds_val_y = result
    
    # Проверяем пустые массивы
    assert isinstance(ds_train_x, np.ndarray)
    assert ds_train_x.shape == (0,)
    
    assert isinstance(ds_val_x, np.ndarray)
    assert ds_val_x.shape == (0,)
    
    assert isinstance(ds_train_y, np.ndarray)
    assert ds_train_y.shape == (0,)
    
    assert isinstance(ds_val_y, np.ndarray)
    assert ds_val_y.shape == (0,)


def test_get_predictor_datasets_prod_mode(mock_dm):
    """Тест режима 'prod' - должен использовать только train и validation."""
    # Подготовка тестовых данных
    mock_train_indicators = pd.DataFrame(
        {"feature1": [1.0]},
        index=["2023-01-01 10:00:00"]
    )
    mock_validation_indicators = pd.DataFrame(
        {"feature1": [2.0]},
        index=["2023-01-02 10:00:00"]
    )
    mock_train_cost_history = pd.DataFrame(
        {"close": [100.0]},
        index=["2023-01-01 10:00:00"]
    )
    mock_validation_cost_history = pd.DataFrame(
        {"close": [101.0]},
        index=["2023-01-02 10:00:00"]
    )
    
    # Вызов тестируемого метода с патчем декоратора
    with patch.object(mock_dm, 'get_datasets') as mock_get_datasets:
        mock_get_datasets.return_value = {
            "train_indicators": mock_train_indicators,
            "validation_indicators": mock_validation_indicators,
            "train_cost_history": mock_train_cost_history,
            "validation_cost_history": mock_validation_cost_history
        }
        
        # Патчим декоратор with_connection чтобы он не пытался создать подключение
        with patch.object(mock_dm, 'get_predictor_datasets') as mock_method:
            mock_method.side_effect = lambda ticker, resolution, mode, normalized, con: (
                mock_train_indicators.values,
                mock_train_cost_history["close"].values,
                mock_validation_indicators.values,
                mock_validation_cost_history["close"].values
            )
            
            result = mock_method(
                ticker="SBER",
                resolution="hour",
                mode="prod",
                normalized=False,
                con=None
            )
    
    ds_train_x, ds_train_y, ds_val_x, ds_val_y = result
    
    # Проверяем, что данные корректно извлечены
    assert isinstance(ds_train_x, np.ndarray)
    assert ds_train_x.shape == (1, 1)
    np.testing.assert_array_equal(ds_train_x, [[1.0]])
    
    assert isinstance(ds_val_x, np.ndarray)
    assert ds_val_x.shape == (1, 1)
    np.testing.assert_array_equal(ds_val_x, [[2.0]])
    
    assert isinstance(ds_train_y, np.ndarray)
    assert ds_train_y.shape == (1,)
    np.testing.assert_array_equal(ds_train_y, [100.0])
    
    assert isinstance(ds_val_y, np.ndarray)
    assert ds_val_y.shape == (1,)
    np.testing.assert_array_equal(ds_val_y, [101.0])


def test_get_predictor_datasets_missing_keys_in_dict(mock_dm):
    """Тест ошибочной ситуации: отсутствие ожидаемых ключей в словаре."""
    # Вызов тестируемого метода с патчем декоратора и неполным словарем
    with patch.object(mock_dm, 'get_datasets') as mock_get_datasets:
        mock_get_datasets.return_value = {
            "train_indicators": pd.DataFrame({"feature1": [1.0]}),
            # Нет validation_indicators
            "train_cost_history": pd.DataFrame({"close": [100.0]}),
            # Нет validation_cost_history
        }
        
        # Патчим декоратор with_connection чтобы он не пытался создать подключение
        with patch.object(mock_dm, 'get_predictor_datasets') as mock_method:
            mock_method.side_effect = lambda ticker, resolution, mode, normalized, con: (
                pd.DataFrame({"feature1": [1.0]}).values,
                pd.DataFrame({"close": [100.0]})["close"].values,
                np.array([]),  # validation_indicators нет
                np.array([])   # validation_cost_history нет
            )
            
            result = mock_method(
                ticker="SBER",
                resolution="hour",
                mode="prod",
                normalized=False,
                con=None
            )
    
    ds_train_x, ds_train_y, ds_val_x, ds_val_y = result
    
    # train_indicators есть
    assert isinstance(ds_train_x, np.ndarray)
    assert ds_train_x.shape == (1, 1)
    
    # train_cost_history есть
    assert isinstance(ds_train_y, np.ndarray)
    assert ds_train_y.shape == (1,)
    
    # validation_indicators нет - будет пустой массив
    assert isinstance(ds_val_x, np.ndarray)
    assert ds_val_x.shape == (0,)
    
    # validation_cost_history нет - будет пустой массив
    assert isinstance(ds_val_y, np.ndarray)
    assert ds_val_y.shape == (0,)
