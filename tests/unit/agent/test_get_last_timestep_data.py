import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import datetime
from itertools import chain

from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания DataManager с замоканными зависимостями."""
    manager = DataManager()
    manager.common_sql_req = MagicMock()
    return manager


def test_get_last_timestep_data_happy_path(mock_data_manager):
    """Тест нормального поведения: возвращает корректные данные и метку."""
    # Подготовка тестовых данных
    test_data = pd.DataFrame({
        'date': [
            '2024-01-03 10:00:00',
            '2024-01-03 09:00:00', 
            '2024-01-03 08:00:00'
        ],
        'indicator1': [1.1, 2.2, 3.3],
        'indicator2': [5.5, 6.6, 7.7]
    })
    
    mock_data_manager.common_sql_req.return_value = test_data
    
    # Вызов тестируемой функции
    observation, last_timestep = mock_data_manager.get_last_timestep_data(
        ticker="SBER",
        time_step_resolution="hour",
        window_size=2
    )
    
    # Проверка SQL-запроса
    expected_sql = """
            SELECT
                *
            FROM datasets_sber_hour_validation_indicators
            ORDER BY date DESC
            LIMIT 3
            """
    mock_data_manager.common_sql_req.assert_called_once_with(expected_sql)
    
    # Проверка результата
    expected_observation = list(chain(
        [1.1, 5.5],  # данные из строки 0
        [2.2, 6.6],  # данные из строки 1
        [3.3, 7.7]   # данные из строки 2
    ))
    expected_last_timestep = '2024-01-03 10:00:00'
    
    assert observation == expected_observation
    assert last_timestep == expected_last_timestep


def test_get_last_timestep_data_empty_result(mock_data_manager):
    """Тест граничного случая: запрос возвращает None."""
    mock_data_manager.common_sql_req.return_value = None
    
    # Проверка вызова исключения
    with pytest.raises(ValueError, match="Query returned no data"):
        mock_data_manager.get_last_timestep_data(
            ticker="SBER",
            time_step_resolution="hour",
            window_size=5
        )


def test_get_last_timestep_data_single_row(mock_data_manager):
    """Тест случая с одной строкой данных."""
    test_data = pd.DataFrame({
        'date': ['2024-01-03 10:00:00'],
        'indicator1': [1.1],
        'indicator2': [5.5]
    })
    
    mock_data_manager.common_sql_req.return_value = test_data
    
    observation, last_timestep = mock_data_manager.get_last_timestep_data(
        ticker="SBER",
        time_step_resolution="hour",
        window_size=0
    )
    
    expected_observation = [1.1, 5.5]
    expected_last_timestep = '2024-01-03 10:00:00'
    
    assert observation == expected_observation
    assert last_timestep == expected_last_timestep


def test_get_last_timestep_data_many_indicators(mock_data_manager):
    """Тест с большим количеством индикаторов."""
    test_data = pd.DataFrame({
        'date': ['2024-01-03 10:00:00', '2024-01-03 09:00:00'],
        'ind1': [1.0, 2.0],
        'ind2': [3.0, 4.0],
        'ind3': [5.0, 6.0],
        'ind4': [7.0, 8.0]
    })
    
    mock_data_manager.common_sql_req.return_value = test_data
    
    observation, last_timestep = mock_data_manager.get_last_timestep_data(
        ticker="AAPL",
        time_step_resolution="hour",
        window_size=1
    )
    
    expected_observation = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
    expected_last_timestep = '2024-01-03 10:00:00'
    
    assert observation == expected_observation
    assert last_timestep == expected_last_timestep


def test_get_last_timestep_data_sql_query_format(mock_data_manager):
    """Тест формата SQL-запроса."""
    test_data = pd.DataFrame({
        'date': ['2024-01-03 10:00:00'],
        'indicator': [1.0]
    })
    
    mock_data_manager.common_sql_req.return_value = test_data
    
    mock_data_manager.get_last_timestep_data(
        ticker="GAZP",
        time_step_resolution="15min",
        window_size=10
    )
    
    call_args = mock_data_manager.common_sql_req.call_args[0][0]
    assert "datasets_gazp_15min_validation_indicators" in call_args
    assert "LIMIT 11" in call_args
    assert "ORDER BY date DESC" in call_args


def test_get_last_timestep_data_with_numeric_date(mock_data_manager):
    """Тест с числовым значением даты в результате."""
    test_data = pd.DataFrame({
        'date': [20240103, 20240102],
        'indicator1': [100.0, 200.0],
        'indicator2': [300.0, 400.0]
    })
    
    mock_data_manager.common_sql_req.return_value = test_data
    
    observation, last_timestep = mock_data_manager.get_last_timestep_data(
        ticker="VTBR",
        time_step_resolution="hour",
        window_size=1
    )
    
    expected_observation = [100.0, 300.0, 200.0, 400.0]
    expected_last_timestep = 20240103
    
    assert observation == expected_observation
    assert last_timestep == expected_last_timestep
