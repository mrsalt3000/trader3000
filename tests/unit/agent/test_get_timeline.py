import datetime
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np
from src.trader3000.core.data_management import DataManager


@pytest.fixture
def mock_data_manager():
    """Фикстура для создания экземпляра DataManager с мокнутыми зависимостями."""
    manager = DataManager()
    manager.psql_main_creds = "postgresql://test:test@localhost/test"
    return manager


def test_get_timeline_happy_path(mock_data_manager):
    """Тест нормального поведения: формирование таймлайна для одного торгового дня."""
    # Подготовка тестовых данных
    schedule_data = {
        'work_seconds': [7200],  # 2 часа
        'start_time': ['2024-01-15 10:00:00']
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Вызов тестируемой функции
    result = mock_data_manager.get_timeline(schedule_df, timestep_seconds=1800)  # шаг 30 минут
    
    # Проверки
    assert isinstance(result, list)
    assert len(result) == 5  # 2 часа / 30 минут = 4 шага + 1 из-за range(timesteps_within + 1)
    assert all(isinstance(dt, datetime.datetime) for dt in result)
    
    # Проверка конкретных значений
    expected_times = [
        datetime.datetime(2024, 1, 15, 10, 0, 0),
        datetime.datetime(2024, 1, 15, 10, 30, 0),
        datetime.datetime(2024, 1, 15, 11, 0, 0),
        datetime.datetime(2024, 1, 15, 11, 30, 0),
        datetime.datetime(2024, 1, 15, 12, 0, 0)
    ]
    assert result == expected_times


def test_get_timeline_boundary_case(mock_data_manager):
    """Тест граничного случая: work_seconds меньше timestep_seconds."""
    # Подготовка тестовых данных
    schedule_data = {
        'work_seconds': [1799],  # меньше 30 минут
        'start_time': ['2024-01-15 10:00:00']
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Вызов тестируемой функции
    result = mock_data_manager.get_timeline(schedule_df, timestep_seconds=1800)  # шаг 30 минут
    
    # Проверки
    assert isinstance(result, list)
    assert len(result) == 1  # только start_time (1799 // 1800 = 0, range(0+1) = 1)
    assert result[0] == datetime.datetime(2024, 1, 15, 10, 0, 0)


def test_get_timeline_multiple_days(mock_data_manager):
    """Тест с несколькими торговыми днями."""
    # Подготовка тестовых данных
    schedule_data = {
        'work_seconds': [3600, 7200],  # 1 час и 2 часа
        'start_time': ['2024-01-15 09:00:00', '2024-01-16 10:00:00']
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Вызов тестируемой функции
    result = mock_data_manager.get_timeline(schedule_df, timestep_seconds=1800)  # шаг 30 минут
    
    # Проверки
    assert isinstance(result, list)
    # 1 час: 3600//1800 = 2, range(2+1) = 3 шага
    # 2 часа: 7200//1800 = 4, range(4+1) = 5 шагов
    # Итого: 3 + 5 = 8 шагов
    assert len(result) == 8
    
    # Проверка первого дня
    assert result[0] == datetime.datetime(2024, 1, 15, 9, 0, 0)
    assert result[1] == datetime.datetime(2024, 1, 15, 9, 30, 0)
    assert result[2] == datetime.datetime(2024, 1, 15, 10, 0, 0)
    
    # Проверка второго дня
    assert result[3] == datetime.datetime(2024, 1, 16, 10, 0, 0)
    assert result[4] == datetime.datetime(2024, 1, 16, 10, 30, 0)
    assert result[5] == datetime.datetime(2024, 1, 16, 11, 0, 0)
    assert result[6] == datetime.datetime(2024, 1, 16, 11, 30, 0)
    assert result[7] == datetime.datetime(2024, 1, 16, 12, 0, 0)


def test_get_timeline_zero_work_seconds(mock_data_manager):
    """Тест случая с нулевым рабочим временем."""
    # Подготовка тестовых данных
    schedule_data = {
        'work_seconds': [0],
        'start_time': ['2024-01-15 10:00:00']
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Вызов тестируемой функции
    result = mock_data_manager.get_timeline(schedule_df, timestep_seconds=1800)
    
    # Проверки
    assert isinstance(result, list)
    assert len(result) == 1  # 0 // 1800 = 0, range(0+1) = 1
    assert result[0] == datetime.datetime(2024, 1, 15, 10, 0, 0)


def test_get_timeline_invalid_timestep_zero(mock_data_manager):
    """Тест с нулевым значением timestep_seconds (должно вызвать ZeroDivisionError)."""
    # Подготовка тестовых данных
    schedule_data = {
        'work_seconds': [3600],
        'start_time': ['2024-01-15 10:00:00']
    }
    schedule_df = pd.DataFrame(schedule_data)
    
    # Вызов тестируемой функции с нулевым шагом
    with pytest.raises(ZeroDivisionError):
        mock_data_manager.get_timeline(schedule_df, timestep_seconds=0)


def test_get_timeline_empty_schedule(mock_data_manager):
    """Тест с пустым расписанием."""
    # Подготовка пустого DataFrame
    schedule_df = pd.DataFrame(columns=['work_seconds', 'start_time'])
    
    # Вызов тестируемой функции
    result = mock_data_manager.get_timeline(schedule_df, timestep_seconds=1800)
    
    # Проверки
    assert isinstance(result, list)
    assert len(result) == 0
