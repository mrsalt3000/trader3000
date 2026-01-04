import pytest
import datetime
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from src.trader3000.core.data_management import DataManager


def test_seconds_between_dt_happy_path():
    """Тестирование нормального случая с положительной разницей."""
    manager = DataManager()
    
    start_dt = datetime.datetime(2023, 1, 1, 10, 30, 0)
    finish_dt = datetime.datetime(2023, 1, 1, 12, 45, 30)
    
    result = manager.seconds_between_dt(start_dt, finish_dt)
    
    expected_seconds = 2 * 3600 + 15 * 60 + 30
    assert result == expected_seconds


def test_seconds_between_dt_boundary_cases():
    """Тестирование граничных случаев."""
    manager = DataManager()
    
    dt = datetime.datetime(2023, 1, 1, 10, 30, 0)
    result = manager.seconds_between_dt(dt, dt)
    assert result == 0
    
    start_dt = datetime.datetime(2023, 1, 1, 10, 30, 0)
    finish_dt = datetime.datetime(2023, 1, 3, 10, 30, 0)
    result = manager.seconds_between_dt(start_dt, finish_dt)
    assert result == 2 * 24 * 3600
    
    start_dt = datetime.datetime(2023, 1, 2, 10, 30, 0)
    finish_dt = datetime.datetime(2023, 1, 1, 10, 30, 0)
    result = manager.seconds_between_dt(start_dt, finish_dt)
    assert result == -24 * 3600


def test_seconds_between_dt_with_microseconds():
    """Тестирование с микросекундами."""
    manager = DataManager()
    
    start_dt = datetime.datetime(2023, 1, 1, 10, 30, 0, 500000)
    finish_dt = datetime.datetime(2023, 1, 1, 10, 30, 1, 250000)
    
    result = manager.seconds_between_dt(start_dt, finish_dt)
    
    assert result == 0


def test_seconds_between_dt_with_timezone():
    """Тестирование с временными зонами."""
    manager = DataManager()
    
    start_dt = datetime.datetime(2023, 1, 1, 10, 30, 0, tzinfo=datetime.timezone.utc)
    finish_dt = datetime.datetime(2023, 1, 1, 11, 30, 0, tzinfo=datetime.timezone.utc)
    
    result = manager.seconds_between_dt(start_dt, finish_dt)
    
    assert result == 3600


def test_seconds_between_dt_large_difference():
    """Тестирование большой разницы во времени."""
    manager = DataManager()
    
    start_dt = datetime.datetime(2000, 1, 1)
    finish_dt = datetime.datetime(2023, 1, 1)
    
    result = manager.seconds_between_dt(start_dt, finish_dt)
    
    assert result > 0
    assert result > 700_000_000
