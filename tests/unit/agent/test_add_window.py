import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, 'src')
from trader3000.core.data_management import DataManager


def test_add_window_happy_path():
    """Тест нормального поведения с корректными входными данными"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    y_in = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    n_steps = 3
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (2, 6)  # 2 окна, каждое из 3 шагов * 2 признака
    assert y_out.shape == (2,)    # 2 целевых значения
    
    expected_first_window = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.allclose(x_out[0], expected_first_window)
    
    expected_second_window = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert np.allclose(x_out[1], expected_second_window)
    
    expected_y = np.array([10.0, 20.0])
    assert np.allclose(y_out, expected_y)


def test_add_window_boundary_case_n_steps_equals_one():
    """Тест граничного случая, когда n_steps = 1"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    y_in = np.array([10.0, 20.0, 30.0])
    n_steps = 1
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (2, 3)  # 2 окна, каждое из 1 шага * 3 признака
    assert y_out.shape == (2,)    # 2 целевых значения
    
    assert np.allclose(x_out[0], [1.0, 2.0, 3.0])
    assert np.allclose(x_out[1], [4.0, 5.0, 6.0])
    assert np.allclose(y_out, [10.0, 20.0])


def test_add_window_without_y_in():
    """Тест работы метода без передачи y_in (y_in = None)"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    n_steps = 2
    
    # Act
    x_out, y_out = dm.add_window(x_in, None, n_steps)
    
    # Assert
    assert x_out.shape == (1, 4)  # 1 окно, 2 шага * 2 признака
    assert y_out is None
    
    expected_window = [1.0, 2.0, 3.0, 4.0]
    assert np.allclose(x_out[0], expected_window)


def test_add_window_n_steps_greater_than_data_length():
    """Тест случая, когда n_steps больше длины входных данных"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    y_in = np.array([10.0, 20.0])
    n_steps = 3  # Больше, чем 2 строки в x_in
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (0,)  # Пустой массив
    assert y_out.shape == (0,)  # Пустой массив


def test_add_window_with_single_feature():
    """Тест работы с одномерными признаками (один признак на шаг)"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])
    y_in = np.array([10.0, 20.0, 30.0, 40.0])
    n_steps = 2
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (2, 2)  # 2 окна, каждое из 2 шагов * 1 признак
    assert y_out.shape == (2,)
    
    assert np.allclose(x_out[0], [1.0, 2.0])
    assert np.allclose(x_out[1], [2.0, 3.0])
    assert np.allclose(y_out, [10.0, 20.0])


def test_add_window_empty_input():
    """Тест с пустыми входными данными"""
    # Arrange
    dm = DataManager()
    x_in = np.array([]).reshape(0, 3)  # Пустой массив с 3 колонками
    y_in = np.array([])
    n_steps = 1
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (0,)  # Пустой массив
    assert y_out.shape == (0,)  # Пустой массив


def test_add_window_y_in_shorter_than_x_in():
    """Тест, когда y_in короче, чем x_in"""
    # Arrange
    dm = DataManager()
    x_in = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    y_in = np.array([10.0, 20.0])  # На один элемент меньше
    n_steps = 1
    
    # Act
    x_out, y_out = dm.add_window(x_in, y_in, n_steps)
    
    # Assert
    assert x_out.shape == (2, 2)  # 2 окна
    assert y_out.shape == (1,)    # y_in[:-1] даст только первый элемент
    
    assert np.allclose(y_out, [10.0])
