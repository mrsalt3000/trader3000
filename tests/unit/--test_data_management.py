# tests/unit/core/test_data_management.py

import datetime as dt
import datetime
from itertools import chain
import random
from sqlalchemy.sql import text
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, ANY

from src.trader3000.core.data_management import (
    DataManager,
    with_connection,
)

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value = conn
    return engine, conn


@pytest.fixture
def dm(monkeypatch, mock_engine):
    engine, conn = mock_engine
    # мокаем create_engine, чтобы не ходить в реальную БД
    monkeypatch.setattr(
        "src.trader3000.core.data_management.sqlalchemy.create_engine",
        lambda url: engine,
    )
    # мокаем env-переменную
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    trading_manager = MagicMock()
    return DataManager(trading_manager=trading_manager)


# ---------- with_connection декоратор ----------

def test_with_connection_calls_commit_and_close(dm, mock_engine):
    engine, conn = mock_engine

    @with_connection
    def func(self, con=None):
        con.execute("SELECT 1")
        return "ok"

    result = func(dm)

    assert result == "ok"
    conn.execute.assert_called_once()
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


def test_with_connection_rollback_on_exception(dm, mock_engine):
    engine, conn = mock_engine

    @with_connection
    def func(self, con=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        func(dm)

    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# ---------- get_sql_table_info / get_tables_info ----------

def test_get_sql_table_info_query(dm):
    sql = dm.get_sql_table_info("sber_hour")
    assert "from candles.sber_hour" in sql.lower()

def test_get_tables_info(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    # Мокаем engine и connection из sqlalchemy
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    # Функция для подмены pd.read_sql
    def fake_read_sql(sql, con):
        sql_lower = sql.lower()
        if "information_schema.tables" in sql_lower:
            return pd.DataFrame({"table_name": ["sber_hour", "gazp_1min"]})
        elif "from candles.sber_hour" in sql_lower:
            return pd.DataFrame({
                "rows_count": [100],
                "min_date": ["2020-01-01"],
                "max_date": ["2020-02-01"]
            })
        elif "from candles.gazp_1min" in sql_lower:
            return pd.DataFrame({
                "rows_count": [200],
                "min_date": ["2020-02-01"],
                "max_date": ["2020-03-01"]
            })
        else:
            raise ValueError(f"Unexpected SQL: {sql}")

    monkeypatch.setattr("src.trader3000.core.data_management.pd.read_sql", fake_read_sql)

    with patch("src.trader3000.core.data_management.sqlalchemy.create_engine", return_value=mock_engine):
        dm = DataManager()
        result_df = dm.get_tables_info()

    assert isinstance(result_df, pd.DataFrame)
    for col in ["rows_count", "min_date", "max_date"]:
        assert col in result_df.columns
    assert len(result_df) == 2

# ---------- commonsqlreq / commonsqlcomand ----------

def test_common_sql_req_success(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    # Подменяем pd.read_sql на функцию, возвращающую DataFrame
    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        lambda sql, con: pd.DataFrame({"a": [1, 2]}),
    )

    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    with patch("src.trader3000.core.data_management.sqlalchemy.create_engine", return_value=mock_engine):
        dm = DataManager()
        df = dm.common_sql_req("SELECT 1")

    assert isinstance(df, pd.DataFrame)
    assert list(df["a"]) == [1, 2]



def test_common_sql_req_returns_none_on_failure(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    calls = {"n": 0}

    def failing_read_sql(sql, con):
        calls["n"] += 1
        raise RuntimeError("db error")

    monkeypatch.setattr("src.trader3000.core.data_management.pd.read_sql", failing_read_sql)

    # Мокаем create_engine и соединение, чтобы не ходить в реальную базу
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    with patch("src.trader3000.core.data_management.sqlalchemy.create_engine", return_value=mock_engine):
        dm = DataManager()
        df = dm.common_sql_req("SELECT 1", n_attempts=2)

    assert df is None
    assert calls["n"] == 2


def test_common_sql_command_logs_error_but_not_raise(dm, monkeypatch):
    calls = {"n": 0}

    def failing_read_sql(sql, con):
        calls["n"] += 1
        raise RuntimeError("db error")

    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        failing_read_sql
    )

    # не должно выбрасывать исключение наружу
    dm.common_sql_comand("DELETE FROM x")
    assert calls["n"] == 1


# ---------- get_candles_last_dt ----------

def test_get_candles_last_dt_ok(dm, monkeypatch):
    df = pd.DataFrame({"ans": ["2024-01-02 13:45:00"]})
    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        lambda sql, con: df,
    )

    last_dt = dm.get_candles_last_dt("SBER", "hour")

    assert isinstance(last_dt, dt.datetime)
    assert last_dt.year == 2024
    assert last_dt.tzinfo is dt.timezone.utc


def test_get_candles_last_dt_raises_if_no_data(dm, monkeypatch):
    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        lambda sql, con: pd.DataFrame({"ans": []}),
    )

    with pytest.raises(ValueError):
        dm.get_candles_last_dt("SBER", "hour")


# ---------- seconds_between_dt ----------

def test_seconds_between_dt(dm):
    start = dt.datetime(2024, 1, 1, 10, 0, 0)
    finish = dt.datetime(2024, 1, 1, 11, 30, 0)

    seconds = dm.seconds_between_dt(start, finish)

    assert seconds == 90 * 60


# ---------- get_timeline ----------

def test_get_timeline(dm):
    schedule = pd.DataFrame(
        [
            {
                "schedule": "MOEX",
                "date": "2024-01-01",
                "work_seconds": 3600,        # правильное имя колонки
                "start_time": "2024-01-01 10:00:00",  # правильное имя колонки
            }
        ]
    )

    timeline = dm.get_timeline(schedule, timestep_seconds=900)

    # проверки, например, длина timeline, типы элементов, etc.
    assert len(timeline) == 5  # 3600/900=4 интервала + 1
    assert all(isinstance(t, datetime.datetime) for t in timeline)



# ---------- add_window ----------

def test_add_window_only_x(dm):
    x_in = np.arange(10).reshape(-1, 1)
    x_out, y_out = dm.add_window(x_in, y_in=None, n_steps=3)

    assert x_out.shape[0] == 7  # 10 - 3
    assert isinstance(y_out, np.ndarray)
    assert y_out.shape == ()
    assert y_out.item() is None





def test_add_window_with_y(dm):
    x_in = np.arange(10).reshape(-1, 1)
    y_in = np.arange(10)

    x_out, y_out = dm.add_window(x_in, y_in=y_in, n_steps=3)

    # Проверяем размеры выходных данных
    assert x_out.shape[0] == y_out.shape[0]
    assert y_out.shape[0] == 7  # длина y_out = len(y_in) - n_steps



# ---------- get_orders_* простая выборка ----------

def test_get_orders_bought_empty(dm, monkeypatch):
    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        lambda sql, con: pd.DataFrame(),
    )

    arr = dm.get_orders_bought()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0, 2)


def test_get_orders_bought_non_empty(dm, monkeypatch):
    df = pd.DataFrame(
        {
            "orderid": [1, 2],
            "estimatedpricechange": [0.1, -0.2],
        }
    )
    monkeypatch.setattr(
        "src.trader3000.core.data_management.pd.read_sql",
        lambda sql, con: df,
    )

    arr = dm.get_orders_bought()
    assert arr.shape == (2, 2)
    assert arr[0, 0] == 1

def test_training_datasets_download():
    with patch.dict('os.environ', {'DATABASE_URL': 'sqlite:///:memory:'}):
        mock_trading_manager = MagicMock()
        data_manager = DataManager(trading_manager=mock_trading_manager)

        # Мок данных, который возвращает get_market_trading_history
        mock_data = MagicMock()
        mock_data.__len__.return_value = 5
        mock_data.to_sql = MagicMock()

        mock_trading_manager.get_market_trading_history.return_value = mock_data

        tickers = ["AAPL"]
        time_steps = ["1min"]

        with patch("time.sleep", return_value=None), \
             patch.object(data_manager, 'get_last_update_date', return_value=None), \
             patch.object(data_manager, 'del_ticker_from_scope'), \
             patch.object(data_manager, 'drop_tables_by_ticker'):

            result = data_manager.training_datasets_download(
                tickers=tickers,
                time_step_resolutions=time_steps,
                start_date=datetime.datetime(2019, 1, 1),
                finish_date=None,
                save_mode="replace",
                # con не передаем, отдается декоратором
            )

        mock_trading_manager.get_market_trading_history.assert_called_once_with(
            ticker="AAPL",
            start_date=datetime.datetime(2019, 1, 1),
            finish_date=None,
            time_step_resolution="1min",
        )

        mock_data.to_sql.assert_called_once_with(
            con=ANY,  # Игнорируем конкретный объект соединения
            name="aapl_1min",
            schema="candles",
            if_exists="replace",
            index=False,
        )

        assert result is None


def test_download_candles_delta(monkeypatch, dm):
    # Мок функции get_candles_last_dt, чтобы вернуть дату старта
    monkeypatch.setattr(dm, "get_candles_last_dt", lambda ticker, resolution: datetime.datetime(2024, 1, 1))

    # Мок метода training_datasets_download, чтобы не делать реальных вызовов
    monkeypatch.setattr(dm, "training_datasets_download", MagicMock())

    # Вызов с правильными списками аргументов
    dm.download_candles_delta(
        tickers=["SBER"],
        time_step_resolutions=["hour"]
    )

    # Проверяем, что training_datasets_download вызван с правильными параметрами
    dm.training_datasets_download.assert_called_once()
    args, kwargs = dm.training_datasets_download.call_args
    assert kwargs["tickers"] == ["SBER"]
    assert kwargs["time_step_resolutions"] == ["hour"]
    assert kwargs["save_mode"] == "append"



def test_load_all_candles(monkeypatch, dm):
    # Мокаем методы, используемые в load_all_candles
    monkeypatch.setattr(dm, "get_all_candles_loaded", MagicMock(return_value=["sber"]))
    monkeypatch.setattr(dm.trading_manager, "get_all_instruments_tickers", MagicMock(return_value=["SBER", "AAPL"]))

    # Мокаем сам вызов training_datasets_download, чтобы не выполнять реальную загрузку
    monkeypatch.setattr(dm, "training_datasets_download", MagicMock())

    # Мокаем sqlalchemy.create_engine и engine.connect, чтобы подавить эффект декоратора
    mock_con = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_con
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        dm.load_all_candles(mode="add")

    # Проверяем, что training_datasets_download вызван корректно
    dm.training_datasets_download.assert_called_once()

    args, kwargs = dm.training_datasets_download.call_args

    expected_tickers = ["AAPL"]  # "SBER" в candles_loaded, загружаем только "AAPL"
    assert kwargs["tickers"] == expected_tickers

    # Проверяем остальные параметры
    assert kwargs["time_step_resolutions"] == ["hour"]
    assert kwargs["start_date"] == datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
    assert kwargs["finish_date"] == datetime.datetime(2024, 12, 31, tzinfo=datetime.timezone.utc)
    assert kwargs["save_mode"] == "replace"

def test_update_training_data(monkeypatch, dm):
    tickers = ["SBER", "AAPL", "GOOG"]

    # Мокаем случайный выбор тикера, чтобы всегда возвращался "SBER"
    monkeypatch.setattr(random, "choice", lambda x: "SBER")

    # Мокаем common_sql_req чтобы вернуть дату в нужном формате
    mock_result = MagicMock()
    mock_result.values = [["2025-11-20"]]  # допустим, 4 дня назад от current date 24 ноября 2025
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=mock_result))

    # Мокаем get_all_candles_loaded — допустим все тикеры загружены, включая SBER и AAPL
    monkeypatch.setattr(dm, "get_all_candles_loaded", MagicMock(return_value=["SBER", "AAPL"]))

    # Мокаем download_candles_delta, чтобы не выполнять реальные вызовы
    monkeypatch.setattr(dm, "download_candles_delta", MagicMock())

    # Чтобы контролировать текущую дату, патчим datetime.datetime.now()
    fixed_today = datetime.datetime(2025, 11, 24)
    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_today

    with patch('datetime.datetime', FixedDateTime):
        dm.update_training_data(tickers)

    # Проверяем, что download_candles_delta вызвался с правильными тикерами
    dm.download_candles_delta.assert_called_once()
    called_args = dm.download_candles_delta.call_args[0][0]
    # Входные тикеры фильтруются по загруженным, т.е. 'GOOG' исключается
    assert set(called_args) == {"SBER", "AAPL"}

    # Проверка что common_sql_req вызван с корректным SQL
    dm.common_sql_req.assert_called_once()
    assert "candles.sber_hour" in dm.common_sql_req.call_args[0][0].lower()


def test_recalculate_indicators_total_mode(monkeypatch, dm):
    # Мокаем список таблиц
    monkeypatch.setattr(dm, "get_all_candles_table_names", MagicMock(return_value=["candles_sber_hour"]))

    # Мокаем common_sql_req, возвращая некоторый DataFrame с нужными данными
    sample_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4),
        "price": [100, 101, 102, 103]
    })
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=sample_df.copy()))

    # Мокаем IndicatorsCalculator и его метод calculate_indicators
    mock_calculator = MagicMock()
    mock_calculated_data = pd.DataFrame({
        "ind1": [1, 2, 3, 4],
        "ind2": [10, 20, 30, 40],
        "ind3": [100, 200, 300, 400],
        "ind4": [1000, 2000, 3000, 4000],
        "ind5": [10000, 20000, 30000, 40000],
    }, index=pd.date_range("2023-01-01", periods=4))
    mock_calculator.calculate_indicators.return_value = mock_calculated_data

    monkeypatch.setattr("src.trader3000.core.data_management.IndicatorsCalculator", MagicMock(return_value=mock_calculator))

    # Мокаем to_sql метода у DataFrame
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())

    # Мокаем sqlalchemy.create_engine и его соединение
    mock_con = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_con
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        dm.recalculate_indicators(mode="total")

    # Проверяем, что to_sql вызывался с правильными параметрами
    assert pd.DataFrame.to_sql.call_count >= 2  # минимум 2 вызова в цикле по TRESHOLDS

    # Можно дополнительно проверить корректность вызова, например:
    called_args = pd.DataFrame.to_sql.call_args
    assert called_args.kwargs["con"] == mock_con
    assert called_args.kwargs["if_exists"] == "replace"
    assert called_args.kwargs["index"] is True


def test_get_datasets_returns_df(monkeypatch, dm):
    # Мокируем get_sql_select_candles, чтобы возвращал фиктивный SQL
    monkeypatch.setattr(dm, "get_sql_select_candles", lambda *args, **kwargs: "SELECT 1")

    # Мокируем pd.read_sql чтобы возвращать минимальный DataFrame с колонкой 'date'
    monkeypatch.setattr(pd, "read_sql", lambda sql, con: pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "value": [1, 2]
    }))

    # Вызываем метод с обязательными аргументами ticker и resolution
    df_dict = dm.get_datasets("SBER", "hour")

    # Проверяем, что вернулся словарь с нужными ключами и DataFrame
    expected_keys_test = {"train_cost_history", "test_cost_history", "validation_cost_history",
                         "train_indicators", "test_indicators", "validation_indicators"}
    assert set(df_dict.keys()) == expected_keys_test

    # Проверяем, что каждый элемент словаря — DataFrame и не пустой
    for df in df_dict.values():
        assert not df.empty
        assert "date" in df.index.names or "date" in df.columns


def test_get_predictor_datasets(monkeypatch, dm):
    # Подготовка моков для get_datasets с необходимой структурой
    monkeypatch.setattr(dm, "get_datasets", MagicMock(return_value={
        "train_indicators": pd.DataFrame({"f1": [1, 2]}),
        "validation_indicators": pd.DataFrame({"f1": [3, 4]}),
        "train_cost_history": pd.DataFrame({"close": [10, 20]}),
        "validation_cost_history": pd.DataFrame({"close": [30, 40]}),
    }))

    # Корректный вызов метода с обязательными параметрами
    ds_train_x, ds_train_y, ds_val_x, ds_val_y = dm.get_predictor_datasets(
        ticker="SBER",
        resolution="hour",
        mode="test",
        normalized=False
    )

    # Проверки возвращаемых numpy массивов
    assert (ds_train_x == np.array([[1],[2]])).all()
    assert (ds_val_x == np.array([[3],[4]])).all()
    assert (ds_train_y == np.array([10, 20])).all()
    assert (ds_val_y == np.array([30, 40])).all()


def test_update_normalized_datasets(monkeypatch, dm):
    # Замокать get_indicators_tables, чтобы вернуть список таблиц
    monkeypatch.setattr(dm, "get_indicators_tables", MagicMock(return_value=["indicators_sber"]))

    # Замокать common_sql_req, чтобы вернуть DataFrame с данными
    df = pd.DataFrame({
        "date": [1, 2, 3],
        "col1": [10, -5, 0],
        "col2": [2, 0, 4]
    })
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=df.copy()))

    # Замокать pd.DataFrame.to_sql, чтобы не писать в базу
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())

    # Замокать create_engine и connect
    mock_con = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_con
    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        dm.update_normalized_datasets()

    # Проверить, что to_sql вызван
    assert pd.DataFrame.to_sql.called

    # Проверить, что таблица сохранилась с суффиксом "_normalized"
    args, kwargs = pd.DataFrame.to_sql.call_args
    assert kwargs["name"].endswith("_normalized")
    assert kwargs["con"] == mock_con
    assert kwargs["if_exists"] == "replace"


def test_get_schedules_and_update(monkeypatch, dm):
    # Мокаем trading_manager.get_trading_schedules с необходимой структурой
    class FakeDay:
        def __init__(self, is_trading_day, date, start_time, end_time):
            self.is_trading_day = is_trading_day
            self.date = date
            self.start_time = start_time
            self.end_time = end_time

    class FakeSchedule:
        def __init__(self, exchange, days):
            self.exchange = exchange
            self.days = days

    fake_days = [
        FakeDay(True, pd.Timestamp("2025-11-23"), pd.Timestamp("2025-11-23 09:00"), pd.Timestamp("2025-11-23 17:00")),
        FakeDay(False, pd.Timestamp("2025-11-24"), pd.Timestamp("2025-11-24 09:00"), pd.Timestamp("2025-11-24 17:00")),
    ]
    fake_schedule = FakeSchedule("EX1", fake_days)

    dm.trading_manager.get_trading_schedules = MagicMock(return_value=[fake_schedule])

    # Мокаем метод seconds_between_dt (если он есть)
    dm.seconds_between_dt = lambda start, end: (end - start).total_seconds()

    # Мокаем pd.DataFrame.to_sql
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())

    # Вызываем метод
    result = dm.get_schedules()

    # Проверяем, что метод ничего не возвращает
    assert result is None

    # Проверяем, что to_sql вызван
    assert pd.DataFrame.to_sql.called

def test_get_target_sell_dt(monkeypatch, dm):
    # Мокаем common_sql_req через pd.read_sql с нужными колонками
    monkeypatch.setattr(
        pd,
        "read_sql",
        lambda sql, con: pd.DataFrame({
            "work_seconds": [3600, 3600],
            "start_time": [
                datetime.datetime(2025, 11, 24, 9, 0),
                datetime.datetime(2025, 11, 25, 9, 0)
            ]
        })
    )

    monkeypatch.setattr(
        dm,
        "get_timeline",
        lambda schedule, timestep_seconds: [
            datetime.datetime(2025, 11, 24, 10, 0),  # step 0
            datetime.datetime(2025, 11, 24, 11, 0),  # step 1
            datetime.datetime(2025, 11, 24, 12, 0),  # step 2
            # удалить или изменить 4-й элемент, чтобы не был возвращён
        ]
    )

    schedule_name = "SBER"
    buy_dt = datetime.datetime(2025, 11, 24, 10, 0)
    resolution = "hour"
    steps = 2

    result = dm.get_target_sell_dt(schedule_name, buy_dt, resolution, steps)

    assert result == datetime.datetime(2025, 11, 24, 12, 0)


def test_is_trading_day_check(monkeypatch, dm):
    # Мокаем common_sql_req через pd.read_sql, возвращая DataFrame с данными
    monkeypatch.setattr(pd, "read_sql", lambda sql, con: pd.DataFrame({"is_open": [True]}))

    # Вызываем метод с корректным именем расписания
    result = dm.is_trading_day("SBER")

    # Проверяем, что метод возвращает True
    assert result is True


def test_get_tickers_from_scope_table_exists(monkeypatch, dm):
    monkeypatch.setattr(dm, "check_if_table_exists", MagicMock(return_value=True))
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})))

    tickers = dm.get_tickers_from_scope()

    assert tickers == ["AAA", "BBB", "CCC"]

def test_get_tickers_from_scope_table_missing(monkeypatch, dm):
    monkeypatch.setattr(dm, "check_if_table_exists", MagicMock(side_effect=[False, True]))
    monkeypatch.setattr(dm, "create_tickers_scope", MagicMock())

    # Возвращаем DataFrame, а не словарь
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=pd.DataFrame({"ticker": ["DDD", "EEE"]})))

    tickers = dm.get_tickers_from_scope()

    assert tickers == ["DDD", "EEE"]
    assert dm.check_if_table_exists.call_count == 2
    dm.create_tickers_scope.assert_called_once()


def test_create_tickers_scope(monkeypatch, dm):
    # Мокируем get_table_names_by_schema, чтобы вернуть фиктивный список таблиц
    monkeypatch.setattr(dm, "get_table_names_by_schema", MagicMock(return_value=[
        "candles_sber_hour",
        "candles_aapl_hour",
        "candles_goog_hour"
    ]))

    # Мокаем to_sql, чтобы не писать в БД
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())

    mock_con = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_con

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        tickers_df = dm.create_tickers_scope()

    # Проверка, что to_sql вызван
    assert pd.DataFrame.to_sql.called

    # Проверка, что в df есть ожидаемые тикеры без префиксов и суффиксов
    expected_tickers = {"sber", "aapl", "goog"}
    assert set(tickers_df["ticker"]) == expected_tickers

def test_del_ticker_from_scope(monkeypatch, dm):
    # Мокаем get_tickers_from_scope, возвращая список
    monkeypatch.setattr(dm, "get_tickers_from_scope", MagicMock(return_value=["SBER", "AAPL"]))

    # Мокаем to_sql, чтобы не делать реальные записи
    monkeypatch.setattr(pd.DataFrame, "to_sql", MagicMock())

    mock_con = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_con

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        dm.del_ticker_from_scope("SBER")

    # Проверяем, что to_sql был вызван (для обновления таблицы)
    assert pd.DataFrame.to_sql.called

def test_get_all_features_names_with_sets(monkeypatch, dm):
    # Мокаем common_sql_req, чтобы вернуть DataFrame с колонками (первая колонка - дата)
    mock_columns = ["date", "feature1", "feature2", "feature3"]
    mock_df = pd.DataFrame(columns=mock_columns)

    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=mock_df))

    features = dm.get_all_features_names_with_sets()

    # Ожидаем все колонки, кроме первой
    expected_features = mock_columns[1:]
    assert features == expected_features

def test_get_all_features_names_with_sets_empty(monkeypatch, dm):
    # Мокаем common_sql_req, чтобы вернуть None (имитация отсутствия данных)
    monkeypatch.setattr(dm, "common_sql_req", MagicMock(return_value=None))

    features = dm.get_all_features_names_with_sets()

    # Ожидаем пустой список
    assert features == []

def test_get_all_features_names(monkeypatch, dm):
    # Мокаем get_all_features_names_with_sets, возвращая список с разными префиксами
    monkeypatch.setattr(
        dm,
        "get_all_features_names_with_sets",
        MagicMock(return_value=[
            "feature1",
            "set_feature2",
            "feature3",
            "set_feature4",
            "feature5",
        ])
    )

    filtered_features = dm.get_all_features_names()

    expected_features = ["feature1", "feature3", "feature5"]
    assert filtered_features == expected_features

def test_get_last_update_date(monkeypatch, dm):
    # Создаем мок коннекции и результата
    mock_fetchone = MagicMock(return_value=(datetime.datetime(2025, 11, 25, 10, 0),))
    mock_execute = MagicMock()
    mock_execute.fetchone = mock_fetchone
    mock_con = MagicMock()
    mock_con.execute.return_value = mock_execute

    ticker = "SBER"

    # Вызываем метод с мок коннекцией
    result = dm.get_last_update_date(ticker, con=mock_con)

    # Проверка правильности выполнения SQL и возврата
    expected_sql = f"SELECT max(date) FROM candles.{ticker.lower()}_hour"
    mock_con.execute.assert_called_once()
    args, _ = mock_con.execute.call_args
    assert isinstance(args[0], type(text("")))  # Первый аргумент - SQL выражение
    
    assert result == datetime.datetime(2025, 11, 25, 10, 0)

def test_get_last_update_date_none_result(monkeypatch, dm):
    mock_execute = MagicMock()
    mock_execute.fetchone = MagicMock(return_value=None)
    mock_con = MagicMock()
    mock_con.execute.return_value = mock_execute

    ticker = "SBER"
    result = dm.get_last_update_date(ticker, con=mock_con)
    assert result is None

def test_get_last_update_date_none_date(monkeypatch, dm):
    mock_execute = MagicMock()
    mock_execute.fetchone = MagicMock(return_value=(None,))
    mock_con = MagicMock()
    mock_con.execute.return_value = mock_execute

    ticker = "SBER"
    result = dm.get_last_update_date(ticker, con=mock_con)
    assert result is None

def test_get_last_update_date_no_con(monkeypatch, dm):
    ticker = "SBER"
    import pytest
    with pytest.raises(ValueError):
        dm.get_last_update_date(ticker, con=None)