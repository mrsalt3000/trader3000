import datetime
import os
import random
import time
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from sqlalchemy import text
from zoneinfo import ZoneInfo

from .indicators_calculator import IndicatorsCalculator

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound="DataManager")

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
else:
    Connection = object  # заглушка для runtime


def with_connection(func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = cast(DataManager, args[0])
        engine = sqlalchemy.create_engine(self.psql_main_creds)
        con = engine.connect()
        try:
            result = func(*args, con=con, **kwargs)  # type: ignore[arg-type]
            con.commit()
            return result
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    return wrapper


load_dotenv()  # Загружаем переменные окружения из .env


class DataManager:
    def __init__(
        self,
        trading_manager=None,
    ):
        self.CANDLES_SART_DATE = datetime.datetime(2019, 1, 1)  # 2019
        self.trading_manager = trading_manager
        self.psql_main_creds = os.getenv("DATABASE_URL")
        self.TZ = os.getenv("TZ")

    def get_sql_table_info(self, table_name):
        return f"""
        SELECT
            count(*) AS rows_count
            ,min(date) AS min_date
            ,max(date) AS max_date
        FROM candles.{table_name}
        """

    @with_connection
    def get_tables_info(
        self, con: Optional["sqlalchemy.engine.Connection"] = None
    ) -> pd.DataFrame:
        """
        Получить информацию о таблицах из схемы 'candles' в базе данных.

        Выполняет SQL-запрос к information_schema, чтобы получить список таблиц типа BASE TABLE,
        затем загружает информацию по каждой таблице и собирает её в один DataFrame.

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.

        Returns:
            pd.DataFrame: DataFrame с информацией по всем таблицам схемы 'candles', с индексами сброшенными.
        """
        sql = """
                SELECT
                    table_name
                FROM information_schema.tables
                WHERE table_type ='BASE TABLE'
                    AND table_schema IN ('candles')
            """
        table_names = sorted(pd.read_sql(sql, con)["table_name"].values)

        data = pd.DataFrame()
        for table_name in table_names:
            info = pd.read_sql(self.get_sql_table_info(table_name), con)
            data = pd.concat([data, info])
        return pd.DataFrame(data).reset_index(drop=True)
    
    def atom_get_market_trading_history(self, ticker,start_date,finish_date,time_step_resolution,):
        n_retries = 5
        s_wait = 30
        for i in range(n_retries):
            data = self.trading_manager.get_market_trading_history(
                ticker=ticker,
                start_date=start_date,
                finish_date=finish_date,
                time_step_resolution=time_step_resolution,
            )
            if data is not None and len(data) > 0:
                return data
            else:
                print("atom_get_market_trading_history: no data, retry:", i+1)
                time.sleep(s_wait)
        return None
    
    @with_connection
    def training_datasets_download(
        self,
        tickers: list[str],
        time_step_resolutions: list[str],
        con: Optional["sqlalchemy.engine.Connection"] = None,
        start_date: datetime.datetime = datetime.datetime(2019, 1, 1),
        finish_date: datetime.datetime | None = None,
        save_mode: str = "replace",
    ) -> None:
        """
        Загружает тренировочные наборы данных по указанным тикерам и временным разрешениям, сохраняя их в базу данных.

        Для каждого тикера и временного разрешения скачивается история торгов. Если данные получены,
        они сохраняются в таблицы схемы 'candles' в базе данных. При необходимости выполняется пауза между загрузками.

        Args:
            tickers (list[str]): Список тикеров для загрузки данных.
            time_step_resolutions (list[str]): Список временных разрешений (например, '1min', '1h').
            start_date (datetime.datetime, optional): Начальная дата для загрузки данных. По умолчанию 2019-01-01.
            finish_date (datetime.datetime | None, optional): Конечная дата для загрузки данных. Если None, загрузка до текущей даты.
            save_mode (str, optional): Режим сохранения данных в базу ('replace', 'append' и т.п.). По умолчанию 'replace'.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных, создается декоратором.

        Returns:
            None: Метод не возвращает значения.
        """
        for ticker in tickers:
            print(ticker)
            if "@" not in ticker and "-" not in ticker:
                for time_step_resolution in time_step_resolutions:
                    data = self.atom_get_market_trading_history(
                        ticker=ticker,
                        start_date=start_date,
                        finish_date=finish_date,
                        time_step_resolution=time_step_resolution,
                    )

                    if data is not None and len(data) > 0:
                        print(len(data), "rows downloaded")
                        table_name = f"{ticker.lower()}_{time_step_resolution}"
                        data.to_sql(
                            con=con,
                            name=table_name,
                            schema="candles",
                            if_exists=save_mode,
                            index=False,
                        )
                        if (len(tickers) + len(time_step_resolutions)) > 2:
                            time.sleep(60)
                    else:
                        # Проверим дату последнего обновления по тикеру
                        last_update_date = self.get_last_update_date(ticker)
                        if last_update_date is None:
                            # Если даты нет — удаляем
                            delete = True
                        else:
                            days_since_update = (
                                datetime.datetime.now() - last_update_date
                            ).days
                            delete = (
                                days_since_update > 30
                            )  # больше месяца нет обновлений
                        if delete:
                            print(
                                f"No updates for {ticker} for over 30 days, deleting data"
                            )
                            self.del_ticker_from_scope(ticker.upper())
                            self.drop_tables_by_ticker(ticker.lower())
                        else:
                            if last_update_date is not None:
                                print(
                                    f"Ticker {ticker} has recent updates ({days_since_update} days ago), skipping deletion"
                                )
        print("data loaded")
        return None

    @with_connection
    def common_sql_req(
        self,
        sql: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
        n_attempts: int = 4,
    ) -> pd.DataFrame | None:
        """
        Выполняет SQL-запрос к базе данных с несколькими попытками в случае ошибки.

        Пытается выполнить заданный SQL-запрос через подключение к базе данных,
        повторяя попытку до `n_attempts` раз при возникновении исключений с таймаутом 5 секунд между попытками.

        Args:
            sql (str): SQL-запрос для выполнения.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных, передается декоратором.
            n_attempts (int, optional): Максимальное число попыток выполнения запроса. По умолчанию 4.

        Returns:
            pd.DataFrame | None: Результат выполнения SQL-запроса в виде DataFrame, либо None, если все попытки не удались.
        """
        counter = 1
        for _ in range(n_attempts):
            try:
                data = pd.read_sql(sql, con)
                return data
            except Exception as error:
                print(error)
                counter += 1
                print("attempt №", counter)
                time.sleep(5)
        return None

    @with_connection
    def common_sql_comand(
        self,
        sql: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Выполняет SQL-команду в базе данных с одной попыткой и обработкой исключений.

        Пытается выполнить SQL-команду через переданное подключение. В случае ошибки
        выводит сообщение об ошибке

        Args:
            sql (str): SQL-команда для выполнения.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных, передается декоратором.

        Returns:
            None: Метод не возвращает значения.
        """
        try:
            pd.read_sql(sql, con)
            print("common_sql_command:",sql, "applied")
        except Exception as error:
            print(error)
            #time.sleep(10)

    @with_connection
    def get_all_candles_table_names(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> list[str]:
        """
        Получить список имен всех таблиц схемы 'candles' в базе данных.

        Выполняет SQL-запрос к information_schema для выбора всех базовых таблиц,
        принадлежащих схеме 'candles', и возвращает их имена.

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.

        Returns:
            list[str]: Список имен таблиц схемы 'candles'.
        """
        sql = """
                SELECT
                    table_name AS name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_schema = 'candles'
            """
        data = pd.read_sql(sql, con)["name"].values
        return data

    @with_connection
    def recalculate_indicators(
        self,
        mode: str = "update",
        tickers = [],
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Пересчитывает и обновляет индикаторы для наборов данных свечей в базе.

        В режиме "total" выполняется полный перерасчет и запись индикаторов по временным диапазонам
        (train, test, validation, total) для всех свечных таблиц.
        В режиме "update" перерасчет и обновление выполняется на последних 1000 записях,
        объединяя новые данные со старыми.

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.
            mode (str, optional): Режим работы, один из ["update", "total"]. По умолчанию "total".

        Returns:
            None: Метод не возвращает значения.
        """
        print("recalculating indicators","mode:",mode)
        if len(tickers)==0:
            candles_tables = self.get_all_candles_table_names()
        else:
            candles_tables = [
                table for ticker in tickers 
                 for table in self.get_table_names_by_key(ticker, "candles")
            ]

        for candles_table in candles_tables:
            print(candles_table)
            candles = self.common_sql_req(
                sql=f'SELECT DISTINCT * FROM candles."{candles_table}" ORDER BY date DESC'
            )
            if candles is None:
                continue
            candles.drop_duplicates(inplace=True)
            indicators_calculator = IndicatorsCalculator(candles)
            #print("max(indicators_calculator.data[date]):",max(indicators_calculator.data.index))
            calculated_data = indicators_calculator.calculate_indicators()
            #print("max(calculated_data.index):",max(calculated_data.index))
            calculated_data_columns = calculated_data.columns
            calculated_data.index = pd.to_datetime(calculated_data.index)

            dataset_type = "total"
            
            if mode == "total":
                start = datetime.datetime(1900, 1, 1, 0, 0)
                if_exists = "replace"
            elif mode == "update":
                ticker = candles_table.split("_")[0]
                last_saved_candle =  self.get_last_update_date(ticker)
                start = last_saved_candle + datetime.timedelta(hours=1)
                if_exists = "append"

            table_name = f"{candles_table}_{dataset_type}_cost_history"
            slice_ = calculated_data.loc[
                start:, calculated_data_columns[:5]  # type: ignore
            ]
            slice_.index = slice_.index.strftime('%Y-%m-%d %H:%M:%S')
            slice_.index.name = "date"
            slice_.to_sql(
                con = con,
                schema = "datasets",
                name = table_name,
                if_exists = if_exists,
                index = True,
            )

            table_name = f"{candles_table}_{dataset_type}_indicators"
            slice_ = calculated_data.loc[
                start:, calculated_data_columns[5:]  # type: ignore
            ]
            slice_.index = slice_.index.strftime('%Y-%m-%d %H:%M:%S')
            slice_.index.name = "date"
            slice_.to_sql(
                con = con,
                schema = "datasets",
                name = table_name,
                if_exists = if_exists,
                index = True,
            )

    @with_connection
    def get_trading_log(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> pd.DataFrame:
        """
        Получить полный лог торговых операций из таблицы trading_log.

        Выполняет SQL-запрос для выборки всех записей из таблицы trading_log
        и возвращает результат в виде DataFrame.

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.

        Returns:
            pd.DataFrame: DataFrame с данными из таблицы trading_log.
        """
        sql = """
                SELECT
                    *
                FROM trading_log
    
            """
        data = pd.read_sql(sql, con)
        return data

    def get_sql_select_candles(
        self,
        ticker: str,
        resolution: str,
        dataset_type: str,
        part: str,
        normalized: bool = False,
    ) -> str:
        """
        Формирует SQL-запрос для выбора всех данных из таблицы свечей с заданными параметрами.

        Конструирует имя таблицы на основе тикера, разрешения времени, типа набора данных,
        части данных и флага нормализации, затем возвращает SQL-запрос для выборки всех данных из этой таблицы.

        Args:
            ticker (str): Биржевой тикер, например 'AAPL'.
            resolution (str): Временное разрешение данных, например '1min', '1h'.
            dataset_type (str): Тип набора данных, например 'train', 'test'.
            part (str): Часть данных, например 'cost_history', 'indicators'.
            normalized (bool, optional): Флаг, указывающий на выбор нормализованных данных. По умолчанию False.

        Returns:
            str: SQL-запрос для выборки всех данных из сформированной таблицы.
        """
        table_name = f"datasets_{ticker}_{resolution}_{dataset_type}_{part}"
        if normalized:
            table_name = table_name + "_normalized"
        sql = f"""
        SELECT
            *
        FROM {table_name}
        """
        return sql

    @with_connection
    def get_datasets(
        self,
        ticker: str,
        resolution: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
        mode: str = "test",
        normalized: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Загружает датасеты для указанного тикера и разрешения из базы данных.

        В режиме "test" возвращает наборы для train, test и validation, в режиме "prod"
        загружает только train и validation. Поддерживается загрузка нормализованных показателей.

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.
            ticker (str): Биржевой тикер или "multy" для загрузки множества тикеров.
            resolution (str): Временное разрешение данных, например 'hour'.
            mode (str, optional): Режим загрузки: "test" или "prod". По умолчанию "test".
            normalized (bool, optional): Если True, загружаются нормализованные индикаторы. По умолчанию False.

        Returns:
            dict[str, pd.DataFrame]: Словарь датасетов с ключами вида "{type}_{key}", где type - train/test/validation,
                                    key - cost_history или indicators.
        """
        datasets = {}
        if ticker.lower() != "multy":
            tickers = [ticker]
        else:
            tickers = self.get_tickers_from_datasets(resolution="hour")

        if mode == "test":
            for type in ["train", "test", "validation"]:
                for key in ["cost_history", "indicators"]:
                    datasets[f"{type}_{key}"] = pd.DataFrame()
            for ticker_ in tickers:
                for type in ["train", "test", "validation"]:
                    sql = self.get_sql_select_candles(
                        ticker_, resolution, type, "cost_history"
                    )
                    datasets[f"{type}_cost_history"] = pd.concat(
                        [
                            datasets[f"{type}_cost_history"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )
                    sql = self.get_sql_select_candles(
                        ticker_, resolution, type, "indicators", normalized
                    )
                    datasets[f"{type}_indicators"] = pd.concat(
                        [
                            datasets[f"{type}_indicators"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )

        elif mode == "prod":
            for type in ["train", "validation"]:
                for key in ["cost_history", "indicators"]:
                    datasets[f"{type}_{key}"] = pd.DataFrame()
            for ticker_ in tickers:
                for type in ["train", "test"]:
                    sql = self.get_sql_select_candles(
                        ticker_, resolution, type, "cost_history"
                    )
                    datasets["train_cost_history"] = pd.concat(
                        [
                            datasets["train_cost_history"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )
                    sql = self.get_sql_select_candles(
                        ticker_, resolution, type, "indicators", normalized
                    )
                    datasets["train_indicators"] = pd.concat(
                        [
                            datasets["train_indicators"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )

                    sql = self.get_sql_select_candles(
                        ticker_, resolution, "validation", "cost_history"
                    )
                    datasets["validation_cost_history"] = pd.concat(
                        [
                            datasets["validation_cost_history"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )
                    sql = self.get_sql_select_candles(
                        ticker_,
                        resolution,
                        "validation",
                        "indicators",
                        normalized,
                    )
                    datasets["validation_indicators"] = pd.concat(
                        [
                            datasets["validation_indicators"],
                            pd.read_sql(sql, con).set_index("date"),
                        ]
                    )
        return datasets

    def get_candles_last_dt(
        self,
        ticker: str = "SBER",
        time_step_resolution: str = "hour",
    ) -> datetime.datetime:
        """
        Получить дату и время последней свечи в таблице с указанным тикером и разрешением.

        Выполняет SQL-запрос для получения максимальной даты из таблицы соответствующей свечи,
        затем преобразует строку в объект datetime с UTC таймзоной.

        Args:
            ticker (str, optional): Биржевой тикер, по умолчанию "SBER".
            time_step_resolution (str, optional): Временное разрешение свечи, например "hour", по умолчанию "hour".

        Returns:
            datetime.datetime: Дата и время последней свечи с учетом часового пояса UTC.
        """
        sql = f"""
        CREATE SCHEMA IF NOT EXISTS candles;
        CREATE TABLE IF NOT EXISTS candles.{ticker.lower()}_{time_step_resolution} (

            date   TEXT,
            open   FLOAT,
            high   FLOAT,
            low    FLOAT,
            close  FLOAT,
            volume FLOAT
        );
        SELECT
            max(date) AS ans
        FROM candles.{ticker.lower()}_{time_step_resolution}
        """
        result = self.common_sql_req(sql)
        if result is None or result.empty or result["ans"].isnull().all():
            last_dt = datetime.datetime.strptime(
                "2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
        else:
            last_dt_str = result["ans"].values[0]
            last_dt = datetime.datetime.strptime(
                last_dt_str, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
        return last_dt

    def download_candles_delta(
        self,
        tickers: list[str] = ["SBER"],
        time_step_resolutions: list[str] = ["hour"],
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Загружает и добавляет в базу новые свечи для указанных тикеров с момента последней записи.

        Для каждого тикера и временного разрешения определяется самая свежая дата в таблице,
        после чего запрашиваются только недостающие данные у трейдинг-менеджера и выполняется их загрузка в базу по режиму "append".

        Args:
            tickers (list[str], optional): Список тикеров для обновления. По умолчанию ["SBER"].
            time_step_resolutions (list[str], optional): Список разрешений свечей ("hour", "15min" и т.д.). По умолчанию ["hour"].

        Returns:
            None: Метод не возвращает значения.
        """
        print("\ndownload_candles_delta len(tickers)", len(tickers))
        for ticker in tickers:
            for time_step_resolution in time_step_resolutions:
                if time_step_resolution == "hour":
                    time_delta = datetime.timedelta(hours=1)
                elif time_step_resolution == "15min":
                    time_delta = datetime.timedelta(minutes=15)
                start_dt = (
                    self.get_candles_last_dt(ticker, time_step_resolution)
                    + time_delta
                )
                try:
                    self.training_datasets_download(
                        tickers=[ticker],
                        time_step_resolutions=[time_step_resolution],
                        start_date=start_dt,
                        finish_date=datetime.datetime.now().replace(
                            tzinfo=datetime.timezone.utc
                        ),
                        save_mode="append",
                    )
                    break  # успешное завершение — выходим из цикла
                except Exception as e:
                    if (
                        hasattr(e, "args")
                        and len(e.args) > 0
                        and "RESOURCE_EXHAUSTED" in str(e.args[0])
                    ):
                        print(
                            "Превышен лимит запросов. Ожидание 30 секунд перед повтором..."
                        )
                        time.sleep(30)
                    else:
                        raise

    @with_connection
    def get_schedules(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
        from_: datetime.datetime = datetime.datetime.now().replace(
            tzinfo=datetime.timezone.utc
        ),
        to: datetime.datetime = datetime.datetime.now().replace(
            tzinfo=datetime.timezone.utc
        )
        + datetime.timedelta(days=1),
    ) -> None:
        """
        Получает торговые расписания за указанный период, формирует DataFrame и сохраняет в базу.

        Вызывает метод торгового менеджера для получения расписаний, фильтрует торговые дни,
        вычисляет количество рабочих секунд, собирает данные в DataFrame и записывает их в таблицу базы.

        Args:
            from_ (datetime.datetime, optional): Начальная дата и время периода. По умолчанию текущий момент UTC.
            to (datetime.datetime, optional): Конечная дата и время периода. По умолчанию через сутки от текущего момента UTC.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            None: Метод не возвращает значения.
        """
        schedules_raw = self.trading_manager.get_trading_schedules(
            from_=from_, to=to
        )
        schedules = []
        for schedule_raw in schedules_raw:
            schedule_name = schedule_raw.exchange
            for day in schedule_raw.days:
                if day.is_trading_day:
                    row = {
                        "schedule": schedule_name,
                        "date": day.date.date(),
                        "is_trading_day": int(day.is_trading_day),
                        "start_time": day.start_time,  # .replace(tzinfo=None)
                        "end_time": day.end_time,  # .replace(tzinfo=None)
                        "work_seconds": self.seconds_between_dt(
                            day.start_time, day.end_time
                        ),
                    }
                    schedules.append(row)
        schedules = pd.DataFrame(schedules)
        schedules.to_sql(
            con=con,
            name="schedules",
            schema="trading",
            if_exists="append",
            index=False,
        )

    def update_schedules(self) -> None:
        """
        Обновляет таблицу торговых расписаний начиная с последней сохраненной даты.

        Определяет максимальную дату, для которой уже сохранено расписание в базе,
        затем последовательно загружает расписания на следующие 13 дней, сохраняя их в базу.
        В случае ошибки выводит сообщение и прекращает обновление.

        Returns:
            None: Метод не возвращает значения.
        """
        now = str(datetime.datetime.now())[:10]
        self.common_sql_comand(
            f"DELETE FROM trading.schedules WHERE date > '{now}'"
        )
        start_date = datetime.datetime.now()
        for i in range(13):
            try:
                start_date_ = start_date + datetime.timedelta(days=i)#
                final_date = start_date_ + datetime.timedelta(days=1)
                print(
                    "start_date",
                    start_date_,
                    "final_date",
                    final_date,
                )
                self.get_schedules(
                    from_=start_date_.replace(tzinfo=datetime.timezone.utc),
                    to=final_date.replace(tzinfo=datetime.timezone.utc),
                )
            except Exception as e:
                print(f"Error: \n{e}")
                break

    def get_last_timestep_data(
        self,
        ticker: str,
        time_step_resolution: str,
        window_size: int,
    ) -> Tuple[List[float], str]:
        """
        Получить данные по индикаторам за последние временные шаги для указанного тикера.

        Выполняет SQL-запрос к таблице валидированных индикаторов,
        выбирает последние (window_size + 1) записей, выделяет метку последнего временного шага и формирует плоский список наблюдений.

        Args:
            ticker (str): Биржевой тикер, например "SBER".
            time_step_resolution (str): Временное разрешение данных, например "hour".
            window_size (int): Размер окна для выборки данных (количество временных шагов).

        Returns:
            Tuple[List[float], str]: Кортеж из списка наблюдений (слева направо, по всем индикаторам) и метки (даты) последнего временного шага.
        """
        sql = f"""
            SELECT
                *
            FROM datasets_{ticker.lower()}_{time_step_resolution}_validation_indicators
            ORDER BY date DESC
            LIMIT {window_size+1}
            """
        result = self.common_sql_req(sql)
        if result is None:
            # Обработка отсутствия данных:
            raise ValueError("Query returned no data")
        data = result.values
        last_timestep = data[0][0]
        data = [d[1:] for d in data]
        observation = list(chain(*data))

        return observation, last_timestep

    @with_connection
    def save_transaction(
        self,
        transaction: dict,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Сохраняет одну запись транзакции в таблицу базы данных.

        Принимает словарь с данными транзакции, формирует DataFrame из одной строки
        и добавляет её в таблицу "transactions" схемы "transaction".

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.
            transaction (dict): Словарь с данными транзакции, ключи соответствуют колонкам таблицы.

        Returns:
            None: Метод не возвращает значения.
        """
        columns = [i for i in transaction.keys()]
        values = [[i for i in transaction.values()]]
        df = pd.DataFrame(data=values, columns=columns)
        df.to_sql(
            con=con,
            schema="transaction",
            name="transactions",
            if_exists="append",
            index=False,
        )

    def get_last_price(
        self,
        ticker: str,
        resolution: str,
    ) -> float:
        """
        Получить последнюю цену закрытия свечи для указанного тикера и временного разрешения.

        Выполняет SQL-запрос к таблице свечей, сортирует по дате в обратном порядке,
        возвращает значение закрытия самой последней свечи.

        Args:
            ticker (str): Биржевой тикер, например "SBER".
            resolution (str): Временное разрешение свечи, например "hour".

        Returns:
            float: Последняя цена закрытия.
        """
        sql = f"""
            SELECT
                close
            FROM candles.{ticker}_{resolution}
            ORDER BY date DESC 
            LIMIT 1
            """
        result = self.common_sql_req(sql)
        if result is None or result.empty:
            raise ValueError(f"No data returned for query: {sql}")
        return result["close"].values[0]

    def get_transactions(self) -> pd.DataFrame:
        """
        Получить все записи транзакций, отсортированные по дате покупки в порядке убывания.

        Выполняет SQL-запрос, который выбирает все колонки из таблицы "transactions",
        сортирует результат по полю buy_dt по убыванию и возвращает как DataFrame.

        Returns:
            pd.DataFrame: DataFrame с информацией о всех транзакциях, отсортированный по дате покупки.
        """
        sql = """
            SELECT
                *
            FROM transactions
            ORDER BY buy_dt DESC 
            """
        return self.common_sql_req(sql)

    def seconds_between_dt(
        self,
        start_dt: datetime.datetime,
        finish_dt: datetime.datetime,
    ) -> int:
        """
        Вычисляет количество секунд между двумя временными отметками.

        Вычисляет разницу между конечной и начальной датами и возвращает результат в секундах.

        Args:
            start_dt (datetime.datetime): Начальная дата и время.
            finish_dt (datetime.datetime): Конечная дата и время.

        Returns:
            int: Количество секунд между start_dt и finish_dt.
        """
        date_diff_raw = finish_dt - start_dt
        date_diff_seconds = (
            date_diff_raw.days * 24 * 3600 + date_diff_raw.seconds
        )
        return date_diff_seconds

    def get_timeline(
        self,
        schedule: pd.DataFrame,
        timestep_seconds: int,
    ) -> List[datetime.datetime]:
        """
        Формирует линейку временных меток (таймлайна) на основе торгового расписания.

        Для каждого торгового дня из расписания вычисляет количество временных шагов,
        затем создает список datetime объектов с шагом timestep_seconds, начиная с времени начала сессии.

        Args:
            schedule (pd.DataFrame): DataFrame с торговым расписанием, содержащий колонки 'work_seconds' и 'start_time'.
            timestep_seconds (int): Длительность одного временного шага в секундах.

        Returns:
            List[datetime.datetime]: Список временных меток для всех временных шагов в расписании.
        """
        timeline = []
        for day in schedule.to_dict("records"):
            timesteps_within = day["work_seconds"] // timestep_seconds
            if isinstance(day, str):
                start_time = datetime.datetime.strptime(
                    day["start_time"][:19], "%Y-%m-%d %H:%M:%S"
                )
            else:
                start_time = day["start_time"]
            for t in range(timesteps_within + 1):
                item = (start_time
                    + datetime.timedelta(seconds=timestep_seconds * t)).replace(tzinfo=ZoneInfo("UTC"))
                timeline.append(item)
        return timeline

    def get_target_sell_dt(
        self,
        schedule_name: str,
        buy_dt: datetime.datetime,
        resolution: str,
        steps: int,
    ) -> datetime.datetime:
        """
        Вычисляет целевую дату и время продажи, отталкиваясь от даты покупки и торгового расписания.

        Определяет длительность одного временного шага по разрешению (час или 15 минут),
        загружает торговое расписание, создает временную шкалу и находит момент времени,
        который находится на заданном количестве шагов после даты покупки.
        Если такого шага нет, возвращает далекую будущую дату по умолчанию.

        Args:
            schedule_name (str): Имя торгового расписания.
            buy_dt (datetime.datetime): Дата и время покупки.
            resolution (str): Временное разрешение, например "hour" или "15min".
            steps (int): Количество временных шагов до продажи.

        Returns:
            datetime.datetime: Целевая дата и время продажи.
        """
        buy_dt = buy_dt.astimezone(ZoneInfo("UTC"))
        if resolution == "hour":
            timestep_seconds = 3600
        elif resolution == "15min":
            timestep_seconds = 15 * 60
        timestep_seconds * steps
        buy_date = buy_dt.date()

        sql = f"""
        SELECT DISTINCT
            *
        FROM trading.schedules
        WHERE schedule = '{schedule_name}'
            AND date >= '{str(buy_date)}'
        ORDER BY date
        """
        schedule = self.common_sql_req(sql)
        timeline = self.get_timeline(schedule, timestep_seconds)
        timeline = list(filter(lambda x: x >= buy_dt, timeline))
        if len(timeline) > steps:
            sell_date = timeline[steps - 1]
        else:
            sell_date = datetime.datetime(2100, 1, 1, 1)
        return sell_date

    def is_trading_day(
        self,
        schedule: str,
    ) -> bool:
        """
        Проверяет, является ли текущий день торговым днем по заданному торговому расписанию.

        Выполняет SQL-запрос, который проверяет наличие записи о торговом дне в таблице trading_schedules
        для указанного расписания и текущей даты.

        Args:
            schedule (str): Имя торгового расписания.

        Returns:
            bool: True, если сегодня торговый день, иначе False.
        """
        schedule = schedule.upper()
        today = str(datetime.datetime.now().date())
        sql = f"""
        SELECT
            *
        FROM trading.schedules
        WHERE schedule = '{schedule}'
        AND date = '{today}'
                """
        resp = self.common_sql_req(sql)
        if resp is None:
            return False
        return len(resp) > 0

    def get_indicators_tables(self) -> list[str]:
        """
        Получить список имен всех таблиц в схеме 'indicators'.

        Выполняет SQL-запрос к information_schema для получения всех таблиц, принадлежащих схеме 'indicators',
        и возвращает список их имен.

        Returns:
            list[str]: Список имен таблиц в схеме 'indicators'.
        """
        sql = """
                SELECT
                    *
                FROM information_schema.tables
                WHERE table_schema = 'datasets'
            """
        result = self.common_sql_req(sql)
        if result is None or result.empty:
            return []  # Возвращаем пустой список, если никаких таблиц нет
        tables = result["table_name"].values
        #tables= result
        return tables

    @with_connection
    def update_normalized_datasets(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Обновляет нормализованные версии датасетов индикаторов, масштабируя значения по максимальному модулю.

        Для каждой таблицы в схеме 'indicators' загружает все данные, нормализует значения каждой колонки
        (кроме первой) по максимальному абсолютному значению, а затем сохраняет результат в схему 'datasets'
        под именем с суффиксом "_normalized".

        Args:
            con (sqlalchemy.engine.Connection): Активное подключение к базе данных.

        Returns:
            None: Метод не возвращает значения.
        """
        tables = self.get_indicators_tables()
        for table_name in tables:
            sql = f"""
                    SELECT
                        *
                    FROM {table_name}
                """
            data = self.common_sql_req(sql)
            if data is None or data.empty:
                continue  # Пропускаем пустые или отсутствующие таблицы

            for col in data.columns[1:]:
                max_value = max(data[col].unique())
                if max_value == 0:
                    max_value = 1
                elif max_value < 0:
                    max_value *= -1
                data[col] = data[col].apply(lambda x: x / max_value)
            data.to_sql(
                con=con,
                schema="datasets",
                name=table_name + "_normalized",
                if_exists="replace",
                index=False,
            )

    @with_connection
    def get_tickers_from_datasets(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
        resolution: str = "hour",
    ) -> list[str]:
        """
        Извлекает список уникальных тикеров из имен таблиц в схеме 'datasets' для заданного временного разрешения.

        Фильтрует таблицы, имена которых начинаются с "datasets" и содержат указанный resolution,
        затем парсит тикер из имени таблиц и возвращает их уникальный список.

        Args:
            resolution (str, optional): Временное разрешение таблиц, например "hour". По умолчанию "hour".

        Returns:
            list[str]: Список уникальных тикеров, найденных в именах таблиц.
        """
        resp = self.get_tables_info()
        len_keyword = len("datasets")
        tables = list(
            filter(
                lambda x: x[:len_keyword] == "datasets"
                and f"_{resolution}_" in x,
                resp["table"].values,
            )
        )
        tickers = list(
            set(
                [
                    table[
                        len_keyword
                        + 1 : table[len_keyword + 1 :].find("_")
                        + len_keyword
                        + 1
                    ]
                    for table in tables
                ]
            )
        )
        return tickers

    @with_connection
    def get_predictor_datasets(
        self,
        ticker: str,
        resolution: str,
        mode: str,
        normalized: bool,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> tuple:
        """
        Загружает и готовит массивы признаков и целевых значений для модели на основе датасетов.

        Использует метод get_datasets для получения наборов train и validation,
        разделяет данные на индикаторы (признаки) и стоимость закрытия (целевые значения),
        возвращая numpy-массивы для обучения и валидации.

        Args:
            ticker (str): Биржевой тикер, например "SBER".
            resolution (str): Временное разрешение данных, например "hour".
            mode (str): Режим загрузки датасетов ("test", "prod" и т.п.).
            normalized (bool): Использовать ли нормализованные данные.

        Returns:
            tuple: Кортеж из четырех элементов:
                - ds_train_x (np.ndarray): Признаки для обучения.
                - ds_train_y (np.ndarray): Целевые значения для обучения.
                - ds_val_x (np.ndarray): Признаки для валидации.
                - ds_val_y (np.ndarray): Целевые значения для валидации.
        """
        datasets = self.get_datasets(
            ticker=ticker,
            resolution=resolution,
            mode=mode,
            normalized=normalized,
        )

        ds_train_x_raw = datasets["train_indicators"]
        ds_val_x_raw = datasets["validation_indicators"]

        ds_train_y_raw = datasets["train_cost_history"]["close"]
        ds_val_y_raw = datasets["validation_cost_history"]["close"]

        ds_train_x = ds_train_x_raw.values
        ds_val_x = ds_val_x_raw.values

        ds_train_y = ds_train_y_raw.values
        ds_val_y = ds_val_y_raw.values

        return ds_train_x, ds_train_y, ds_val_x, ds_val_y

    def add_window(
        self,
        x_in: np.ndarray,
        y_in: Optional[np.ndarray] = None,
        n_steps: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Преобразует входные данные временного ряда, создавая окна фиксированной длины для моделирования.

        Для каждого окна длиной n_steps создает одномерный массив из последовательных временных шагов входных данных.
        Если переданы целевые значения y_in, они усекаются на длину n_steps, чтобы соответствовать окнам.

        Args:
            x_in (np.ndarray): Входные данные, где каждая запись соответствует одному временному шагу.
            y_in (Optional[np.ndarray], optional): Целевые значения соответствующие входным данным. По умолчанию None.
            n_steps (int, optional): Длина окна (число временных шагов для каждого примера). По умолчанию 1.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - x_out: Массив окон размерности (число окон, окно * размер признаков).
                - y_out: Целевые значения, соответствующие окнам, либо None если y_in не передавался.
        """
        y_out = y_in[:-n_steps] if y_in is not None else None
        x_out = []
        for p in range(n_steps, len(x_in), 1):
            x_out.append(list(chain.from_iterable(x_in[p - n_steps : p])))
        return np.array(x_out), np.array(y_out)

    def get_orders_bought(self, account_id) -> np.ndarray:
        """
        Получить идентификаторы ордеров и оценку изменения цены для купленных ордеров.

        Выполняет SQL-запрос, который выбирает order_id и estimated_price_change из таблицы transactions,
        фильтруя записи по статусам исполнения (полное или частичное исполнение),
        направлению "покупка" и статусу "bought".

        Returns:
            np.ndarray: Массив значений с колонками order_id и estimated_price_change для соответствующих ордеров.
        """
        sql = f"""
        SELECT
            order_id,
            estimated_price_change
        FROM transaction.transactions
        WHERE execution_report_status IN ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'bought'
            AND account_id = '{account_id}'
        """
        result = self.common_sql_req(sql)
        if result is None or result.empty:
            # Возвращаем пустой numpy массив, если нет данных
            return np.empty((0, 2))
        return result.values

    def get_orders_sell_order_posted(self, account_id) -> np.ndarray:
        """
        Получить идентификаторы ордеров на продажу, статусы которых 'sell_order_posted' и направление 'ORDER_DIRECTION_BUY'.

        Выполняет SQL-запрос, выбирая поле sell_order_id из таблицы transactions,
        фильтруя записи по статусу и направлению.

        Returns:
            np.ndarray: Массив значений с колонкой sell_order_id для соответствующих ордеров.
        """
        sql = f"""
        SELECT
            sell_order_id
        FROM transaction.transactions 
        WHERE status = 'sell_order_posted'
            AND direction = 'ORDER_DIRECTION_BUY'
            AND account_id = '{account_id}'
        """
        result = self.common_sql_req(sql)
        if result is None or result.empty:
            return np.empty((0, 1))
        return result.values

    def get_orders_rancid(
        self,
        account_id
    ) -> list | np.ndarray:
        """
        Получить ордера, у которых истек предполагаемый срок продажи (просроченные ордера).

        Выполняет SQL-запрос для выбора ордеров с заполненным или частично заполненным статусом,
        направлением "покупка", статусом "take_profit_posted" и датой предполагаемой продажи
        не позже указанного порога.

        Args:
            days_treshold (int, optional): Количество дней для порога устаревания. По умолчанию 0 (сегодня).

        Returns:
            list | np.ndarray: Массив данных ордеров (order_id, buy_dt, lots_requested) либо пустой список,
                            если таких ордеров нет.
        """
        date_treshold = str(datetime.datetime.now().date())
        sql = f"""
        SELECT
            order_id
            ,buy_dt
            ,lots_requested
        FROM transaction.transactions 
        WHERE execution_report_status IN ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status IN  ('take_profit_posted', 'STOP_ORDER_STATUS_ACTIVE')
            AND estimated_sell_date <= '{date_treshold}'
            AND account_id = '{account_id}'
        """
        result = self.common_sql_req(sql)
        if result is not None and len(result) > 0:
            return result.values
        else:
            return []

    def get_all_candles_loaded(self) -> list[str]:
        """
        Получить уникальные имена тикеров из таблиц схем 'candles' и 'analytics'.

        Выполняет SQL-запрос для получения всех базовых таблиц из указанных схем,
        затем удаляет префиксы и суффиксы в именах таблиц, чтобы получить имена тикеров.

        Returns:
            list[str]: Список уникальных имен тикеров доступных в схемах 'candles' и 'analytics'.
        """
        sql = """
        SELECT DISTINCT
            replace(replace(replace(table_name,'candles_',''),'_hour',''),'_15min','') AS name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
            AND table_schema IN ('candles','analytics')
        """
        result = self.common_sql_req(sql)
        if result is None or result.empty:
            return []
        return result["name"].values.tolist()

    def get_orders_to_sell_by_exp_time(
        self,
        days_treshold: int,
        account_id,
    ) -> np.ndarray:
        """
        Получить ордера на продажу с датой покупки, не позднее заданного порога.

        Выполняет SQL-запрос для выборки ордеров с заполненным или частично заполненным статусом,
        направлением "покупка", статусом "take_profit_posted" и датой покупки не позже порога.

        Args:
            days_treshold (int): Число дней для порога даты покупки.

        Returns:
            np.ndarray: Массив значений с колонками order_id и estimated_price_change.
        """
        treshold_date = str(
            (
                datetime.datetime.now()
                - datetime.timedelta(days=days_treshold)
            ).date()
        )
        sql = f"""
        SELECT
            order_id
            ,estimated_price_change
        FROM transaction.transactions 
        WHERE execution_report_status in ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'take_profit_posted'
            AND buy_dt <= '{treshold_date}'
            AND account_id = '{account_id}'
        """
        result = self.common_sql_req(sql)
        if result is None:
            # Здесь можно вернуть пустой массив или другое значение по умолчанию
            return np.array([])
        return result.values

    @with_connection
    def load_all_candles(
        self,
        mode: str = "add",
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Загружает данные всех свечных таблиц для доступных тикеров по заданному режиму.

        В режиме "add" загружает только тикеры, которых еще нет в базе.
        В режиме "reload" загружает данные для всех тикеров заново.
        Запускает загрузку через метод training_datasets_download с фиксированным временным интервалом и разрешением "hour".

        Args:
            mode (str, optional): Режим загрузки данных: "add" для дозагрузки новых тикеров,
                                "reload" для полной перезагрузки всех тикеров. По умолчанию "add".

        Returns:
            None: Метод не возвращает значения.
        """
        candles_loaded = self.get_all_candles_loaded()
        instruments_tickers = (
            self.trading_manager.get_all_instruments_tickers()
        )

        if mode == "add":
            tickers_to_load = [
                t
                for t in instruments_tickers
                if t.lower() not in candles_loaded
            ]
        elif mode == "reload":
            tickers_to_load = instruments_tickers

        self.training_datasets_download(
            tickers=tickers_to_load,
            time_step_resolutions=["hour"],
            start_date=datetime.datetime(
                2019, 1, 1, tzinfo=datetime.timezone.utc
            ),
            finish_date=datetime.datetime(
                2024, 12, 31, tzinfo=datetime.timezone.utc
            ),
            save_mode="replace",
        )

    @with_connection
    def update_training_data(
        self,
        tickers: list[str],
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Обновляет тренировочные данные.

        Определяет максимальную дату имеющихся данных свечей по случайно выбранному тикеру из списка,
        вычисляет лаг в днях с текущей датой,
        фильтрует тикеры по наличию в загруженных данных. Если лаг превышает 2 дня,
        запускает дозагрузку обновлений свечей и пересчет индикаторов.

        Args:
            tickers (list[str]): Список доступных тикеров для обновления.

        Returns:
            None: Метод не возвращает значения.
        """
        print("update_training_data", len(tickers), "tickers total")
        ticker = random.choice(tickers)
        result = self.common_sql_req(
            f"SELECT max(date) FROM candles.{ticker}_hour"
        )
        if result is None:
            raise ValueError(f"No data returned for ticker {ticker}")
        max_available_date = result.values[0][0]
        if max_available_date is None:
            raise ValueError(f"No max date found for ticker {ticker}")
        max_available_date = max_available_date[:10]
        print("max_available_date", max_available_date)
        today = datetime.datetime.now()
        max_available_date = datetime.datetime.strptime(
            max_available_date, "%Y-%m-%d"
        )
        lag = (today - max_available_date).days
        all_tickers_from_candles = self.get_all_candles_loaded()
        print(
            "all_tickers_from_candles",
            len(all_tickers_from_candles),
            "tickers total",
        )
        tickers = [t for t in tickers if t in all_tickers_from_candles]
        if lag > 1:
            print("downloading updates")
            self.download_candles_delta(tickers)
        print("candles data up to date", "\n")

    @with_connection
    def check_if_table_exists(
        self,
        table: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> bool:
        """
        Проверяет существование базовой таблицы с указанным именем в базе данных.

        Выполняет SQL-запрос к information_schema, который проверяет наличие таблицы с заданным именем
        и типом 'BASE TABLE'. Возвращает True, если таблица существует, иначе False.

        Args:
            table (str): Имя таблицы для проверки.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            bool: True, если таблица существует, иначе False.
        """
        sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table}'
            """
        return pd.read_sql(sql, con)["is_exists"].values[0] > 0

    @with_connection
    def get_table_names_by_key(
        self,
        table: str,
        schema: str = "",
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> list[str]:
        """
        Получить список имен таблиц, содержащих заданный ключ в имени, из базы данных.

        Выполняет SQL-запрос по information_schema для поиска таблиц с именем типа 'BASE TABLE',
        включающих указанный подстрочный ключ в имени таблицы.

        Args:
            table (str): Подстрока для поиска в именах таблиц.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            list[str]: Список найденных имен таблиц, содержащих ключ.
        """
        schema_sql = f"AND table_schema = '{schema}'" if len(schema)>0 else ""
        sql = f"""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name LIKE '%{table}%'
                    {schema_sql}
            """
        if con is None:
            raise ValueError(
                "Database connection 'con' is required and cannot be None"
            )
        result = con.execute(text(sql))
        return [row[0] for row in result] if result is not None else []

    @with_connection
    def get_table_names_by_schema(
        self,
        schema: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> list[str]:
        """
        Получить список имен таблиц из базы данных, принадлежащих указанной схеме.

        Выполняет SQL-запрос к information_schema, который фильтрует таблицы типа 'BASE TABLE'
        по указанному имени схемы с использованием LIKE для частичного совпадения.

        Args:
            schema (str): Имя или часть имени схемы для поиска таблиц.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            list[str]: Список имен таблиц из указанной схемы.
        """
        sql = f"""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_schema LIKE '%{schema}%'
            """
        if con is None:
            raise ValueError(
                "Database connection 'con' is required and cannot be None"
            )
        result = con.execute(text(sql))
        return [row[0] for row in result]

    @with_connection
    def drop_table_by_name(
        self,
        table: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Удаляет таблицу с указанным именем из схемы 'candles', если такая таблица существует.

        Проверяет существование таблицы с помощью метода get_table_names_by_key,
        и если таблица найдена, выполняет команду DROP TABLE.

        Args:
            table (str): Имя таблицы для удаления.

        Returns:
            None: Метод не возвращает значения.
        """
        print(f"DROPPING {table}")
        if self.get_table_names_by_key(table):
            sql = f"""DROP TABLE IF EXISTS candles.{table}"""
            print(sql)
            self.common_sql_comand(sql)

    @with_connection
    def drop_tables_by_ticker(
        self,
        ticker: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Удаляет все таблицы из базы данных, содержащие имя указанного тикера.

        Использует метод get_table_names_by_key для получения списка таблиц с тикером в имени,
        затем последовательно удаляет каждую таблицу через drop_table_by_name.

        Args:
            ticker (str): Имя тикера для поиска таблиц.

        Returns:
            None: Метод не возвращает значения.
        """
        tables = self.get_table_names_by_key(ticker.lower())
        for table in tables:
            self.drop_table_by_name(table)

    def get_tickers_from_scope(self) -> list[str]:
        """
        Получить список тикеров из таблицы 'scope' схемы 'tickers'.

        Проверяет существование таблицы 'scope', если таблица отсутствует,
        создает её через метод create_tickers_scope, затем повторяет запрос.
        Возвращает список тикеров из таблицы.

        Returns:
            list[str]: Список тикеров из таблицы 'tickers.scope'.
        """
        if self.check_if_table_exists("scope"):
            sql = """
            SELECT
                ticker
            FROM tickers.scope 
            """
            result = self.common_sql_req(sql)
            if result is None:
                # Обработка отсутствия результата, например, вернуть пустой список
                return []
            return list(result["ticker"].values)
        else:
            self.create_tickers_scope()
            return self.get_tickers_from_scope()

    @with_connection
    def del_ticker_from_scope(
        self,
        ticker: str,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> None:
        """
        Удаляет заданный тикер из таблицы 'scope' в схеме 'tickers', если он там присутствует.

        Загружает текущий список тикеров, удаляет указанный тикер, если он найден,
        затем перезаписывает таблицу 'scope' обновленным списком.

        Args:
            ticker (str): Название тикера для удаления.
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            None: Метод не возвращает значения.
        """
        tickers = self.get_tickers_from_scope()
        if ticker in tickers:
            tickers.remove(ticker)
            tickers_df = pd.DataFrame(tickers, columns=["ticker"])
            tickers_df.to_sql(
                con=con,
                schema="tickers",
                name="scope",
                if_exists="replace",
                index=False,
            )
            print(f"{ticker} deleted from scope")

    @with_connection
    def create_tickers_scope(
        self,
        con: Optional["sqlalchemy.engine.Connection"] = None,
    ) -> pd.DataFrame:
        """
        Создает таблицу 'scope' в схеме 'tickers', содержащую уникальный список тикеров.

        Получает список всех тикеров по апи, потом достает только российские, с расписанием и доступные для торговли по апи
        Сохраняет результат в базу с перезаписью и возвращает датафреймом

        Args:
            con (sqlalchemy.engine.Connection | None, optional): Активное подключение к базе данных.

        Returns:
            pd.DataFrame: DataFrame со списком уникальных тикеров.
        """
        tickers = [
            i for i in self.trading_manager.get_all_stocks() 
            if i.get("schedule") != "unknown" 
            and i["type"] == "stock" 
            and i["currency"] == "rub" 
            and i["api_trade_available_flag"] 
            and i["is_available_for_trading"]
            and not i["for_qual_investor_flag"]
        ]
        tickers_df = pd.DataFrame(tickers, columns=["ticker"])
        tickers_df.to_sql(
            con=con,
            schema="tickers",
            name="scope",
            if_exists="replace",
            index=False,
        )
        print("tickers scope created")
        return tickers_df

    def get_all_features_names_with_sets(self) -> list[str]:
        """
        Получить список всех названий признаков из таблицы индикаторов, исключая групповые фичи.

        Выполняет запрос к таблице с валидационными индикаторами для тикера "SBER" с часовым разрешением,
        извлекает названия колонок, кроме первой (предположительно даты), и возвращает их списком.
        Предполагается, что групповые фичи (например, календарные) особо обрабатываются в генераторе признаков.

        Returns:
            list[str]: Список названий признаков для использования в генерации наборов признаков.
        """
        # возвращает названия всех фичей кроме тех, что идут пачками
        # на сегодня это фичи календаря
        # так сделано чтобы в генератор набора фичей некоторые фичи попадали только целиком
        sql = """
            SELECT
                *
            FROM datasets_sber_hour_validation_indicators
            LIMIT 1
            """
        data = self.common_sql_req(sql)
        if data is None:
            return []
        features_names = list(data.columns)[1:]
        return features_names

    def get_all_features_names(self) -> list[str]:
        """
        Получить список названий всех признаков, исключая признаки, имена которых начинаются с 'set_'.

        Использует метод get_all_features_names_with_sets для получения полного списка признаков,
        затем фильтрует и возвращает только те, которые не начинаются с 'set_'.

        Returns:
            list[str]: Список названий признаков без тех, что начинаются с 'set_'.
        """
        features_names = self.get_all_features_names_with_sets()
        features_names = [
            i for i in features_names if i[: len("set_")] != "set_"
        ]
        return features_names
    
    def get_last_update_date(
        self, ticker: str
        ) -> Optional[datetime.datetime]:
        table_name = f"candles.{ticker.lower()}_hour"
        sql = f"SELECT max(date) FROM {table_name}"
        result = self.common_sql_req(sql).values[0][0]
        if result is None or result[0] is None:
            return None
        if isinstance(result[0], datetime.datetime):
            return result[0]
        if isinstance(result[0], str):
            # Исправлено: берем первые 10 символов (YYYY-MM-DD) и парсим
            date_str = result[:10]
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")
        # Для других типов (date) преобразуем в datetime
        return datetime.datetime.combine(result[0], datetime.time.min)
    
    def get_last_update_dt(
        self, ticker: str
        ) -> Optional[datetime.datetime]:
        table_name = f"candles.{ticker.lower()}_hour"
        sql = f"SELECT max(date) FROM {table_name}"
        result = self.common_sql_req(sql).values[0][0]
        if result is None or result[0] is None:
            return None
        if isinstance(result[0], datetime.datetime):
            return result[0]
        if isinstance(result[0], str):
            # Исправлено: берем первые 1 символов (YYYY-MM-DD) и парсим
            date_str = result[:19]
            return datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        # Для других типов (date) преобразуем в datetime
        return datetime.datetime.combine(result[0], datetime.time.min)
