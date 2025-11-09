import os

from dotenv import load_dotenv

load_dotenv()  # Загружаем переменные окружения из .env
import datetime
import os
import random
import shutil
import time
from itertools import chain

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text

from .indicators_calculator import IndicatorsCalculator


class DataManager:
    def __init__(
        self,
        trading_manager=None,
    ):
        self.CANDLES_SART_DATE = datetime.datetime(2019, 1, 1)  # 2019
        self.root = os.getcwd()
        self.trading_manager = trading_manager
        self.psql_main_creds = os.getenv("DATABASE_URL")
        if trading_manager:
            from tinkoff.invest import Client

            from .tinkoff_trading import TinkoffTrading

    def with_connection(func):
        """Декоратор для управления подключением к БД."""

        def wrapper(self, *args, **kwargs):
            engine = sqlalchemy.create_engine(self.psql_main_creds)
            con = engine.connect()
            try:
                # Передаём подключение в метод как именованный аргумент

                result = func(self, *args, con=con, **kwargs)
                con.commit()  # Фиксируем изменения, если не было ошибок
                return result
            except Exception as e:
                con.rollback()  # Откатываем в случае ошибки
                raise e
            finally:
                con.close()  # Всегда закрываем подключение

        return wrapper

    def get_sql_table_info(self, table_name):
        return f"""
        SELECT
            count(*) AS rows_count
            ,min(date) AS min_date
            ,max(date) AS max_date
        FROM candles.{table_name}
        """

    @with_connection
    def get_tables_info(self, con):
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

    @with_connection
    def training_datasets_download(
        self,
        tickers,
        time_step_resolutions,
        start_date=datetime.datetime(2019, 1, 1),
        finish_date=None,
        save_mode="replace",
        con=None,
    ):
        for ticker in tickers:
            print(ticker)
            if "@" not in ticker and "-" not in ticker:
                for time_step_resolution in time_step_resolutions:
                    data = self.trading_manager.get_market_trading_history(
                        ticker=ticker,
                        start_date=start_date,
                        finish_date=finish_date,
                        time_step_resolution=time_step_resolution,
                    )
                    if type(data) != type(None):
                        if len(data) > 0:
                            print(len(data), "rows downloaded")
                            table_name = (
                                f"{ticker.lower()}_{time_step_resolution}"
                            )
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
                        self.del_ticker_from_scope(ticker.upper())
                        self.drop_tables_by_ticker(ticker.lower())
        print("data loaded")

    @with_connection
    def common_sql_req(self, sql, con=None, n_attempts=4):
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

    @with_connection
    def common_sql_comand(self, sql, con=None):
        counter = 1
        try:
            con.execute(text(sql))
        except Exception as error:
            print(error)
            counter += 1
            print("attempt №", counter)
            time.sleep(10)

    @with_connection
    def get_all_candles_table_names(self, con):
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
        self, con, mode="total"
    ):  # mode in ["update","total"]
        candles_tables = self.get_all_candles_table_names()
        if mode == "total":
            TRESHOLDS = {
                "train": [
                    datetime.datetime(2019, 1, 1, 0),
                    datetime.datetime(2021, 6, 1, 0),
                ],
                "test": [
                    datetime.datetime(2021, 6, 1, 0),
                    datetime.datetime(2024, 1, 1, 0),
                ],
                "validation": [
                    datetime.datetime(2024, 1, 1, 0),
                    datetime.datetime(2100, 1, 1, 0),
                ],
                "total": [
                    datetime.datetime(2000, 1, 1, 0),
                    datetime.datetime(2100, 1, 1, 0),
                ],
            }
            for candles_table in candles_tables:
                candles = self.common_sql_req(
                    sql=f'SELECT DISTINCT * FROM candles."{candles_table}" ORDER BY date DESC'
                )
                candles.drop_duplicates(inplace=True)
                indicators_calculator = IndicatorsCalculator(candles)
                calculated_data = indicators_calculator.calculate_indicators()
                calculated_data_columns = calculated_data.columns
                calculated_data.index = [
                    datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
                    for i in calculated_data.index
                ]

                for dataset_type, [start, finish] in TRESHOLDS.items():
                    table_name = f"{candles_table.replace('candles_','datasets_')}_{dataset_type}_cost_history"
                    slice = calculated_data.loc[start:finish][
                        calculated_data_columns[:5]
                    ]
                    slice.index = (str(i) for i in slice.index)
                    slice.index.name = "date"
                    slice.to_sql(
                        con=con,
                        name=table_name,
                        if_exists="replace",
                        index=True,
                    )

                    table_name = f"{candles_table.replace('candles_','datasets_')}_{dataset_type}_indicators"
                    slice = calculated_data[start:finish][
                        calculated_data_columns[5:]
                    ]
                    slice.index = (str(i) for i in slice.index)
                    slice.index.name = "date"
                    slice.to_sql(
                        con=con,
                        name=table_name,
                        if_exists="replace",
                        index=True,
                    )
        elif mode == "update":
            for candles_table in candles_tables:
                candles = self.common_sql_req(
                    sql=f'SELECT DISTINCT * FROM candles."{candles_table}" ORDER BY date DESC LIMIT 1000'
                )
                candles.drop_duplicates(inplace=True)
                candles.sort_values(by="date", ascending=False, inplace=True)
                indicators_calculator = IndicatorsCalculator(candles)
                calculated_data = indicators_calculator.calculate_indicators()
                calculated_data_columns = calculated_data.columns

                for prefix in ["validation", "total"]:
                    table_name = f"{candles_table.replace('candles_','datasets_')}_{prefix}_cost_history"
                    slice_ = calculated_data[calculated_data_columns[:5]]
                    dates_from_slice = (
                        "(" + ",".join([f"'{i}'" for i in slice_.index]) + ")"
                    )
                    old_cost_history = self.common_sql_req(
                        sql=f'SELECT DISTINCT * FROM candles."{table_name}" WHERE date not in {dates_from_slice} ORDER BY date DESC'
                    ).set_index("date")
                    data = pd.concat([slice_, old_cost_history]).sort_values(
                        by="date", ascending=False
                    )
                    data.to_sql(
                        con=con,
                        name=table_name,
                        if_exists="replace",
                        index=True,
                    )

                for prefix in ["validation", "total"]:
                    table_name = f"{candles_table.replace('candles_','datasets_')}_{prefix}_indicators"
                    slice_ = calculated_data[calculated_data_columns[5:]]
                    dates_from_slice = (
                        "(" + ",".join([f"'{i}'" for i in slice_.index]) + ")"
                    )
                    old_indicators = self.common_sql_req(
                        sql=f'SELECT DISTINCT * FROM candles."{table_name}" WHERE date not in {dates_from_slice} ORDER BY date DESC'
                    ).set_index("date")
                    data = pd.concat([slice_, old_indicators]).sort_values(
                        by="date", ascending=False
                    )
                    data.to_sql(
                        con=con,
                        name=table_name,
                        if_exists="replace",
                        index=True,
                    )

    @with_connection
    def get_trading_log(self, con):
        sql = """
                SELECT
                    *
                FROM trading_log
    
            """
        data = pd.read_sql(sql, con)
        return data

    def get_sql_select_candles(
        self, ticker, resolution, dataset_type, part, normalized=False
    ):
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
        self, con, ticker, resolution, mode="test", normalized=False
    ):
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
                        datasets[f"{type}_cost_history"],
                        pd.read_sql(sql, con).set_index("date"),
                    )
                    sql = self.get_sql_select_candles(
                        ticker_, resolution, type, "indicators", normalized
                    )
                    datasets[f"{type}_indicators"] = pd.concat(
                        datasets[f"{type}_indicators"],
                        pd.read_sql(sql, con).set_index("date"),
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

    def get_candles_last_dt(self, ticker="SBER", time_step_resolution="hour"):
        sql = f"""
        SELECT
            max(date) AS ans
        FROM candles.{ticker.lower()}_{time_step_resolution}
        """
        last_dt = self.common_sql_req(sql)["ans"].values[0]
        last_dt = datetime.datetime.strptime(
            last_dt, "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        return last_dt

    def download_candles_delta(
        self, tickers=["SBER"], time_step_resolutions=["hour"]
    ):
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
                self.training_datasets_download(
                    tickers=[ticker],
                    time_step_resolutions=[time_step_resolution],
                    start_date=start_dt,
                    finish_date=datetime.datetime.now().replace(
                        tzinfo=datetime.timezone.utc
                    ),
                    save_mode="append",
                )

    @with_connection
    def get_schedules(
        self,
        from_=datetime.datetime.now().replace(tzinfo=datetime.timezone.utc),
        to=datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
        + datetime.timedelta(days=1),
        con=None,
    ):
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

    def update_schedules(self):
        max_date_saved = self.common_sql_req(
            "SELECT max(date) AS ans FROM trading.schedules"
        )["ans"].values[0]
        max_date_saved = datetime.datetime.strptime(
            max_date_saved[:19] + " 00:00:00", "%Y-%m-%d %H:%M:%S"
        )
        print("max_date_saved", max_date_saved)
        today = datetime.datetime.now()
        start_date = max_date_saved if max_date_saved > today else today
        for i in range(13):
            try:
                start_date_ = start_date + datetime.timedelta(days=i + 1)
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
        self, ticker, time_step_resolution, window_size
    ):
        sql = f"""
            SELECT
                *
            FROM datasets_{ticker.lower()}_{time_step_resolution}_validation_indicators
            ORDER BY date DESC
            LIMIT {window_size+1}
            """
        data = self.common_sql_req(sql).values
        last_timestep = data[0][0]
        data = [d[1:] for d in data]
        observation = list(chain(*data))

        return observation, last_timestep

    @with_connection
    def save_transaction(self, con, transaction):
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

    def get_last_price(self, ticker, resolution):
        sql = f"""
            SELECT
                close
            FROM candles_{ticker}_{resolution}
            ORDER BY date DESC 
            LIMIT 1
            """
        return self.common_sql_req(sql)["close"].values[0]

    def get_transactions(self):
        sql = f"""
            SELECT
                *
            FROM transactions
            ORDER BY buy_dt DESC 
            """
        return self.common_sql_req(sql)

    def seconds_between_dt(self, start_dt, finish_dt):
        date_diff_raw = finish_dt - start_dt
        date_diff_seconds = (
            date_diff_raw.days * 24 * 3600 + date_diff_raw.seconds
        )
        return date_diff_seconds

    def get_timeline(self, schedule, timestep_seconds):
        timeline = []
        for day in schedule.to_dict("records"):
            timesteps_within = day["work_seconds"] // timestep_seconds
            start_time = datetime.datetime.strptime(
                day["start_time"][:19], "%Y-%m-%d %H:%M:%S"
            )
            for t in range(timesteps_within + 1):
                timeline.append(
                    start_time
                    + datetime.timedelta(seconds=timestep_seconds * t)
                )
        return timeline

    def get_target_sell_dt(self, schedule_name, buy_dt, resolution, steps):
        if resolution == "hour":
            timestep_seconds = 3600
        elif resolution == "15min":
            timestep_seconds = 15 * 60
        hold_seconds = timestep_seconds * steps
        buy_date = buy_dt.date()

        sql = f"""
        SELECT DISTINCT
            *
        FROM trading_schedules
        WHERE schedule = '{schedule_name}'
            AND date >= '{str(buy_date)}'
        ORDER BY date
        """
        schedule = self.common_sql_req(sql)
        timeline = self.get_timeline(schedule, timestep_seconds)
        timeline = list(filter(lambda x: x >= buy_dt, timeline))
        if len(timeline) > steps:
            sell_date = timeline[steps]
        else:
            sell_date = datetime.datetime(2100, 1, 1, 1)
        return sell_date

    def is_trading_day(self, schedule):
        schedule = schedule.upper()
        today = str(datetime.datetime.now().date())
        sql = f"""
        SELECT
            *
        FROM trading_schedules
        WHERE schedule = '{schedule}'
        AND date = '{today}'
        """
        resp = self.common_sql_req(sql)
        return len(resp) > 0

    def get_indicators_tables(self):
        sql = """
                SELECT
                    *
                FROM information_schema.tables
                WHERE table_schema = 'indicators'
            """
        tables = self.common_sql_req(sql)["name"].values
        return tables

    @with_connection
    def update_normalized_datasets(self, con):
        tables = self.get_indicators_tables()
        for table_name in tables:
            sql = f"""
                    SELECT
                        *
                    FROM {table_name}
                """
            data = self.common_sql_req(sql)

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

    def get_tickers_from_datasets(self, resolution="hour"):
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

    def get_predictor_datasets(self, ticker, resolution, mode, normalized):
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

    def add_window(self, x_in, y_in=None, n_steps=1):
        y_out = y_in[:-n_steps] if y_in else None
        x_out = []
        for p in range(n_steps, len(x_in), 1):
            x_out.append(list(chain.from_iterable(x_in[p - n_steps : p])))
        return np.array(x_out), np.array(y_out)

    def get_oreders_bought(self):
        sql = """
        SELECT
            order_id
            ,estimated_price_change
        FROM transactions 
        WHERE execution_report_status in ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'bought'
        """
        return self.common_sql_req(sql).values

    def get_orders_sell_order_posted(self):
        sql = """
        SELECT
            sell_order_id
        FROM transactions 
        WHERE status = 'sell_order_posted'
            AND direction = 'ORDER_DIRECTION_BUY'
        """
        return self.common_sql_req(sql).values

    def get_oreders_rancid(self, days_treshold=0):
        date_treshold = str(
            datetime.datetime.now().date()
            - datetime.timedelta(days=days_treshold)
        )
        sql = f"""
        SELECT
            order_id
            ,buy_dt
            ,lots_requested
        FROM transactions 
        WHERE execution_report_status in ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'take_profit_posted'
            AND estimated_sell_date <= '{date_treshold}'
        """
        result = self.common_sql_req(sql)
        if len(result):
            return result.values
        else:
            return []

    def get_all_candles_loaded(self):
        sql = """
        SELECT DISTINCT
            replace(replace(replace(table_name,'candles_',''),'_hour',''),'_15min','') AS name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
            AND table_schema IN ('candles','analytics')
        """
        return self.common_sql_req(sql)["name"].values

    def get_orders_to_sell_by_exp_time(self, days_treshold):
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
        FROM transactions 
        WHERE execution_report_status in ('EXECUTION_REPORT_STATUS_FILL','EXECUTION_REPORT_STATUS_PARTIALLYFILL')
            AND direction = 'ORDER_DIRECTION_BUY'
            AND status = 'take_profit_posted'
            AND buy_dt <= '{treshold_date}'
        """
        return self.common_sql_req(sql).values

    def load_all_candles(self, mode="add"):
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

    def update_training_data(self, tickers):
        ticker = random.choice(tickers)
        max_available_date = self.common_sql_req(
            f"SELECT max(date) FROM candles.{ticker}_hour"
        ).values[0][0][:10]
        print("max_available_date", max_available_date)
        today = datetime.datetime.now()
        lag = (
            today - datetime.datetime.strptime(max_available_date, "%Y-%m-%d")
        ).days
        all_tickers_from_candles = self.get_all_candles_loaded()
        ticker = [t for t in ticker if t in all_tickers_from_candles]
        if lag > 2:
            print("downloading updates")
            self.download_candles_delta(tickers)
            self.recalculate_indicators()
        print("training data up to date", "\n")

    @with_connection
    def check_if_table_exists(self, table, con=None):
        sql = f"""
                SELECT
                    count(*)!=0 AS is_exists 
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name = '{table}'
            """
        return pd.read_sql(sql, con)["is_exists"].values[0] > 0

    @with_connection
    def get_table_names_by_key(self, table, con=None):
        sql = f"""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_name LIKE '%{table}%'
            """
        result = con.execute(text(sql))
        return [row[0] for row in result]

    @with_connection
    def get_table_names_by_schema(self, schema, con=None):
        sql = f"""
                SELECT
                    table_name AS table
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                    AND table_schema LIKE '%{schema}%'
            """
        result = con.execute(text(sql))
        return [row[0] for row in result]

    def drop_table_by_name(self, table):
        if self.get_table_names_by_key(table):
            sql = f"""DROP TABLE candles.{table}"""
            self.common_sql_comand(sql)

    def drop_tables_by_ticker(self, ticker):
        tables = self.get_table_names_by_key(ticker.lower())
        for table in tables:
            self.drop_table_by_name(table)

    def get_tickers_from_scope(self):
        if self.check_if_table_exists("scope"):
            sql = f"""
            SELECT
                ticker
            FROM tickers.scope 
            """
            return list(self.common_sql_req(sql)["ticker"].values)
        else:
            self.create_tickers_scope()
            return self.get_tickers_from_scope()

    @with_connection
    def del_ticker_from_scope(self, ticker, con=None):
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
    def create_tickers_scope(self, con=None):
        tickers = list(
            set(
                [
                    i.replace("candles_", "").replace("_hour", "")
                    for i in self.get_table_names_by_schema("candles")
                ]
            )
        )
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

    def get_all_features_names_with_sets(self):
        # возвращает названия всех фичей кроме тех, что идут пачками
        # на сегодня это фичи календаря
        # так сделано чтобы в генератор набора фичей некоторые фичи попадали только целиком
        sql = f"""
            SELECT
                *
            FROM datasets_sber_hour_validation_indicators
            LIMIT 1
            """
        data = self.common_sql_req(sql)
        features_names = list(data.columns)[1:]
        return features_names

    def get_all_features_names(self):
        features_names = self.get_all_features_names_with_sets()
        features_names = [
            i for i in features_names if i[: len("set_")] != "set_"
        ]
        return features_names

    def backup_db(self):
        today = str(datetime.datetime.now().date())
        shutil.copyfile("db/main.db", f"db/main_{today}.db")
