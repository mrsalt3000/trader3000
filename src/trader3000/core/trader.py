import datetime
import os
import pickle
import time
from decimal import Decimal

import pandas as pd
from dotenv import load_dotenv
from tinkoff import invest
from tinkoff.invest import Client

from .data_management import DataManager
from .tinkoff_trading import TinkoffTrading

load_dotenv()  # загружает .env файл


class Trader:
    def __init__(
        self,
        timestep_resolution,
        path_agents
        # ,models_roots
        ,
        transaction_treshold,
        mode="prod",
        quant_cost=300,
        delay_seconds=15,
        buy_increment=1.0005,
        is_first_cycle=True,
        aquisition_hold_days=14,
    ):
        assert mode in ["test", "prod"]
        TOKEN = os.getenv("TINVEST_TOKEN")
        self.quant_cost = quant_cost
        self.path_agents = path_agents
        self.trading_manager = TinkoffTrading(
            TOKEN, Client, {"QUANTUM_PRICE": quant_cost}
        )
        self.data_manager = DataManager(self.trading_manager)
        self.timestep_resolution = timestep_resolution
        self.models_roots = self.get_models_roots()
        self.tickers_models = self.get_tickers_models()

        self.last_timestep = "1970-01-01"
        self.delay_seconds = delay_seconds

        self.mode = mode
        self.transaction_treshold = transaction_treshold
        self.buy_increment = buy_increment
        self.daily_teshold_increment_constant = 1.5
        self.date_updated_schedules = datetime.datetime.now().date()
        self.aquisition_hold_days = aquisition_hold_days
        self.date_agents_loaded = datetime.datetime.now().date()

    def get_models_roots(self):
        models_roots = [
            self.path_agents + "\\" + i for i in os.listdir(self.path_agents)
        ]
        return models_roots

    def unpickle(self, path):
        with open(f"{path}", "rb") as ff:
            data = pickle.load(ff)
        return data

    def load_model(self, path):
        data = self.unpickle(path)
        model = data["raw"]["model"]
        params = {
            "scaler": data["raw"]["scaler"],
            "shift_steps": data["input_args"]["shift_steps"],
            "feature_set": list(
                data["model_params"]["model_coefficients"].keys()
            ),
            "mean_error_test": data["metrics"]["test"]["mean_error"],
        }
        return model, params

    def get_tickers_models(self):
        tickers_models = {}
        for model_root in self.models_roots:
            ticker = self.get_ticker_from_path(model_root)
            model_id = self.get_model_id(model_root)
            model, params = self.load_model(model_root)
            tickers_models[ticker] = [model, model_id, params]
        return tickers_models

    def get_ticker_from_path(self, path: str):
        ticker = path.split("\\")[-1].split("_")[0]
        return ticker

    def get_model_id(self, path):
        return path.split("\\")[-1]

    def get_work_day_bound_for_schedule(self, schedule, mode, bound):
        if mode == "today":
            date = datetime.datetime.now()
        elif mode == "next_day":
            date = datetime.datetime.now() + datetime.timedelta(days=1)
        date = str(date)[:10]
        sql = f"""
        SELECT 
            * 
        FROM trading_schedules 
        WHERE schedule = '{schedule}' 
            AND date LIKE '{date}%'
            """
        resp = self.data_manager.common_sql_req(sql)[bound]
        if len(resp) > 0:
            date_str = resp.values[0]
        else:
            date_str = "1970-01-01 00:00:00"
        return datetime.datetime.strptime(
            date_str[:19], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)

    def sleep_till_next_iter(self, schedule):
        next_iter_start_by_resolution = self.data_manager.get_target_sell_dt(
            schedule_name=schedule,
            buy_dt=self.get_now("dt"),
            resolution=self.timestep_resolution,
            steps=1,
        ).replace(tzinfo=datetime.timezone.utc)
        next_iter_start_by_resolution += datetime.timedelta(
            seconds=self.delay_seconds
        )

        seconds_to_sleep = (
            next_iter_start_by_resolution
            - self.get_now("dt").replace(tzinfo=datetime.timezone.utc)
        ).seconds
        print(
            "Sleeping till",
            str(next_iter_start_by_resolution + datetime.timedelta(hours=3))[
                :16
            ],
        )
        # print(seconds_to_sleep)
        time.sleep(seconds_to_sleep)

    def get_now(self, mode="str"):
        now = datetime.datetime.now() - datetime.timedelta(hours=3)
        if mode == "str":
            return str(now)[:19]
        elif mode == "dt":
            return now

    def merge_dicts(self, a, b):
        return {k: v for k, v in list(a.items()) + list(b.items())}

    def watch_buyer(self):
        while True:
            if datetime.datetime.now().date() > self.date_updated_schedules:
                self.data_manager.update_schedules()
            self.data_manager.download_candles_delta(
                [ticker.upper() for ticker in self.tickers_models.keys()]
            )
            self.data_manager.recalculate_indicators()
            self.post_sell_rancid_orders(
                aquisition_hold_days=self.aquisition_hold_days
            )
            datetime.datetime.now().date()
            if (
                datetime.datetime.now().date() - self.date_agents_loaded
            ).days > 0:
                self.tickers_models = self.get_tickers_models()
                self.data_manager.backup_db()
                print("updating agents list")
            for ticker, [
                model,
                model_id,
                params,
            ] in self.tickers_models.items():
                print("\n", ticker)
                ticker_is_available_for_trading = (
                    self.trading_manager.ticker_is_available_for_trading(
                        ticker
                    )
                )
                schedule_name = self.trading_manager.get_ticker_info(ticker)[
                    "schedule"
                ].upper()
                if ticker_is_available_for_trading:
                    if self.data_manager.is_trading_day(schedule_name):
                        self.update_order_statuses()
                        self.update_sell_orders()
                        sql = f"SELECT * FROM datasets_{ticker}_hour_validation_indicators ORDER BY date DESC LIMIT 300"
                        indicators = self.data_manager.common_sql_req(sql)

                        prediction = self.get_predictions_from_model(
                            model, indicators, params, mode="last_point"
                        )
                        # return prediction
                        # process predictions
                        last_candle_value = self.data_manager.get_last_price(
                            ticker=ticker, resolution=self.timestep_resolution
                        )
                        prediction = [last_candle_value, prediction[0]]
                        # return
                        # prediction[0]=[last_candle_value]
                        # prediction = {k:prediction[k] for k in sorted(prediction.keys())}
                        prediction = {
                            0: prediction[0],
                            params["shift_steps"]: prediction[1],
                        }
                        fact_dt = self.data_manager.get_candles_last_dt(
                            ticker=ticker,
                            time_step_resolution=self.timestep_resolution,
                        )

                        prediction_processed = {
                            self.data_manager.get_target_sell_dt(
                                schedule_name=schedule_name,
                                buy_dt=fact_dt.replace(tzinfo=None),
                                resolution="hour",
                                steps=t,
                            ): round(prediction[t] / last_candle_value, 2)
                            for t in prediction.keys()
                        }

                        # get raw buy estimations
                        deals = []
                        for dt, value in prediction_processed.items():
                            days_to_deal = (
                                dt.date() - self.get_now("dt").date()
                            ).days
                            transaction_treshold_weighted = round(
                                self.transaction_treshold
                                + days_to_deal
                                * self.daily_teshold_increment_constant
                                / 100,
                                2,
                            )
                            if value >= transaction_treshold_weighted:
                                deals.append(
                                    {
                                        "estimated_sell_dt": dt,
                                        "estimated_sell_date": dt.date(),
                                        "estimated_price_change": value,
                                        "is_to_be_sold_today": dt.date()
                                        == self.get_now("dt").date(),
                                        "treshold": transaction_treshold_weighted,
                                        "ticker": ticker,
                                        "model_id": model_id,
                                    }
                                )
                        if len(deals) > 0:
                            raise "Переписать с SQLITE3 и PD на common_sql_command"
                            # with sqlite3.connect(
                            #     self.data_manager.CONN_STRING
                            # ) as con:
                            #     table_name = "transactions_estimations_raw"
                            #     pd.DataFrame(deals).to_sql(
                            #         con=con,
                            #         name=table_name,
                            #         if_exists="replace",
                            #         index=False,
                            #     )

                            # # get best estimations by day
                            # sql = """
                            # SELECT
                            #     *
                            # FROM (
                            #     SELECT
                            #         *
                            #         ,row_number() OVER(
                            #             PARTITION BY ticker,estimated_sell_date
                            #             ORDER BY estimated_price_change DESC, estimated_sell_dt ASC
                            #         ) = 1 AS is_best
                            #     FROM transactions_estimations_raw
                            #     )
                            # WHERE is_best
                            # """
                            # transactions_estimations_combed = (
                            #     self.data_manager.common_sql_req(sql)
                            # )
                            # with sqlite3.connect(
                            #     self.data_manager.CONN_STRING
                            # ) as con:
                            #     table_name = "transactions_estimations_combed"
                            #     transactions_estimations_combed.to_sql(
                            #         con=con,
                            #         name=table_name,
                            #         if_exists="append",
                            #         index=False,
                            #     )

                            # if num_quants := len(
                            #     transactions_estimations_combed
                            # ):
                            #     print("going to buy", num_quants, "quants")
                            #     estimated_sell_date = (
                            #         transactions_estimations_combed[
                            #             "estimated_sell_date"
                            #         ].max()
                            #     )
                            #     estimated_price_change = (
                            #         transactions_estimations_combed[
                            #             "estimated_price_change"
                            #         ].max()
                            #     )
                            #     transaction_from_trading_manager = (
                            #         self.trading_manager.quantum_buy(
                            #             ticker,
                            #             price=last_candle_value
                            #             * self.buy_increment,
                            #             n_quants=num_quants,
                            #         )
                            #     )
                            #     print(f"Bought {num_quants} quants")
                            #     transaction_meta = {
                            #         "buy_dt": self.get_now(),
                            #         "num_quants": num_quants,
                            #         "sell_dt_fact": datetime.date(2100, 1, 1),
                            #         "status": "bought",
                            #         "quant_cost": self.quant_cost,
                            #         "sell_cost": None,
                            #         "model_id": model_id,
                            #         "estimated_price_change": estimated_price_change,
                            #         "estimated_sell_date": estimated_sell_date,
                            #     }
                            #     transaction = self.merge_dicts(
                            #         transaction_meta,
                            #         transaction_from_trading_manager,
                            #     )
                            #     self.data_manager.save_transaction(transaction)
                        else:
                            print("no lucky moments")
                        self.post_take_profits()
                    else:
                        print("instrument is not available for trading")
            self.sleep_till_next_iter(schedule_name)
        self.is_first_cycle = False

    def watch_seller(self):
        while True:
            # Пробуем найти дату продажи для дальних транзакций (на которые не хватило расписания)
            self.data_manager.update_schedules()
            sql = """
            SELECT
                *
            FROM transactions
            WHERE sell_dt_plan = '2100-01-01'
            """
            transactions_to_process = self.data_manager.common_sql_req(
                sql
            ).to_dict("records")

            for transaction in transactions_to_process:
                ticker = transaction["ticker"]
                figi = transaction["figi"]
                buy_dt = datetime.datetime.strptime(
                    transaction["buy_dt"], "%Y-%m-%d %H:%M:%S"
                )
                lots = transaction["lots_executed"]
                steps_to_sell = transaction["steps_to_sell"]
                schedule_name = self.trading_manager.get_ticker_info(ticker)[
                    "schedule"
                ]
                order_id = transaction["order_id"]

                target_sell_dt = self.data_manager.get_target_sell_dt(
                    schedule_name=schedule_name,
                    buy_dt=buy_dt,
                    resolution="hour",
                    steps=steps_to_sell,
                )
                if str(target_sell_dt)[:10] != "2100-01-01":
                    sql = f"""
                        UPDATE transactions
                        SET sell_dt_plan = '{target_sell_dt}'
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
            # Ищем транзакции, которые пора продавать
            sql = f"""
                SELECT
                    *
                FROM transactions
                WHERE sell_dt_plan <= '{self.get_now()}'
                AND status = 'bought'
                """
            transactions_to_process = self.data_manager.common_sql_req(
                sql
            ).to_dict("records")

            for transaction in transactions_to_process:
                figi = transaction["figi"]
                lots = transaction["lots_executed"]
                order_id = transaction["order_id"]

                sell_cost = self.trading_manager.quantum_sell(figi, lots)
                sql = f"""
                    UPDATE transactions
                    SET sell_dt_fact = '{self.get_now()}'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
                sql = f"""
                    UPDATE transactions
                    SET status = 'sold'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
                sql = f"""
                    UPDATE transactions
                    SET sell_cost = {sell_cost}
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
            self.sleep_till_next_iter(schedule_name)

    # ЗАМЕНИТЬ НА ТАКУЮ ЖЕ ФУНКЦИЮ ИЗ training_manager
    def get_predictions_from_model(
        self, model, data_df, params, mode="common"
    ):
        assert isinstance(data_df, pd.DataFrame)
        params["shift_steps"]

        # Пока не включу использование фичей
        # data = data_df[params["feature_set"]].values
        # print(len(data_df.columns))
        # print(data_df.columns)
        data_df = data_df[[i for i in data_df.columns if i != "date"]]
        data = data_df.values
        data = params["scaler"].transform(data)
        if mode == "last_point":
            data = data[-10:]
        prediction = model.predict(data)
        prediction = prediction - params["mean_error_test"]
        if mode == "last_point":
            prediction = prediction[-1:]
        return prediction

    def median(self, inp):
        return sorted(inp)[len(inp) // 2]

    def get_new_orders(self):
        sql = "SELECT order_id FROM transactions WHERE execution_report_status = 'EXECUTION_REPORT_STATUS_NEW'"
        return self.data_manager.common_sql_req(sql)["order_id"].values

    def update_order_statuses(self):
        new_orders = self.get_new_orders()
        for order_id in new_orders:
            order_status = self.trading_manager.get_order_status(order_id)
            sql = f"UPDATE transactions SET execution_report_status = '{order_status}' WHERE order_id = '{order_id}'"
            self.data_manager.common_sql_comand(sql)

    def update_sell_orders(self):
        new_sell_orders = self.data_manager.get_orders_sell_order_posted()
        for [sell_order_id] in new_sell_orders:
            with self.trading_manager.client(
                self.trading_manager.token
            ) as client:
                sell_order_info = client.orders.get_order_state(
                    account_id=self.trading_manager.account_id,
                    order_id=sell_order_id,
                )
            if (
                sell_order_info.execution_report_status.name
                == "EXECUTION_REPORT_STATUS_FILL"
            ):
                sql = f"""
                    UPDATE transactions
                    SET sell_cost = {self.trading_manager.quotation_to_float(sell_order_info.average_position_price)}
                    WHERE sell_order_id = '{sell_order_info.order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
                sql = f"""
                    UPDATE transactions
                    SET status = 'sold'
                    WHERE sell_order_id = '{sell_order_info.order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

    def post_sell_orders(self):
        orders_bought = self.data_manager.get_oreders_bought()
        for buy_order_id, estimated_price_change in orders_bought:
            with self.trading_manager.client(
                self.trading_manager.token
            ) as client:

                buy_order_info = client.orders.get_order_state(
                    account_id=self.trading_manager.account_id,
                    order_id=buy_order_id,
                )

                sell_price = self.trading_manager.float_to_quotation(
                    self.trading_manager.quotation_to_float(
                        buy_order_info.average_position_price
                    )
                    * estimated_price_change
                )

                sell_order_info = client.orders.post_order(
                    figi=buy_order_info.figi,
                    quantity=buy_order_info.lots_executed,
                    direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                    account_id=self.trading_manager.account_id,
                    order_type=invest.schemas.OrderType.ORDER_TYPE_LIMIT,
                    price=sell_price,
                )

                sql = f"""
                    UPDATE transactions
                    SET status = 'sell_order_posted'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET sell_order_id = '{sell_order_info.order_id}'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

    def post_take_profits(self):
        orders_bought = self.data_manager.get_oreders_bought()
        for buy_order_id, estimated_price_change in orders_bought:
            with self.trading_manager.client(
                self.trading_manager.token
            ) as client:
                buy_order_info = client.orders.get_order_state(
                    account_id=self.trading_manager.account_id,
                    order_id=buy_order_id,
                )

                sell_price = invest.utils.quotation_to_decimal(
                    buy_order_info.average_position_price
                ) * Decimal(estimated_price_change)
                min_price_increment = client.instruments.get_instrument_by(
                    id_type=invest.InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                    id=buy_order_info.figi,
                ).instrument.min_price_increment
                # number_digits_after_point = (
                #     9 - len(str(min_price_increment.nano)) + 1
                # )
                min_price_increment = invest.utils.quotation_to_decimal(
                    min_price_increment
                )
                sell_price = (
                    round(sell_price / min_price_increment)
                    * min_price_increment
                )
                sell_price = invest.utils.decimal_to_quotation(
                    Decimal(sell_price)
                )

                sell_order_info = client.stop_orders.post_stop_order(
                    figi=buy_order_info.figi,
                    quantity=buy_order_info.lots_executed,
                    direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                    account_id=self.trading_manager.account_id,
                    stop_order_type=invest.schemas.StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                    price=sell_price,
                    stop_price=sell_price,
                    expire_date=None,
                    expiration_type=invest.schemas.StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                )

                sql = f"""
                    UPDATE transactions
                    SET status = 'take_profit_posted'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET sell_order_id = '{sell_order_info.stop_order_id}'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

    def post_sell_rancid_orders(self, aquisition_hold_days):
        print("post_sell_rancid_orders")
        orders_rancid = self.data_manager.get_oreders_rancid(
            aquisition_hold_days
        )
        for order_id, buy_dt, lots_requested in orders_rancid:
            with self.trading_manager.client(
                self.trading_manager.token
            ) as client:
                now = str(datetime.datetime.now())

                buy_order_info = client.orders.get_order_state(
                    account_id=self.trading_manager.account_id,
                    order_id=order_id,
                )

                sell_order_info = client.orders.post_order(
                    figi=buy_order_info.figi,
                    quantity=buy_order_info.lots_executed,
                    direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                    account_id=self.trading_manager.account_id,
                    order_type=invest.schemas.OrderType.ORDER_TYPE_BESTPRICE,
                )
                total_order_amount = self.trading_manager.quotation_to_float(
                    sell_order_info.total_order_amount
                )
                sql = f"""
                    UPDATE transactions
                    SET status = 'rancid_sold_market'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET sell_order_id = '{sell_order_info.order_id}'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET executed_order_price = '{total_order_amount}'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET sell_cost = {round(total_order_amount/buy_order_info.lots_executed,1)}
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transactions
                    SET sell_dt_fact = '{now}'
                    WHERE order_id = '{order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
