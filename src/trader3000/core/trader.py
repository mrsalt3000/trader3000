import datetime
import os
import pickle
import time
from decimal import Decimal
from typing import Any, Sequence, Union
from zoneinfo import ZoneInfo
#from IPython.display import clear_output

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tinkoff import invest
from tinkoff.invest import Client

from .data_management import DataManager
from .tinkoff_trading import TinkoffTrading
from .model_manager import ModelManager

load_dotenv()

class Trader:
    def __init__(
        self,
        timestep_resolution = "hour",
        delay_seconds=120, #меньше 2 минут - иногда свечи от апи не успевают обновиться
        buy_increment=1.0005,
        is_first_cycle=True,
    ):
        quant_cost=int(os.getenv("QUANT_COST")),
        TOKEN = os.getenv("TINVEST_TOKEN")
        self.quant_cost = quant_cost
        self.trading_manager = TinkoffTrading(
            TOKEN, Client, {"QUANT_COST": quant_cost}
        )
        self.data_manager = DataManager(self.trading_manager)
        self.timestep_resolution = timestep_resolution

        self.last_timestep = "1970-01-01"
        self.delay_seconds = delay_seconds

        self.buy_increment = buy_increment
        self.date_updated_schedules = datetime.datetime.now().date()
        self.date_agents_loaded = datetime.datetime.now().date()
        self.TZ = os.getenv("TZ")
        self.model_manager = ModelManager()
        self.stop_loss_rate = float(os.getenv("DEFAULT_STOP_LOSS_RATE")) if os.getenv("DEFAULT_STOP_LOSS_RATE") else .95
        self.is_main_agent = os.getenv("IS_MAIN_AGENT") == "TRUE"

    def unpickle(self, path: str) -> Any:
        """Загружает и возвращает объект из файла с помощью pickle.

        Открывает файл с бинарным содержимым по указанному пути и десериализует
        объект, сохранённый в этом файле с помощью pickle.

        Args:
            path (str): Путь к файлу, содержащему сериализованный объект.

        Returns:
            Любой десериализованный объект, сохранённый в файле.

        Raises:
            FileNotFoundError: Если файл по указанному пути не существует.
            pickle.UnpicklingError: Если файл повреждён или содержит некорректные данные.
            EOFError: При неожиданном окончании файла во время загрузки.
            Exception: Иные ошибки, возникающие при десериализации.
        """
        with open(f"{path}", "rb") as ff:
            data = pickle.load(ff)
        return data

    def load_model(self, path: str) -> tuple[Any, dict]:
        """Загружает модель и её параметры из сериализованного файла.

        Извлекает модель и конфигурационные параметры из pickle-файла,
        включая scaler, параметры сдвига, набор признаков и метрику ошибки.

        Args:
            path (str): Путь к pickle-файлу с сохранённой моделью.

        Returns:
            Кортеж из модели и словаря параметров:
                - model (Any): Загруженная модель из data["raw"]["model"].
                - params (dict): Словарь с scaler, shift_steps, feature_set и mean_error_test.

        Raises:
            KeyError: Если в файле отсутствуют ожидаемые ключи структуры данных.
            FileNotFoundError: Если pickle-файл не найден.
            pickle.UnpicklingError: При повреждении pickle-файла.
        """
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

    def get_tickers_models(self) -> dict[str, list[Any]]:
        """Возвращает словарь моделей, сгруппированных по их тикерам.

        Для каждого пути в `self.models_roots` извлекает тикер, идентификатор модели,
        загружает модель с параметрами и формирует словарь, где ключ —
        тикер, а значение — список из модели, её ID и параметров.

        Returns:
            dict[str, list[Any, str, dict]]: Словарь, где ключ — тикер (str),
                а значение — список из модели (Any), идентификатора модели (str)
                и словаря параметров (dict).
        """
        tickers_models = {}
        for model_root in self.models_roots:
            ticker = self.get_ticker_from_path(model_root)
            model_id = self.get_model_id(model_root)
            model, params = self.load_model(model_root)
            tickers_models[ticker] = [model, model_id, params]
        return tickers_models

    def get_ticker_from_path(self, path: str) -> str:
        """Извлекает тикер из пути к директории модели.

        Парсит путь, берёт последнее имя директории и извлекает первую часть
        до символа подчёркивания (предполагается формат 'TICKER_...').

        Args:
            path (str): Полный путь к директории модели.

        Returns:
            str: Тикер в верхнем регистре из имени директории.

        Raises:
            IndexError: Если путь не содержит ожидаемых разделителей или пуст.
        """
        ticker = path.split("\\")[-1].split("_")[0]
        return ticker

    def get_model_id(self, path: str) -> str:
        """Извлекает идентификатор модели из пути к директории.

        Возвращает имя последней директории в пути, которое используется
        как уникальный идентификатор модели.

        Args:
            path (str): Полный путь к директории модели.

        Returns:
            str: Имя последней директории в пути (идентификатор модели).

        Raises:
            IndexError: Если путь пустой или не содержит директорий.
        """
        return path.split("\\")[-1]

    def get_work_day_bound_for_schedule(
        self, schedule: str, mode: str, bound: str
    ) -> datetime.datetime:
        """Возвращает границу рабочего дня для торгового расписания.

        Находит начало/конец торгового дня в БД trading_schedules по расписанию,
        дате (сегодня/завтра) и типу границы (open_time/close_time).

        Args:
            schedule (str): Идентификатор торгового расписания (например, 'FX').
            mode (str): Режим выбора даты: 'today' или 'next_day'.
            bound (str): Тип границы: 'open_time' или 'close_time'.

        Returns:
            datetime.datetime: Граница рабочего дня в UTC.

        Raises:
            KeyError: Если в ответе БД отсутствует указанный столбец bound.
            ValueError: При некорректном формате даты из БД.
            IndexError: Если SQL-запрос не вернул ожидаемые данные.
        """
        if mode == "today":
            date_obj = datetime.datetime.now()
        elif mode == "next_day":
            date_obj = datetime.datetime.now() + datetime.timedelta(days=1)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        date_str = date_obj.strftime(
            "%Y-%m-%d"
        )  # str(date_obj)[:10] → более читаемо

        sql = f"""
        SELECT * 
        FROM trading_schedules 
        WHERE schedule = '{schedule}' 
            AND date LIKE '{date_str}%'
        """
        resp = self.data_manager.common_sql_req(sql)[bound]
        if len(resp) > 0:
            date_str = resp.values[0]
        else:
            date_str = "1970-01-01 00:00:00"
        return datetime.datetime.strptime(
            date_str[:19], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)

    def sleep_till_next_iter(self, schedule: str = "MOEX_MRNG_EVNG_E_WKND_DLR") -> None:
        """Приостанавливает выполнение до следующей итерации по расписанию.

        Вычисляет время следующей итерации на основе текущего времени, разрешения
        timestep и задержки, затем спит указанное количество секунд.

        Args:
            schedule (str): Идентификатор торгового расписания.

        Raises:
            ValueError: При некорректных данных расписания или разрешениях.
        """
        try:
            next_iter_start_by_resolution = self.data_manager.get_target_sell_dt(
                schedule_name=schedule,
                buy_dt=self.get_now("dt"),
                resolution=self.timestep_resolution,
                steps=1,
            )
    
            next_iter_start_by_resolution += datetime.timedelta(
                seconds=self.delay_seconds
            )
    
            seconds_to_sleep = (
                next_iter_start_by_resolution - self.get_now("dt")
            ).seconds
            print(
                "Sleeping till",
                str(next_iter_start_by_resolution + datetime.timedelta(hours=3))[
                    :16
                ],
            )
        except:
            print("Failed to get schedule info. Sleeping 1 hour")
            seconds_to_sleep = 60*57
        # print(seconds_to_sleep)
        time.sleep(seconds_to_sleep)

    def get_now(self, mode: str = "str") -> Union[str, datetime.datetime]:
        """Возвращает текущее время в указанном формате с коррекцией на 3 часа.

        Корректирует системное время на -3 часа (вероятно, для UTC+3) и возвращает
        либо строку ISO-формата, либо объект datetime.

        Args:
            mode (str, optional): Формат возвращаемого значения.
                'str' - строка YYYY-MM-DD HH:MM:SS (по умолчанию).
                'dt' - объект datetime.datetime.

        Returns:
            Union[str, datetime.datetime]: Текущее время в формате str или datetime.

        Raises:
            ValueError: Если mode не 'str' или 'dt'.
        """
        now = datetime.datetime.now(ZoneInfo(self.TZ))
        if mode == "str":
            return str(now)[:19]
        elif mode == "dt":
            return now
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'str' or 'dt'")

    def merge_dicts(self, a: dict, b: dict) -> dict:
        """Объединяет два словаря в новый словарь.

        Создаёт новый словарь, содержащий все пары ключ-значение из обоих словарей.
        При дублирующихся ключах значения из второго словаря перезаписывают первые.

        Args:
            a (dict): Первый словарь.
            b (dict): Второй словарь (его значения имеют приоритет).

        Returns:
            dict: Новый объединённый словарь.
        """
        return {k: v for k, v in list(a.items()) + list(b.items())}

    def get_new_orders(self) -> np.ndarray:
        """Возвращает массив ID новых ордеров из базы данных.

        Выполняет SQL-запрос для получения order_id транзакций со статусом
        'EXECUTION_REPORT_STATUS_NEW' и возвращает их в виде numpy массива.

        Returns:
            np.ndarray: Массив идентификаторов новых ордеров.
        """
        sql = f"""
        SELECT 
            order_id 
        FROM transaction.transactions 
        WHERE execution_report_status = 'EXECUTION_REPORT_STATUS_NEW'
            AND account_id = '{self.trading_manager.account_id}'
        """
        return self.data_manager.common_sql_req(sql)["order_id"].values

    def update_order_statuses(self) -> None:
        """Обновляет статусы новых ордеров в базе данных.

        Получает список новых ордеров, запрашивает их актуальные статусы
        у trading_manager и обновляет записи в таблице transactions.

        Returns:
            None
        """
        print("update_order_statuses")
        new_orders = self.get_new_orders()
        for order_id in new_orders:
            print("order_id:",order_id)
            order_status = self.trading_manager.get_order_status(order_id)
            sql = f"UPDATE transaction.transactions SET execution_report_status = '{order_status}' WHERE order_id = '{order_id}'"
            self.data_manager.common_sql_comand(sql)

    def update_sell_orders(self) -> None:
        """Обновляет статусы и цены выполненных ордеров на продажу.

        Проверяет новые ордера на продажу, запрашивает их состояние через API,
        обновляет цену продажи и статус 'sold' для полностью исполненных ордеров.

        Returns:
            None
        """
        print("update_sell_orders")
        new_sell_orders = self.data_manager.get_orders_sell_order_posted(self.trading_manager.account_id)
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
                    UPDATE transaction.transactions
                    SET sell_cost = {self.trading_manager.quotation_to_float(sell_order_info.average_position_price)}
                    WHERE sell_order_id = '{sell_order_info.order_id}'
                    """
                self.data_manager.common_sql_comand(sql)
                sql = f"""
                    UPDATE transaction.transactions
                    SET status = 'sold'
                    WHERE sell_order_id = '{sell_order_info.order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

    def post_sell_orders(self) -> None:
        """Размещает ордера на продажу для купленных позиций.

        Для каждой купленной позиции получает текущее состояние ордера покупки,
        рассчитывает цену продажи с учётом предполагаемого изменения цены и
        отправляет лимитный ордер на продажу. Обновляет статусы и sell_order_id
        в базе данных.

        Returns:
            None
        """
        orders_bought = self.data_manager.get_oreders_bought(self.trading_manager.account_id)
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
                    UPDATE transaction.transactions
                    SET status = 'sell_order_posted'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transaction.transactions
                    SET sell_order_id = '{sell_order_info.order_id}'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

    def post_take_profits(self) -> None:
        """Размещает стоп-ордера Take Profit для купленных позиций.

        Для каждой купленной позиции рассчитывает цену Take Profit с округлением
        до минимального шага цены инструмента и отправляет стоп-ордер через API.
        Обновляет статус 'take_profit_posted' и sell_order_id в базе данных.

        Returns:
            None
        """
        orders_bought = self.data_manager.get_orders_bought(self.trading_manager.account_id)
        for buy_order_id, estimated_price_change in orders_bought:
            target_price_change = Decimal(estimated_price_change)+1
            with self.trading_manager.client(
                self.trading_manager.token
            ) as client:
                buy_order_info = client.orders.get_order_state(
                    account_id=self.trading_manager.account_id,
                    order_id=buy_order_id,
                )

                sell_price_tp = invest.utils.quotation_to_decimal(
                    buy_order_info.average_position_price
                ) * target_price_change
                min_price_increment = client.instruments.get_instrument_by(
                    id_type=invest.InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                    id=buy_order_info.figi,
                ).instrument.min_price_increment

                min_price_increment = invest.utils.quotation_to_decimal(
                    min_price_increment
                )
                sell_price_tp_decimal = (
                    round(sell_price_tp / min_price_increment)
                    * min_price_increment
                )
                sell_price_tp_quotation = invest.utils.decimal_to_quotation(
                    Decimal(sell_price_tp_decimal)
                )
                print(
                    "="*10,"post_take_profits",'\n',
                    f"price= {sell_price_tp_quotation}",'\n',
                    f"stop_price= {sell_price_tp_quotation}",'\n',
                    f"buy_order_info.average_position_price",buy_order_info.average_position_price,'\n',
                    f"estimated_price_change",estimated_price_change,'\n',
                )
                
                #TAKE PROFIT
                sell_order_info = client.stop_orders.post_stop_order(
                    figi=buy_order_info.figi,
                    quantity=buy_order_info.lots_executed,
                    direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                    account_id=self.trading_manager.account_id,
                    stop_order_type=invest.schemas.StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                    price=sell_price_tp_quotation,
                    stop_price=sell_price_tp_quotation,
                    expire_date=None,
                    expiration_type=invest.schemas.StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                )

                sql = f"""
                    UPDATE transaction.transactions
                    SET status = 'take_profit_posted'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)

                sql = f"""
                    UPDATE transaction.transactions
                    SET sell_order_id = '{sell_order_info.stop_order_id}'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)


                #STOP LOSS
                print("posting stop loss")
                sell_price_sl = invest.utils.quotation_to_decimal(
                    buy_order_info.average_position_price
                ) * Decimal(self.stop_loss_rate)

                sell_price_sl_decimal = (
                    round(sell_price_sl / min_price_increment)
                    * min_price_increment
                )

                sell_price_sl_quotation = invest.utils.decimal_to_quotation(
                    Decimal(sell_price_sl_decimal)
                )

                sell_order_info = client.stop_orders.post_stop_order(
                    figi=buy_order_info.figi,
                    quantity=buy_order_info.lots_executed,
                    direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                    account_id=self.trading_manager.account_id,
                    stop_order_type=invest.schemas.StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                    price=sell_price_sl_quotation,
                    stop_price=sell_price_sl_quotation,
                    expire_date=None,
                    expiration_type=invest.schemas.StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                )

                sql = f"""
                    UPDATE transaction.transactions
                    SET stop_loss_order_id = '{sell_order_info.stop_order_id}'
                    WHERE order_id = '{buy_order_id}'
                    """
                self.data_manager.common_sql_comand(sql)


    def post_sell_rancid_orders(self) -> None:
        """Размещает рыночные ордера на продажу устаревших позиций.

        Определяет "залежавшиеся" ордера по дате покупки и времени удержания,
        размещает рыночные ордера на их продажу через API, обновляет статус,
        цену, стоимость и фактическую дату продажи в базе данных.

        Args:
            aquisition_hold_days (int): Максимальное количество дней удержания позиции.

        Returns:
            None
        """
        print("\npost_sell_rancid_orders")
        orders_rancid = self.data_manager.get_orders_rancid(self.trading_manager.account_id)
        for order_id, buy_dt, lots_requested in orders_rancid:
            print("order_id:",order_id)
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
                print("sell_order_info.total_order_amount:",sell_order_info.total_order_amount)
                total_order_amount = self.trading_manager.quotation_to_float(
                    sell_order_info.total_order_amount
                )
                print("got total_order_amount successfully")

                try:
                    sql = f"""
                        UPDATE transaction.transactions
                        SET status = 'rancid_sold_market'
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
                    print("transaction.transactions status set successfully")
                except Exception as e:
                    print("Error setting transaction.transactions status\n",e)

                try:
                    sql = f"""
                        UPDATE transaction.transactions
                        SET sell_order_id = '{sell_order_info.order_id}'
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
                    print("transaction.transactions sell_order_id set successfully")
                except Exception as e:
                    print("Error setting transaction.transactions sell_order_id\n",e)

                try:
                    sql = f"""
                        UPDATE transaction.transactions
                        SET executed_order_price = '{total_order_amount}'
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
                    print("transaction.executed_order_price executed_order_price set successfully")
                except Exception as e:
                    print("Error setting transaction.transactions executed_order_price\n",e)
                    
                try:
                    sql = f"""
                        UPDATE transaction.transactions
                        SET sell_cost = {round(total_order_amount/buy_order_info.lots_executed,1)}
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
                    print("transaction.transactions status set successfully")
                except Exception as e:
                    print("Error setting transaction.transactions sell_cost\n",e)

                try:
                    sql = f"""
                        UPDATE transaction.transactions
                        SET sell_dt_fact = '{now}'
                        WHERE order_id = '{order_id}'
                        """
                    self.data_manager.common_sql_comand(sql)
                    print("transaction.transactions sell_dt_fact set successfully")
                except Exception as e:
                    print("Error setting transaction.transactions sell_dt_fact\n",e)

    def trader_routine_preparation(self):
        print("trader_routine_preparation")
        #Если сервис главный - он удалит расписание от СЕГОДНЯ и скачает заново
        # Если не главный - подождет, пока расписание появится
        # Если не ждать - может быть коллизия, неглавный будет смотреть в расписание когда главный его удалил, а новое еще не скачал
        if self.is_main_agent:
            self.data_manager.update_schedules()
        else:
            self.await_schedule()
        
        paths = self.model_manager.get_classifiers_paths()
        tickers_by_models = [i.split("_")[2] for i in paths]
        print(len(tickers_by_models),"models available")
        
        tickers_for_action = []
        for ticker_by_models in tickers_by_models:
            schedule_name = self.trading_manager.get_ticker_info(ticker_by_models)["schedule"].upper()
            is_trading_day = self.data_manager.is_trading_day(schedule_name)
            if is_trading_day:
                tickers_for_action.append(ticker_by_models)
        
        print(len(tickers_for_action),"tickers in market today")
        if len(tickers_for_action)>0:
            self.data_manager.download_candles_delta([ticker.upper() for ticker in tickers_for_action])
            self.data_manager.recalculate_indicators(tickers = tickers_for_action)
            
        recomendations = self.model_manager.get_trading_recomendations()
        
        print(len(recomendations),"tickers available to buy")
        return recomendations

    def trader_routine(self, num_quants = 1):
        print("Trader routine started:",self.get_now())
        schedule_name = "MOEX_MRNG_EVNG_E_WKND_DLR" #расписание по умолчанию. Если торговли не будет, используется чтобы  найти ближайший рабочтий час биржи   
        while True:
            #   try:
            print("="*20,"Iteration",self.get_now(),"="*20)
            print("self.is_main_agent:",self.is_main_agent)

            #if self.is_main_agent:
                # Обработка прошлых действий
            self.update_order_statuses()
            self.update_sell_orders()
            self.update_tp_orders()
            self.post_sell_rancid_orders()
            self.post_take_profits()
            
            #Поиск, что купить

            recomendations = self.trader_routine_preparation()
            for recomendation in recomendations:
                
                ticker = recomendation["ticker"]
                print("\n","="*10,ticker,"="*10)
                estimated_price_change = recomendation["estimated_price_change"]
                hold_tics = recomendation["hold_tics"]
                model_id = recomendation["model_id"]
                
                last_candle_value = self.data_manager.get_last_price(
                ticker=ticker, resolution='hour'
                )
                transaction_from_trading_manager = (
                    self.trading_manager.quantum_buy(
                        ticker,
                        price=last_candle_value * self.buy_increment,
                        n_quants=num_quants,
                    )
                )
            
                schedule_name = self.trading_manager.get_ticker_info(ticker)["schedule"].upper()
                fact_dt = self.data_manager.get_candles_last_dt(
                                            ticker=ticker,
                                            time_step_resolution="hour",
                                        )
                estimated_sell_dt = self.data_manager.get_target_sell_dt(
                                                schedule_name=schedule_name,
                                                buy_dt=fact_dt.replace(tzinfo=None),
                                                resolution="hour",
                                                steps=hold_tics,
                                            )
                estimated_sell_date = (estimated_sell_dt+datetime.timedelta(days=1)).date()
            
                print(f"Bought {ticker}: {num_quants} quants")
                print(f"estimated_sell_date: {estimated_sell_date}",f"estimated_price_change: {estimated_price_change}")
                transaction_meta = {
                    "buy_dt": self.get_now(),
                    "num_quants": num_quants,
                    "sell_dt_fact": datetime.date(2100, 1, 1),
                    "status": "bought",
                    "quant_cost": self.quant_cost,
                    "sell_cost": None,
                    "model_id": model_id,
                    "estimated_price_change": estimated_price_change,
                    "estimated_sell_date": estimated_sell_date,
                }
            
                transaction = transaction_meta | transaction_from_trading_manager
                self.data_manager.save_transaction(transaction)
                time.sleep(1)
            self.sleep_till_next_iter(schedule_name) #schedule_name
            # except Exception as e:
            #     print("ERROR:\n", e)
            #     self.sleep_till_next_iter()
        

    def update_tp_orders(self):
        print("update_tp_orders")
        sql = """
        SELECT
            sell_order_id
        FROM transaction.transactions
        WHERE status IN ('take_profit_posted', 'STOP_ORDER_STATUS_ACTIVE')
        """
        sell_order_ids = self.data_manager.common_sql_req(sql)["sell_order_id"].values
        print(len(sell_order_ids),"stop orders to_update")
        
        with self.trading_manager.client(
            self.trading_manager.token
        ) as client:
            stop_orders = client.stop_orders.get_stop_orders(
                account_id = self.trading_manager.account_id, 
                from_ = datetime.datetime(2026,1,1),
                to = datetime.datetime.now(),
                status = invest.schemas.StopOrderStatusOption.STOP_ORDER_STATUS_ALL
            )
        
        stor_order_statuses = [{"id":i.stop_order_id, "status":i.status.name} for i in stop_orders.stop_orders if i.stop_order_id in sell_order_ids]
        
        for status in stor_order_statuses:
            sql = f"""
                UPDATE transaction.transactions
                SET status = '{status["status"]}'
                WHERE sell_order_id = '{status["id"]}'
            """
            self.data_manager.common_sql_comand(sql)
        print("success")

    def await_schedule(self):
        """
        Расписание обновляет только главный контейнер
        Может произойти коллизия когда неглавный контейнес смотрит, есть ли расписание, а оно не успело скачаться
        Чтобы этого не было, в случае отсутствия расписания делаем задержку
        """
        today = str(datetime.datetime.now())[:10]
        wait_sec = 5
        max_attempts = 5
        for i in range(max_attempts):
            schedule_is_available = self.data_manager.common_sql_req(
                        f"SELECT count(*) FROM trading.schedules  WHERE date = '{today}'"
                    ).values[0][0] > 0
            schedule_is_available = bool(schedule_is_available)
            if schedule_is_available:
                break
            else:
                print("schedule is unavailable, attempt:" ,i+1, "of", max_attempts)
                time.sleep(wait_sec)
        return schedule_is_available