import datetime
import math
import uuid
from datetime import timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import pandas as pd
from tinkoff import invest
from tinkoff.invest import (
    CandleInterval,
    Client,
    OrderDirection,
    OrderType,
    PostOrderResponse,
)
from tinkoff.invest.schemas import Quotation


class Transaction(TypedDict):
    order_id: str
    order_request_id: str
    figi: str
    ticker: str
    execution_report_status: str
    lots_requested: int
    lots_executed: int
    initial_order_price: float
    executed_order_price: float
    total_order_amount: float
    initial_commission: float
    initial_security_price: float
    executed_commission: float
    direction: str  # можно конкретизировать Enum типа OrderDirection.name
    order_type: str  # можно конкретизировать Enum типа OrderType.name


class TinkoffTrading:
    def __init__(self, token, client=Client, params={}):
        self.token = token
        self.client = client
        self.tickers = self.get_all_stocks()
        self.DEFAULT_STOP_LOSS = -0.04
        self.DEFAULT_STOP_ORDER_EXPIRE_DURATION = timedelta(hours=1)
        self.account_id = self.get_account_name()[0]
        self.account_name = self.get_account_name()[1]
        self.BUY_INCREASE = 1.005
        self.QUANTUM_PRICE = (
            params.get("QUANTUM_PRICE")
            if params.get("QUANTUM_PRICE")
            else 3000
        )

        print("account_name:", self.account_name)
        # self.EXPIRATION_TYPE = StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_DATE

    def get_account_name(self) -> Tuple[str, str]:
        with self.client(self.token) as client:
            responce = client.users.get_accounts()
        return responce.accounts[0].id, responce.accounts[0].name

    def get_last_price(self, ticker: str) -> float:
        figi = self.get_figi_by_ticker(ticker)
        with self.client(self.token) as client:
            resp = (
                client.market_data.get_last_prices(figi=[figi])
                .last_prices[0]
                .price
            )
        return self.quotation_to_float(resp)

    def get_ticker_by_figi(self, figi: str) -> str:
        return list(filter(lambda x: x["figi"] == figi, self.tickers))[0][
            "ticker"
        ]

    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        data = list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )
        if data:
            return data[0]["figi"]
        else:
            return None

    def get_all_stocks(self) -> List:
        with self.client(self.token) as client:
            pass

            def get_instruments(raw, type):
                instruments = [
                    {
                        "name": getattr(instrument, "name", "-"),
                        "figi": getattr(instrument, "figi", "-"),
                        "ticker": getattr(instrument, "ticker", "-"),
                        "schedule": getattr(instrument, "exchange", "-"),
                        "type": type,
                        "class_code": getattr(instrument, "class_code", "-"),
                        "lot": getattr(instrument, "lot", "-"),
                        "currency": getattr(instrument, "currency", "-"),
                        "sector": getattr(instrument, "sector", "-"),
                        "api_trade_available_flag": getattr(
                            instrument, "api_trade_available_flag", "-"
                        ),
                        "for_qual_investor_flag": getattr(
                            instrument, "for_qual_investor_flag", "-"
                        ),
                        "weekend_flag": getattr(
                            instrument, "weekend_flag", "-"
                        ),
                        "first_1min_candle_date": getattr(
                            instrument, "first_1min_candle_date", "-"
                        ),
                        "first_1day_candle_date": getattr(
                            instrument, "first_1day_candle_date", "-"
                        ),
                        "is_available_for_trading": (
                            not getattr(
                                instrument, "for_qual_investor_flag", False
                            )
                            and getattr(
                                instrument, "api_trade_available_flag", True
                            )
                            and getattr(instrument, "currency", "-") == "rub"
                        ),
                        "min_price_increment": getattr(
                            instrument, "min_price_increment", 0.1
                        ),
                    }
                    for instrument in raw.instruments
                ]
                return instruments

            data = get_instruments(
                client.instruments.currencies(), "currency"
            ) + get_instruments(client.instruments.shares(), "stock")
            data.append(
                {
                    "name": "Российский рубль",
                    "figi": "RUB000UTSTOM",
                    "ticker": "RUB",
                    "type": "currency",
                    "class_code": "-",
                    "lot": 0,
                    "currency": "rub",
                    "sector": "-",
                    "api_trade_available_flag": True,
                    "for_qual_investor_flag": False,
                    "weekend_flag": False,
                    "first_1min_candle_date": datetime.datetime(
                        2222, 1, 1, 1, 1, tzinfo=datetime.timezone.utc
                    ),
                    "first_1day_candle_date": datetime.datetime(
                        2222, 1, 1, 1, 1, tzinfo=datetime.timezone.utc
                    ),
                    "is_available_for_trading": True,
                }
            )
        return data

    def get_market_trading_history(
        self,
        ticker: str,
        start_date: datetime.datetime,
        finish_date: Optional[datetime.datetime] = None,
        time_step_resolution: str = "day",
    ) -> pd.DataFrame:

        start_date = start_date.replace(tzinfo=datetime.timezone.utc)
        if time_step_resolution == "hour":
            interval = CandleInterval.CANDLE_INTERVAL_HOUR
            # timedelta = datetime.timedelta(hours = days*24)
        elif time_step_resolution == "day":
            interval = CandleInterval.CANDLE_INTERVAL_DAY
            # timedelta = datetime.timedelta(days = days)
        elif time_step_resolution == "10min":
            interval = CandleInterval.CANDLE_INTERVAL_10_MIN
        elif time_step_resolution == "15min":
            interval = CandleInterval.CANDLE_INTERVAL_15_MIN
        elif time_step_resolution == "30min":
            interval = CandleInterval.CANDLE_INTERVAL_30_MIN

        figi = self.get_figi_by_ticker(ticker)
        if figi:
            with self.client(self.token) as client:
                history = []
                for candle in client.get_all_candles(
                    figi=figi,
                    from_=start_date,
                    to=finish_date,
                    interval=interval,
                ):
                    candle_dict = {
                        "date": candle.time.replace(tzinfo=None),
                        "open": self.quotation_to_float(candle.open),
                        "high": self.quotation_to_float(candle.high),
                        "low": self.quotation_to_float(candle.low),
                        "close": self.quotation_to_float(candle.close),
                        "volume": candle.volume,
                    }
                    history.append(candle_dict)

            return pd.DataFrame.from_dict(history)

    def quotation_to_float(self, inp: Quotation) -> float:
        return inp.units + (inp.nano / 10 ** len(str(inp.nano)))

    def get_portfolio(self) -> Dict[str, Any]:
        with self.client(self.token) as client:
            portfolio_raw = client.operations.get_portfolio(
                account_id=self.account_id
            )
        positions = [
            {
                "ticker": self.get_ticker_by_figi(position.figi),
                "figi": position.figi,
                "currency": position.average_position_price.currency,
                "quantity": self.quotation_to_float(position.quantity),
                "average_position_price": self.quotation_to_float(
                    position.average_position_price
                ),
                "current_price": self.quotation_to_float(
                    position.current_price
                ),
            }
            for position in portfolio_raw.positions
        ]
        portfolio = {
            "total_amount_rubles": self.quotation_to_float(
                portfolio_raw.total_amount_shares
            )
            + self.quotation_to_float(portfolio_raw.total_amount_currencies),
            "total_amount_shares": self.quotation_to_float(
                portfolio_raw.total_amount_shares
            ),
            "positions": positions,
        }
        return portfolio

    def get_lot_quantity(self, ticker: str) -> int:
        return list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )[0]["lot"]

    def quantum_buy(
        self, ticker: str, price: float, n_quants: int = 1
    ) -> Transaction:

        last_price = self.get_last_price(ticker)
        lot_quantity = self.get_lot_quantity(ticker)
        buy_price = last_price * self.BUY_INCREASE
        min_buy_amount = buy_price * lot_quantity

        lots_to_buy = int(self.QUANTUM_PRICE // min_buy_amount)
        figi = self.get_figi_by_ticker(ticker)

        price = self.round_by_min_price_increment(price, ticker)
        price_quotation = self.float_to_quotation(price)
        # price_quotation.nano = int(str(price_quotation.nano)[:1])#+"0"*1
        print(ticker, price)
        rubles_available = self.get_rubles_available()
        order_total_amount = min_buy_amount * lots_to_buy * n_quants
        is_enough_money = order_total_amount < rubles_available
        print(
            "quantum_buy\n",
            "ticker",
            ticker,
            "price",
            price,
            "n_quants",
            n_quants,
            "last_price",
            last_price,
            "lot_quantity",
            lot_quantity,
            "buy_price",
            buy_price,
            "min_buy_amount",
            min_buy_amount,
            "rubles_available",
            rubles_available,
            # "\n"
        )
        print()

        transaction: Transaction

        if figi is None:
            raise RuntimeError("figi не должен быть None")

        if lots_to_buy > 0 and is_enough_money:
            with self.client(self.token) as client:
                post_order_response: PostOrderResponse = (
                    client.orders.post_order(
                        quantity=lots_to_buy * n_quants,
                        direction=OrderDirection.ORDER_DIRECTION_BUY,
                        account_id=self.account_id,
                        order_type=OrderType.ORDER_TYPE_LIMIT,
                        order_id=uuid.uuid4().hex,
                        instrument_id=figi,
                        price=price_quotation,
                    )
                )

            post_order_response.execution_report_status
            transaction = {
                "order_id": post_order_response.order_id,
                "order_request_id": post_order_response.order_request_id,
                "figi": post_order_response.figi,
                "ticker": self.get_ticker_by_figi(post_order_response.figi),
                "execution_report_status": post_order_response.execution_report_status.name,
                "lots_requested": post_order_response.lots_requested,
                "lots_executed": post_order_response.lots_executed,
                "initial_order_price": self.quotation_to_float(
                    post_order_response.initial_order_price
                ),
                "executed_order_price": self.quotation_to_float(
                    post_order_response.executed_order_price
                ),
                "total_order_amount": self.quotation_to_float(
                    post_order_response.total_order_amount
                ),
                "initial_commission": self.quotation_to_float(
                    post_order_response.initial_commission
                ),
                "initial_security_price": self.quotation_to_float(
                    post_order_response.initial_security_price
                ),
                "executed_commission": self.quotation_to_float(
                    post_order_response.executed_commission
                ),
                "direction": post_order_response.direction.name,
                "order_type": post_order_response.order_type.name,
            }
        else:
            skip_reason = (
                "Not enough money, skipped"
                if is_enough_money
                else "Too small quant price, skipped"
            )
            transaction = {
                "order_id": "",
                "order_request_id": "",
                "figi": figi,
                "ticker": ticker,
                "execution_report_status": skip_reason,
                "lots_requested": lots_to_buy * n_quants,
                "lots_executed": 0,
                "initial_order_price": self.quotation_to_float(
                    price_quotation
                ),
                "executed_order_price": 0,
                "total_order_amount": 0,
                "initial_commission": 0,
                "initial_security_price": 0,
                "executed_commission": 0,
                "direction": OrderDirection.ORDER_DIRECTION_BUY,
                "order_type": OrderType.ORDER_TYPE_LIMIT,
            }
        return cast(Transaction, transaction)

    def get_trading_schedules(
        self, from_: datetime.datetime, to: datetime.datetime
    ) -> Any:
        with self.client(self.token) as client:
            schedules_raw = client.instruments.trading_schedules(
                from_=from_, to=to
            ).exchanges
        return schedules_raw

    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        data = list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )
        if data:
            return data[0]
        return None

    def quantum_sell(self, figi, quantity):
        with self.client(self.token) as client:
            resp = client.orders.post_order(
                figi=figi,
                quantity=quantity,
                direction=invest.schemas.OrderDirection.ORDER_DIRECTION_SELL,
                account_id=self.account_id,
                order_type=invest.schemas.OrderType.ORDER_TYPE_MARKET,
            )
        sell_price_with_commission = self.quotation_to_float(
            resp.total_order_amount
        ) + self.quotation_to_float(resp.initial_commission)
        return sell_price_with_commission

    def float_to_quotation(self, number: float) -> Quotation:
        return invest.utils.decimal_to_quotation(round(Decimal(number), 2))

    def get_order_status(self, order_id: str):
        with self.client(self.token) as client:
            responce = client.orders.get_order_state(
                account_id=self.account_id, order_id=order_id
            )
            return responce.execution_report_status.name

    def get_stop_order_state(self, order_id):
        with self.client(self.token) as client:
            resp = client.stop_orders.get_stop_orders(
                account_id=self.account_id
            )
            return list(
                filter(lambda x: x.stop_order_id == order_id, resp.stop_orders)
            )[0]

    def get_all_instruments_tickers(self):
        return [
            instrument["ticker"]
            for instrument in self.tickers
            if instrument["currency"] == "rub"
        ]

    def get_rubles_available(self):
        return list(
            filter(
                lambda x: x["ticker"] == "RUB",
                self.get_portfolio()["positions"],
            )
        )[0]["quantity"] - self.get_blocked_lots("RUB")

    def ticker_is_available_for_trading(self, ticker):
        not_available_statuses = [
            "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING"
        ]
        with self.client(self.token) as client:
            shares = client.instruments.shares()
        return (
            list(
                filter(
                    lambda x: x.ticker == ticker.upper(), shares.instruments
                )
            )[0].trading_status.name
            not in not_available_statuses
        )

    def round_by_min_price_increment(self, price, ticker):
        min_price_increment = list(
            filter(
                lambda x: x["ticker"].upper() == ticker.upper(), self.tickers
            )
        )[0]["min_price_increment"]
        min_price_increment = self.quotation_to_float(min_price_increment)
        return round(
            math.ceil(price / min_price_increment) * min_price_increment,
            len(str(min_price_increment)),
        )

    def get_blocked_lots(self, ticker):
        figi = self.get_figi_by_ticker(ticker)
        with self.client(self.token) as client:
            portfolio = client.operations.get_portfolio(
                account_id=self.account_id
            )
        ticker_data = list(
            filter(lambda x: x.figi == figi, portfolio.positions)
        )
        if ticker_data:
            ticker_data = ticker_data[0]
            blocked_lots = self.quotation_to_float(ticker_data.blocked_lots)
            return blocked_lots
        else:
            return 0
