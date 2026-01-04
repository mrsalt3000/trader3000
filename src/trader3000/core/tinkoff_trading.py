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
    MoneyValue,
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
        """
        Retrieves the ID and name of the first account from the user's accounts.

        Uses the client's token to fetch accounts and returns the first one's ID and name.

        Returns:
            Tuple[str, str]: A tuple containing (account_id, account_name) of the first account.

        Raises:
            IndexError: If no accounts are available in the response.
            Exception: If client request fails (network, auth, etc.).
        """
        with self.client(self.token) as client:
            responce = client.users.get_accounts()
        return responce.accounts[0].id, responce.accounts[0].name

    def get_last_price(self, ticker: str) -> float:
        """
        Retrieves the last (most recent) price for the given ticker symbol.

        Converts ticker to FIGI identifier, fetches last price via market data API,
        and converts the quotation to float value.

        Args:
            ticker (str): Stock or instrument ticker symbol (e.g., 'AAPL', 'SBER').

        Returns:
            float: Last price value as float.

        Raises:
            ValueError: If ticker is invalid or FIGI lookup fails.
            IndexError: If no price data returned for the FIGI.
            Exception: If client request fails (network, auth, etc.).
        """
        figi = self.get_figi_by_ticker(ticker)
        with self.client(self.token) as client:
            resp = (
                client.market_data.get_last_prices(figi=[figi])
                .last_prices[0]
                .price
            )
        return self.quotation_to_float(resp)

    def get_ticker_by_figi(self, figi: str) -> str:
        """
        Retrieves the ticker symbol for the given FIGI identifier from cached tickers.

        Performs a linear search through the self.tickers list to find matching FIGI
        and returns the associated ticker.

        Args:
            figi (str): Financial Instrument Global Identifier (FIGI) of the instrument.

        Returns:
            str: Ticker symbol corresponding to the FIGI.

        Raises:
            IndexError: If no ticker found with the given FIGI in self.tickers.
            KeyError: If ticker entry lacks "ticker" key or "figi" field malformed.
        """
        return list(filter(lambda x: x["figi"] == figi, self.tickers))[0][
            "ticker"
        ]

    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        """
        Retrieves the FIGI identifier for the given ticker symbol from cached tickers.

        Performs case-insensitive search (uppercases ticker) through self.tickers list
        and returns the first matching FIGI, or None if not found.

        Args:
            ticker (str): Stock or instrument ticker symbol (e.g., 'AAPL', 'SBER').

        Returns:
            Optional[str]: FIGI identifier if found, None otherwise.

        Raises:
            KeyError: If matching ticker entry lacks "figi" key or data malformed.
        """
        data = list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )
        if data:
            return data[0]["figi"]
        else:
            return None

    def get_all_stocks(self) -> List:
        """
        Fetches comprehensive list of all available trading instruments (stocks, currencies).

        Combines shares and currencies data from Tinkoff API, normalizes into unified dict format,
        manually adds RUB base currency record. Filters for RUB trading availability.

        Returns:
            List[Dict[str, Any]]: List of instrument dictionaries with fields:
                name, figi, ticker, schedule, type, class_code, lot, currency, sector,
                api_trade_available_flag, for_qual_investor_flag, weekend_flag,
                first_1min_candle_date, first_1day_candle_date, is_available_for_trading,
                min_price_increment.

        Raises:
            Exception: If client request fails (network, auth, API limits, etc.).
        """
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
        """
        Fetches historical market data (OHLCV candles) for specified ticker and timeframe.

        Converts ticker to FIGI, requests candles via Tinkoff API get_all_candles iterator,
        normalizes timezone and quotation units, returns pandas DataFrame.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP').
            start_date (datetime.datetime): Start date (UTC normalized).
            finish_date (Optional[datetime.datetime]): End date (default: now). UTC.
            time_step_resolution (str): Candle interval: 'day', 'hour', '10min', '15min', '30min'.

        Returns:
            pd.DataFrame: OHLCV data with columns: date, open, high, low, close, volume.

        Raises:
            ValueError: Invalid time_step_resolution or ticker/FIGI not found.
            Exception: API errors, network issues, empty history.
        """
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
        """
        Converts Tinkoff Quotation proto message to float price value.

        Quotation consists of units (integer part) and nano (9 decimal digits fractional part).
        Formula: units + nano / 10^len(str(nano)) reconstructs decimal value.

        Args:
            inp (Quotation): Tinkoff Quotation protobuf message with units and nano fields.

        Returns:
            float: Decimal price value (e.g., Quotation(units=123, nano=450000000) -> 123.45).

        Raises:
            ValueError: If nano conversion fails or Quotation malformed.
            AttributeError: If inp lacks units/nano attributes.
        """
        return inp.units + (inp.nano / 10 ** len(str(inp.nano)))

    def get_portfolio(self) -> Dict[str, Any]:
        """
        Fetches current portfolio positions and totals from Tinkoff API.

        Requests portfolio by account_id, converts all Quotation fields to float,
        maps FIGI to tickers, calculates total portfolio value in RUB.

        Returns:
            Dict[str, Any]: Portfolio structure with:
                total_amount_rubles (float): Total portfolio value in RUB.
                total_amount_shares (float): Total shares value.
                positions (List[Dict]): Position details with ticker, figi, currency,
                    quantity, average_position_price, current_price.

        Raises:
            ValueError: Invalid account_id or missing FIGI/ticker mapping.
            Exception: API errors, network issues, malformed positions data.
        """
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
        """
        Retrieves the lot size (minimum trading quantity) for given ticker from cache.

        Performs case-insensitive search through self.tickers, returns lot value
        as integer for the matching instrument.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP').

        Returns:
            int: Lot size (quantity per trading lot) for the instrument.

        Raises:
            IndexError: If ticker not found in self.tickers cache.
            KeyError: If matching entry lacks "lot" field or data malformed.
            ValueError: If lot value not convertible to int.
        """
        return list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )[0]["lot"]

    def money_value_to_quotation(
        self, money_value: Optional[MoneyValue]
    ) -> Optional[Quotation]:
        """
        Converts Tinkoff MoneyValue protobuf to Quotation protobuf.

        Copies units and nano fields from MoneyValue to Quotation, preserving
        exact monetary precision. Returns None for None input.

        Args:
            money_value (Optional[MoneyValue]): Tinkoff MoneyValue protobuf message.

        Returns:
            Optional[Quotation]: Equivalent Quotation with same units/nano values,
                None if input is None.

        Raises:
            AttributeError: If money_value lacks units/nano attributes.
            ValueError: If units/nano values invalid for Quotation construction.
        """
        if money_value is None:
            return None
        return Quotation(
            units=money_value.units,
            nano=money_value.nano,
            # currency=money_value.currency,
        )

    def money_value_to_float(self, money_value: Optional[MoneyValue]) -> float:
        """
        Converts Tinkoff MoneyValue protobuf to float monetary value.

        Handles None input (returns 0.0), converts MoneyValue to intermediate Quotation,
        then applies quotation_to_float for decimal precision.

        Args:
            money_value (Optional[MoneyValue]): Tinkoff MoneyValue protobuf message.

        Returns:
            float: Decimal monetary value. Returns 0.0 if money_value is None.

        Raises:
            AttributeError: If money_value lacks units/nano attributes.
            ValueError: If units/nano values invalid for Quotation conversion.
        """
        if money_value is None:
            return 0.0
        quotation = Quotation(units=money_value.units, nano=money_value.nano)
        return self.quotation_to_float(quotation)

    def quantum_buy(
        self, ticker: str, price: float, n_quants: int = 1
    ) -> Transaction:
        """
        Executes quantum-sized limit buy order for ticker based on available capital.

        Calculates lot size and max affordable lots using last price * BUY_INCREASE,
        rounds price by min increment, checks RUB balance, places order if affordable.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP').
            price (float): Limit order price per unit.
            n_quants (int): Number of quantum batches to buy (default: 1).

        Returns:
            Transaction: Order result dict with execution status, lots, prices, commissions.
                Returns skipped transaction if insufficient funds or invalid quantum.

        Raises:
            RuntimeError: If FIGI lookup fails for ticker.
            ValueError: Invalid price rounding or quotation conversion.
            Exception: Order API failures, network issues.
        """
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

            transaction = {
                "order_id": post_order_response.order_id,
                "order_request_id": post_order_response.order_request_id,
                "figi": post_order_response.figi,
                "ticker": self.get_ticker_by_figi(post_order_response.figi),
                "execution_report_status": post_order_response.execution_report_status.name,
                "lots_requested": post_order_response.lots_requested,
                "lots_executed": post_order_response.lots_executed,
                "initial_order_price": self.money_value_to_float(
                    post_order_response.initial_order_price
                )
                or 0.0,
                "executed_order_price": self.money_value_to_float(
                    post_order_response.executed_order_price
                )
                or 0.0,
                "total_order_amount": self.money_value_to_float(
                    post_order_response.total_order_amount
                )
                or 0.0,
                "initial_commission": self.money_value_to_float(
                    post_order_response.initial_commission
                )
                or 0.0,
                "initial_security_price": self.money_value_to_float(
                    post_order_response.initial_security_price
                )
                or 0.0,
                "executed_commission": self.money_value_to_float(
                    post_order_response.executed_commission
                )
                or 0.0,
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
                "direction": OrderDirection.ORDER_DIRECTION_BUY.name,
                "order_type": OrderType.ORDER_TYPE_LIMIT.name,
            }
        return cast(Transaction, transaction)

    def get_trading_schedules(
        self, from_: datetime.datetime, to: datetime.datetime
    ) -> Any:
        """
        Fetches trading schedules for all exchanges in specified date range.

        Requests Tinkoff API trading schedules between from_ and to dates,
        returns raw exchanges list with session details (open/close times, holidays).

        Args:
            from_ (datetime.datetime): Start date for schedule query (UTC).
            to (datetime.datetime): End date for schedule query (UTC).

        Returns:
            Any: Raw protobuf list of Exchange objects with trading sessions, holidays.

        Raises:
            ValueError: Invalid date range (from_ > to).
            Exception: API request failures, network issues.
        """
        with self.client(self.token) as client:
            schedules_raw = client.instruments.trading_schedules(
                from_=from_, to=to
            ).exchanges
        return schedules_raw

    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete instrument information for ticker from cached tickers.

        Performs case-insensitive search through self.tickers cache, returns full
        instrument dictionary with all fields (name, figi, lot, currency, etc.).

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP', 'RUB').

        Returns:
            Optional[Dict[str, Any]]: Full instrument info dict if found, None otherwise.

        Raises:
            KeyError: If matching entry data structure malformed.
        """
        data = list(
            filter(lambda x: x["ticker"] == ticker.upper(), self.tickers)
        )
        if data:
            return data[0]
        return None

    def quantum_sell(self, figi: str, quantity: int) -> float:
        """
        Places a market sell order for the specified quantity of an instrument by FIGI.

        Sends a market order sell request to Tinkoff API using account_id and FIGI,
        then calculates total sell amount including initial commission and returns it.

        Args:
            figi (str): Financial Instrument Global Identifier to sell.
            quantity (int): Number of lots to sell.

        Returns:
            float: Total sell price including commission as a float.

        Raises:
            Exception: If order API call fails or invalid parameters.
        """
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
        """
        Converts float price to Tinkoff Quotation protobuf using library utility.

        Rounds float to 2 decimal places with Decimal precision, then uses
        invest.utils.decimal_to_quotation for exact units/nano conversion.

        Args:
            number (float): Decimal price value (e.g., 123.45).

        Returns:
            Quotation: Tinkoff Quotation protobuf with units and nano fields.

        Raises:
            ValueError: Invalid float conversion or decimal rounding failure.
            Exception: invest.utils.decimal_to_quotation conversion error.
        """
        return invest.utils.decimal_to_quotation(round(Decimal(number), 2))

    def get_order_status(self, order_id: str) -> str:
        """
        Retrieves current execution status of a specific order by order_id.

        Queries Tinkoff API order state endpoint with account_id and order_id,
        returns execution_report_status as string (e.g., 'EXECUTION_REPORT_STATUS_FILLED').

        Args:
            order_id (str): Unique order identifier from post_order response.

        Returns:
            str: Execution status name (e.g., 'EXECUTION_REPORT_STATUS_REJECTED',
                'EXECUTION_REPORT_STATUS_FILLED', etc.).

        Raises:
            ValueError: Invalid order_id or account_id.
            Exception: API request failures, network issues, order not found.
        """
        with self.client(self.token) as client:
            responce = client.orders.get_order_state(
                account_id=self.account_id, order_id=order_id
            )
            return responce.execution_report_status.name

    def get_stop_order_state(self, order_id: str) -> invest.schemas.StopOrder:
        """
        Retrieves specific stop order state by stop_order_id from account's stop orders.

        Fetches all active stop orders for account_id, filters by matching stop_order_id,
        returns first matching StopOrder protobuf object.

        Args:
            order_id (str): Unique stop order identifier (stop_order_id).

        Returns:
            invest.schemas.StopOrder: StopOrder protobuf with full order details.

        Raises:
            IndexError: If no stop order found with given order_id.
            ValueError: Invalid account_id or malformed stop orders response.
        """
        with self.client(self.token) as client:
            resp = client.stop_orders.get_stop_orders(
                account_id=self.account_id
            )
            return list(
                filter(lambda x: x.stop_order_id == order_id, resp.stop_orders)
            )[0]

    def get_all_instruments_tickers(self) -> list[str]:
        """
        Retrieves list of all RUB-denominated instrument tickers from cache.

        Filters self.tickers by currency=="rub" field, extracts ticker strings
        for all matching trading instruments (stocks, etc.).

        Returns:
            list[str]: List of RUB ticker symbols (e.g., ['SBER', 'GAZP', 'RUB']).

        Raises:
            KeyError: If ticker entries lack "ticker" or "currency" fields.
        """
        return [
            instrument["ticker"]
            for instrument in self.tickers
            if instrument["currency"] == "rub"
        ]

    def get_rubles_available(self) -> float:
        """
        Calculates available RUB cash balance after subtracting blocked amounts.

        Finds RUB position in portfolio positions, extracts quantity, subtracts
        blocked RUB lots from get_blocked_lots for net available cash.

        Returns:
            float: Net available RUB balance for trading (quantity - blocked).

        Raises:
            IndexError: If RUB position missing from portfolio.
            KeyError: If RUB position lacks "quantity" field.
            Exception: Portfolio fetch or blocked lots calculation fails.
        """
        return list(
            filter(
                lambda x: x["ticker"] == "RUB",
                self.get_portfolio()["positions"],
            )
        )[0]["quantity"] - self.get_blocked_lots("RUB")

    def ticker_is_available_for_trading(self, ticker: str) -> bool:
        """
        Checks if ticker is currently available for trading based on trading_status.

        Fetches shares from Tinkoff API, finds matching ticker (case-insensitive),
        checks if trading_status is not "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING".

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP').

        Returns:
            bool: True if instrument available for trading, False otherwise.

        Raises:
            IndexError: If ticker not found in shares list.
            AttributeError: If instrument lacks trading_status field.
        """
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

    def round_by_min_price_increment(self, price: float, ticker: str) -> float:
        """
        Rounds price up to nearest valid multiple of instrument's min_price_increment.

        Finds ticker's min_price_increment from cache, converts to float, ceilings price
        to next valid tick size for Tinkoff order price validation.

        Args:
            price (float): Original price to round up.
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP').

        Returns:
            float: Rounded up price matching instrument's price increment.

        Raises:
            IndexError: If ticker not found in self.tickers cache.
            KeyError: If ticker entry lacks "min_price_increment" field.
            ValueError: Invalid min_price_increment conversion.
        """
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

    def get_blocked_lots(self, ticker: str) -> float:
        """
        Retrieves blocked (unavailable for trading) lots quantity for specific ticker.

        Converts ticker to FIGI, fetches portfolio positions, finds matching position,
        extracts and converts blocked_lots Quotation to float (0 if no position).

        Args:
            ticker (str): Stock ticker symbol (e.g., 'SBER', 'GAZP', 'RUB').

        Returns:
            float: Blocked lots quantity for the instrument (0 if no position found).

        Raises:
            ValueError: FIGI lookup fails or blocked_lots conversion error.
            Exception: Portfolio API request failures.
        """
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
