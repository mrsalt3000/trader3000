CREATE SCHEMA IF NOT EXISTS transaction;
CREATE TABLE IF NOT EXISTS transaction.transactions (
    id                      BIGSERIAL PRIMARY KEY,
    buy_dt                  TEXT,
    num_quants              INTEGER,
    sell_dt_fact            TEXT,
    status                  TEXT,
    quant_cost              INTEGER,
    sell_cost               NUMERIC(18, 6),
    model_id                TEXT,
    order_id                TEXT,
    order_request_id        TEXT,
    figi                    TEXT,
    ticker                  TEXT,
    execution_report_status TEXT,
    lots_requested          INTEGER,
    lots_executed           INTEGER,
    initial_order_price     NUMERIC(18, 6),
    executed_order_price    NUMERIC(18, 6),
    total_order_amount      NUMERIC(18, 6),
    initial_commission      NUMERIC(18, 6),
    initial_security_price  NUMERIC(18, 6),
    executed_commission     NUMERIC(18, 6),
    direction               TEXT,
    order_type              TEXT,
    sell_order_id           TEXT,
    estimated_price_change  NUMERIC(18, 6),
    estimated_sell_dt       TEXT,
    estimated_sell_date     TEXT,
    stop_loss_order_id      TEXT,
    account_id              TEXT
);

CREATE SCHEMA IF NOT EXISTS trading;
CREATE TABLE IF NOT EXISTS trading.schedules (

    schedule       TEXT,
    date           DATE,
    is_trading_day INTEGER,
    start_time     TIMESTAMP,
    end_time       TIMESTAMP,
    work_seconds   INTEGER
);

CREATE SCHEMA IF NOT EXISTS datasets;

CREATE SCHEMA IF NOT EXISTS tickers;
