from ..core.trader import Trader

def run_trader_routine():
    print("Trader routine started")
    trader = Trader()
    trader.trader_routine()

if __name__ == "__main__":
    run_trader_routine()
