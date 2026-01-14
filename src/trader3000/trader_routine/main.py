from ..core.trader import Trader

def run_trader_routine():
    print("Trader routine started")
    trader = Trader(quant_cost=1000)
    trader.trader_routine()

if __name__ == "__main__":
    run_trader_routine()
