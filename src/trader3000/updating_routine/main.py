import os
import random
from sklearn import preprocessing
from itertools import chain
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


import datetime
import pickle
from ..core.tinkoff_trading import TinkoffTrading
from tinkoff.invest import Client


TOKEN = os.getenv("TINVEST_TOKEN")
trading_manager = TinkoffTrading(TOKEN,Client)
from ..core.data_management import DataManager
data_manager = DataManager(trading_manager) 


import time

def run_main_update():
  print("Updating routine started")
  while True:
    now = str(datetime.datetime.now())[:16]
    print("Starting updating routine at",now)
    try:
      print("update_schedules")
      data_manager.update_schedules()
    except Exception as e:
      print("Error in update_schedules\n",e)
    try:
      now = str(datetime.datetime.now())[:16]
      print("update_training_data",now)
      data_manager.update_training_data(data_manager.get_all_candles_loaded())
      now = str(datetime.datetime.now())[:16]
      print("Finished update_training_data at",now)
    except Exception as e:
      print("Error in update_training_data\n",e)
    try:
      now = str(datetime.datetime.now())[:16]
      print("recalculate_indicators",now)
      data_manager.recalculate_indicators(mode="total")
      now = str(datetime.datetime.now())[:16]
      print("Finished recalculate_indicators at",now)
    except Exception as e:
      print("Error in updating routine\n",e)        
    print("sleeping till next iteration")
    time.sleep(60*60*24)

if __name__ == "__main__":
  run_main_update()