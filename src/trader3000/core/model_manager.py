import pickle
from tinkoff.invest import Client
from dotenv import load_dotenv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import RocCurveDisplay  # для графика
import datetime
import random
from zoneinfo import ZoneInfo

from .data_management import DataManager
from .tinkoff_trading import TinkoffTrading
import sys
from pathlib import Path
MODELS_PATH = Path("/models")  # ← Volume монтируется сюда!


class ModelManager:
    def __init__(
        self,
        models_root = MODELS_PATH
    ):
        self.trading_manager = TinkoffTrading(
            os.getenv("TINVEST_TOKEN"), Client
        )
        self.data_manager = DataManager(self.trading_manager)
        self.models_root = models_root
        print("self.models_root",self.models_root)
        print("os.listdir(self.models_root)",os.listdir(self.models_root))

    def get_experiment_result(self, ticker, shift, increment_p,start_treshold, test_treshold):
        indicators = self.data_manager.common_sql_req(f"SELECT * FROM datasets.{ticker}_hour_total_indicators")
        if increment_p > 0:
            max_min = "MAX"

        else:
            max_min = "MIN"
            
        sql = f"""
        SELECT 
            date
            ,close
            ,{max_min}(close) OVER (ORDER BY date ROWS BETWEEN {shift} PRECEDING AND CURRENT ROW) AS target_within_n
            FROM datasets.{ticker}_hour_total_cost_history
        """
        cost_history = self.data_manager.common_sql_req(sql)    
        features = [i for i in sorted(indicators.columns) if i not in ["date", "index"]]
        cols = features + ["close", "target_within_n"]
    
        #print("cost_history.columns:", cost_history.columns)
        #print("indicators.columns:", indicators.columns)
    
        full_table = indicators.merge(cost_history, on =["date"])
        full_table.set_index("date", inplace = True)
        #print("full_table.columns:", sorted(full_table.columns))
        dataset = full_table[cols]

        if increment_p > 0:
            dataset["target"] = (dataset['target_within_n'] >= dataset['close']*(1+increment_p)).astype(int)

        else:
            dataset["target"] = (dataset['target_within_n'] <= dataset['close']*(1+increment_p)).astype(int)
        
        del dataset["close"]
        del dataset['target_within_n']
        
        X = dataset[features].dropna(); y = dataset['target'].loc[X.index]
        
        X_train = X.loc[:test_treshold]
        X_test  = X.loc[test_treshold:]
    
        y_train = dataset['target'].loc[X_train.index]
        y_test  = dataset['target'].loc[X_test.index]
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # вероятности для класса 1
        
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_result = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC-AUC:", roc_auc)
    
        stats = {
            "accuracy": accuracy,
            "classification_report":classification_report_result,
            "roc_auc":roc_auc
        }
        date_created = str(datetime.datetime.now())[:10]
        
        return model, stats, ticker, features, date_created
    
    def pickle_model(self, model, stats, ticker, features, date_created, estimated_price_change, hold_tics, model_type = "classifier"):
        data = {
            "model": model,
            "stats": stats,
            "ticker": ticker,
            "features": features,
            "date_created": date_created,
            "estimated_price_change":estimated_price_change,
            "hold_tics":hold_tics,
            "model_type": model_type,
        }
        full_path = f"{self.models_root}/{model_type}_{ticker}_{date_created}.pickle".lower()
        with open(full_path, "wb") as file:
            pickle.dump(data,file)
    
    def unpickle_model(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        model = data["model"]
        stats = data["stats"]
        ticker = data["ticker"]
        features = data["features"]
        date_created = data["date_created"]
        estimated_price_change = data["estimated_price_change"]
        model_type = data["model_type"]
        hold_tics = data["hold_tics"]
        return model, stats, ticker, features, model_type, date_created, estimated_price_change, hold_tics
    
    def get_classifiers_paths(self, model_type = "topline"):
        return [i for i in os.listdir(self.models_root) if i.endswith(".pickle") and i.startswith(f"classifier_{model_type}")]
    
    def get_last_data(self, ticker, features, last_candle_check = True):
        features_sql = "\n,".join(i for i in features)
        if last_candle_check:
            target_dt = str(datetime.datetime.now(ZoneInfo("UTC")).replace(minute=0, second=0, microsecond=0))[:19]# - datetime.timedelta(hours = 1)
            print("target_dt",target_dt)
            dt_filter = f"WHERE date >= '{target_dt}'"
        else:
            dt_filter = ""

        sql = f"""
        SELECT 
            max(date)
            FROM datasets.{ticker}_hour_total_indicators
        """
        max_date =  self.data_manager.common_sql_req(sql) 
        print("max_date",max_date.values[0])

        sql = f"""
        SELECT 
            {features_sql}
            FROM datasets.{ticker}_hour_total_indicators
            {dt_filter}
            ORDER BY date DESC
            LIMIT 1
        """
        return self.data_manager.common_sql_req(sql) 
    
    def get_recomendation(self, model, ticker, features):
        last_data = self.get_last_data(ticker, features)
        if len(last_data) > 0:
            print(ticker, "recomendation prepared")
            return bool(model.predict(last_data)[0] == 1)
        else:
            last_candle_dt = self.data_manager.common_sql_req(f"SELECT MAX(date) FROM candles.{ticker}_hour").values[0]
            print(ticker, "no data for recomendation")
            print("last candle dt:", last_candle_dt)
            return False
    
    def get_trading_recomendations(self):
        path_root = self.models_root
        recomendations = []
        paths = self.get_classifiers_paths(model_type = "topline")
        for path in paths:
            model, stats, ticker, features, model_type, date_created, estimated_price_change, hold_tics = self.unpickle_model(f"{path_root}/{path}")
            model_id = f"{date_created}_{ticker}_{estimated_price_change}_{hold_tics}"
            advice = self.get_recomendation(model, ticker, features)
            if advice:
                recomendation_dict = {
                    "ticker": ticker, 
                    "estimated_price_change": estimated_price_change, 
                    "hold_tics": hold_tics,
                    "model_id": model_id
                }
                recomendations.append(recomendation_dict)
        return recomendations
    
    def reaper_routine(
            self, 
            f1_treshold = .90,
            support_treshold_rate = 5,
            shift = 50,
            increment = 0.02,
            start_treshold = None,
            test_treshold = None,
            tickers = [],
            days_train_rate = 1.5,
            days_test_rate = 2,
            ):
        
        if len(tickers) == 0:
            tickers = self.data_manager.get_tickers_from_scope()
            random.shuffle(tickers)

        days_train = int(365 * days_train_rate)
        days_test = int(365 // days_test_rate)

        if start_treshold == None:
            start_treshold = str(datetime.datetime.now() - datetime.timedelta(days=days_train))[:10]
        if test_treshold == None:
            test_treshold = str(datetime.datetime.now() - datetime.timedelta(days=days_test))[:10]
        if increment > 0:
            model_type = "topline"
        elif increment < 0:
            model_type = "bottomline"
        else:
            raise("increment cant be equal to zero")
        
        #Сколько сделок должно быть в тестовом периоде
        #Отсекаем акции, корторые мало торгуются
        support_treshold = days_test * support_treshold_rate
           
        path_root = self.models_root
        process_stats=[]
        
        print(str(datetime.datetime.now())[:10])
        saved_counter = 0
        for t, ticker in enumerate(tickers):
            print("\n","="*10,ticker,"="*10)
            try:
                model, stats, ticker, features, date_created = self.get_experiment_result(ticker, shift, increment,start_treshold, test_treshold)
                f1 = stats["classification_report"]["macro avg"]["f1-score"]
                support = stats["classification_report"]["1"]["support"]
                print("f1:",round(f1,2),"support:",int(support))
        
                process_stats.append(
                    {"ticker": ticker, "stats": stats,"start_treshold":start_treshold,"test_treshold":test_treshold, "shift":shift ,"increment":increment}
                )
                if f1 >= f1_treshold and support >= support_treshold:
                    saved_counter+=1
                    print("saving",saved_counter)
                    self.pickle_model(model, stats, ticker, features, date_created, estimated_price_change = increment, model_type = f"classifier_{model_type}", hold_tics = shift)
            except Exception as e:
                print("error:",e)

    def get_clssifiers_tickers(self, model_type = "topline"):
        model_names = self.get_classifiers_paths(model_type = model_type)
        return [i.split("_")[2].upper() for i in model_names]
    
    def create_classifiers_bottomline(
            self, 
            mode = "add",
            f1_treshold = .90,
            support_treshold_rate = 4,
            shift = 50,
            increment = -0.02,
            start_treshold = None,
            test_treshold = None,
            days_train_rate = 1.5,
            days_test_rate = 2,
            ):
        print("get_classifiers_bottomline")
        topline_classifiers = self.get_clssifiers_tickers(model_type = "topline")
        if mode == "add":
            bottomline_classifiers = self.get_clssifiers_tickers(model_type = "bottomline")
            target_tickers = [i for i in topline_classifiers if i not in bottomline_classifiers]
        elif mode == "replace":
            target_tickers = topline_classifiers
        else:
            raise("mode must be in ['add', 'replace']")
        print(len(target_tickers),"target_tickers")

        self.reaper_routine(
                f1_treshold = f1_treshold,
                support_treshold_rate = support_treshold_rate,
                shift = shift,
                increment = increment,
                start_treshold = start_treshold,
                test_treshold = test_treshold,
                tickers = target_tickers,
                days_train_rate = days_train_rate,
                days_test_rate = days_test_rate,
                )
    