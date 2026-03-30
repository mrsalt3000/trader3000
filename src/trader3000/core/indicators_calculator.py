# imports

import datetime
import warnings

import pandas as pd
import talib
from talib.abstract import (  # type: ignore[attr-defined]
    ADX,
    ADXR,
    APO,
    AROON,
    AROONOSC,
    AVGPRICE,
    BBANDS,
    BOP,
    CCI,
    CDL2CROWS,
    CDL3BLACKCROWS,
    CDL3INSIDE,
    CDL3LINESTRIKE,
    CDL3OUTSIDE,
    CDL3STARSINSOUTH,
    CDL3WHITESOLDIERS,
    CDLABANDONEDBABY,
    CDLADVANCEBLOCK,
    CDLBELTHOLD,
    CDLBREAKAWAY,
    CDLCLOSINGMARUBOZU,
    CDLCONCEALBABYSWALL,
    CDLCOUNTERATTACK,
    CDLDARKCLOUDCOVER,
    CDLDOJI,
    CDLDOJISTAR,
    CDLDRAGONFLYDOJI,
    CDLENGULFING,
    CDLEVENINGDOJISTAR,
    CDLEVENINGSTAR,
    CDLGAPSIDESIDEWHITE,
    CDLGRAVESTONEDOJI,
    CDLHAMMER,
    CDLHANGINGMAN,
    CDLHARAMI,
    CDLHARAMICROSS,
    CDLHIGHWAVE,
    CDLHIKKAKE,
    CDLHIKKAKEMOD,
    CDLHOMINGPIGEON,
    CDLIDENTICAL3CROWS,
    CDLINNECK,
    CDLINVERTEDHAMMER,
    CDLKICKING,
    CDLKICKINGBYLENGTH,
    CDLLADDERBOTTOM,
    CDLLONGLEGGEDDOJI,
    CDLLONGLINE,
    CDLMARUBOZU,
    CDLMATCHINGLOW,
    CDLMATHOLD,
    CDLMORNINGDOJISTAR,
    CDLMORNINGSTAR,
    CDLONNECK,
    CDLPIERCING,
    CDLRICKSHAWMAN,
    CDLRISEFALL3METHODS,
    CDLSEPARATINGLINES,
    CDLSHOOTINGSTAR,
    CDLSHORTLINE,
    CDLSPINNINGTOP,
    CDLSTALLEDPATTERN,
    CDLSTICKSANDWICH,
    CDLTAKURI,
    CDLTASUKIGAP,
    CDLTHRUSTING,
    CDLTRISTAR,
    CDLUNIQUE3RIVER,
    CDLUPSIDEGAP2CROWS,
    CDLXSIDEGAP3METHODS,
    CMO,
    DEMA,
    DX,
    EMA,
    HT_DCPERIOD,
    HT_DCPHASE,
    HT_PHASOR,
    HT_SINE,
    HT_TRENDLINE,
    HT_TRENDMODE,
    KAMA,
    MA,
    MACD,
    MACDEXT,
    MACDFIX,
    MAMA,
    MEDPRICE,
    MFI,
    MIDPOINT,
    MIDPRICE,
    MINUS_DI,
    MINUS_DM,
    MOM,
    PLUS_DI,
    PLUS_DM,
    PPO,
    ROC,
    ROCP,
    ROCR,
    ROCR100,
    RSI,
    SAR,
    SAREXT,
    SMA,
    STOCH,
    STOCHF,
    STOCHRSI,
    T3,
    TEMA,
    TRIMA,
    TRIX,
    TYPPRICE,
    ULTOSC,
    WCLPRICE,
    WILLR,
    WMA,
    Function,
)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class IndicatorsCalculator:
    def __init__(self, data):
        self.data = data
        #self.add_calendar_features()
        if self.data.index.name != "date":
            self.data.set_index("date", inplace=True)
        self.data.sort_index(inplace=True)
        if "index" in  self.data.columns:
            del data["index"]

    def add_calendar_features(self):
        # Календарь
        self.data["date"] = self.data["date"].apply(
            lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        )
        self.data["set_calendar_is_monday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 1)
        )
        self.data["set_calendar_is_tuesday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 2)
        )
        self.data["set_calendar_is_wednesday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 3)
        )
        self.data["set_calendar_is_thursday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 4)
        )
        self.data["set_calendar_is_friday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 5)
        )
        self.data["set_calendar_is_saturday"] = self.data["date"].apply(
            lambda x: int(x.isoweekday() == 6)
        )

        for w in range(1, 52):
            self.data[f"set_calendar_is_week_{w}"] = self.data["date"].apply(
                lambda x: int(x.isocalendar()[1] == w)
            )
        self.data["date"] = self.data["date"].apply(lambda x: str(x))

    def print_all_indicators(self):
        for group, funcs in talib.get_function_groups().items():
            print(group)
            print("-----------------------------------------")
            for func in funcs:
                f = Function(func)
                print("{} - {}".format(func, f.info["display_name"]))
            print()

    def calculate_indicators(self):
        # simple moving average
        for timeperiod in [5, 15, 30, 60, 120]:
            self.data[f"sma{timeperiod}"] = SMA(
                self.data["close"], timeperiod=timeperiod
            )  # / self.data["close"]

        # Cycle Indicators
        # -----------------------------------------
        # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
        self.data["ht_dcperiod"] = HT_DCPERIOD(self.data["close"])

        # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
        self.data["ht_dcphase"] = HT_DCPHASE(self.data["close"])

        # HT_PHASOR - Hilbert Transform - Phasor Components
        self.data["ht_phasor_inphase"], self.data["ht_phasor_quadrature"] = (
            HT_PHASOR(self.data["close"])
        )

        # HT_SINE - Hilbert Transform - SineWave
        self.data["ht_sine_sine"], self.data["ht_sine_leadsine"] = HT_SINE(
            self.data["close"]
        )

        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        self.data["ht_trendmode"] = HT_TRENDMODE(self.data["close"])

        # Momentum Indicators
        # -----------------------------------------
        # ADX - Average Directional Movement Index
        self.data["adx"] = ADX(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ADXR - Average Directional Movement Index Rating
        self.data["adxr"] = ADXR(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # APO - Absolute Price Oscillator
        self.data["apo"] = APO(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # AROON - Aroon
        self.data["aroon_aroondown"], self.data["aroon_aroonup"] = AROON(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # AROONOSC - Aroon Oscillator
        self.data["aroonosc"] = AROONOSC(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # BOP - Balance Of Power
        self.data["bop"] = BOP(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CCI - Commodity Channel Index
        self.data["cci"] = CCI(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CMO - Chande Momentum Oscillator
        self.data["cmo"] = CMO(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # DX - Directional Movement Index
        self.data["dx"] = DX(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACD - Moving Average Convergence/Divergence
        (
            self.data["macd_macd"],
            self.data["macd_macdsignal"],
            self.data["macd_macdhist"],
        ) = MACD(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACDEXT - MACD with controllable MA type
        (
            self.data["macdext_macd"],
            self.data["macdext_macdsignal"],
            self.data["macdext_macdhist"],
        ) = MACDEXT(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        (
            self.data["macdfix_macd"],
            self.data["macdfix_macdsignal"],
            self.data["macdfix_macdhist"],
        ) = MACDFIX(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MFI - Money Flow Index
        self.data["mfi"] = MFI(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            self.data["volume"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MINUS_DI - Minus Directional Indicator
        self.data["minus_di"] = MINUS_DI(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MINUS_DM - Minus Directional Movement
        self.data["minus_dm"] = MINUS_DM(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MOM - Momentum
        self.data["mom"] = MOM(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PLUS_DI - Plus Directional Indicator
        self.data["plus_di"] = PLUS_DI(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PLUS_DM - Plus Directional Movement
        self.data["plus_dm"] = PLUS_DM(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PPO - Percentage Price Oscillator
        self.data["ppo"] = PPO(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROC - Rate of change : ((price/prevPrice)-1)*100
        self.data["roc"] = ROC(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        self.data["rocp"] = ROCP(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCR - Rate of change ratio: (price/prevPrice)
        self.data["rocr"] = ROCR(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        self.data["rocr100"] = ROCR100(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # RSI - Relative Strength Index
        self.data["rsi"] = RSI(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCH - Stochastic
        self.data["stoch_slowk"], self.data["stoch_slowd"] = STOCH(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCHF - Stochastic Fast
        self.data["stochf_fastk"], self.data["stochf_fastd"] = STOCHF(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCHRSI - Stochastic Relative Strength Index
        self.data["stochrsi_fastk"], self.data["stochrsi_fastd"] = STOCHRSI(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        self.data["trix"] = TRIX(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ULTOSC - Ultimate Oscillator
        self.data["ultosc"] = ULTOSC(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # WILLR - Williams' %R
        self.data["willr"] = WILLR(
            self.data["high"], self.data["low"], self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # Overlap Studies
        # -----------------------------------------
        # BBANDS - Bollinger Bands
        (
            self.data["bbands_upperband"],
            self.data["bbands_middleband"],
            self.data["bbands_lowerband"],
        ) = BBANDS(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # DEMA - Double Exponential Moving Average
        self.data["dema"] = DEMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # EMA - Exponential Moving Average
        self.data["ema"] = EMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
        self.data["ht_trendline"] = HT_TRENDLINE(self.data["close"])

        # KAMA - Kaufman Adaptive Moving Average
        self.data["kama"] = KAMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MA - Moving average
        self.data["ma"] = MA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MAMA - MESA Adaptive Moving Average
        self.data["mama_mama"], self.data["mama_fama"] = MAMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MIDPOINT - MidPoint over period
        self.data["midpoint"] = MIDPOINT(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MIDPRICE - Midpoint Price over period
        self.data["midprice"] = MIDPRICE(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SAR - Parabolic SAR
        self.data["sar"] = SAR(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SAREXT - Parabolic SAR - Extended
        self.data["sarext"] = SAREXT(
            self.data["high"], self.data["low"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SMA - Simple Moving Average
        self.data["sma"] = SMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # T3 - Triple Exponential Moving Average (T3)
        self.data["t3"] = T3(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TEMA - Triple Exponential Moving Average
        self.data["tema"] = TEMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TRIMA - Triangular Moving Average
        self.data["trima"] = TRIMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # WMA - Weighted Moving Average
        self.data["wma"] = WMA(
            self.data["close"]
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # Pattern Recognition
        # -----------------------------------------
        # CDL2CROWS - Two Crows
        self.data["cdl2crows"] = CDL2CROWS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3BLACKCROWS - Three Black Crows
        self.data["cdl3blackcrows"] = CDL3BLACKCROWS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3INSIDE - Three Inside Up/Down
        self.data["cdl3inside"] = CDL3INSIDE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3LINESTRIKE - Three-Line Strike
        self.data["cdl3linestrike"] = CDL3LINESTRIKE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3OUTSIDE - Three Outside Up/Down
        self.data["cdl3outside"] = CDL3OUTSIDE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3STARSINSOUTH - Three Stars In The South
        self.data["cdl3starsinsouth"] = CDL3STARSINSOUTH(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDL3WHITESOLDIERS - Three Advancing White Soldiers
        self.data["cdl3whitesoldiers"] = CDL3WHITESOLDIERS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLABANDONEDBABY - Abandoned Baby
        self.data["cdlabandonedbaby"] = CDLABANDONEDBABY(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLADVANCEBLOCK - Advance Block
        self.data["cdladvanceblock"] = CDLADVANCEBLOCK(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLBELTHOLD - Belt-hold
        self.data["cdlbelthold"] = CDLBELTHOLD(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLBREAKAWAY - Breakaway
        self.data["cdlbreakaway"] = CDLBREAKAWAY(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLCLOSINGMARUBOZU - Closing Marubozu
        self.data["cdlclosingmarubozu"] = CDLCLOSINGMARUBOZU(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLCONCEALBABYSWALL - Concealing Baby Swallow
        self.data["cdlconcealbabyswall"] = CDLCONCEALBABYSWALL(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLCOUNTERATTACK - Counterattack
        self.data["cdlcounterattack"] = CDLCOUNTERATTACK(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLDARKCLOUDCOVER - Dark Cloud Cover
        self.data["cdldarkcloudcover"] = CDLDARKCLOUDCOVER(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLDOJI - Doji
        self.data["cdldoji"] = CDLDOJI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLDOJISTAR - Doji Star
        self.data["cdldojistar"] = CDLDOJISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLDRAGONFLYDOJI - Dragonfly Doji
        self.data["cdldragonflydoji"] = CDLDRAGONFLYDOJI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLENGULFING - Engulfing Pattern
        self.data["cdlengulfing"] = CDLENGULFING(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLEVENINGDOJISTAR - Evening Doji Star
        self.data["cdleveningdojistar"] = CDLEVENINGDOJISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLEVENINGSTAR - Evening Star
        self.data["cdleveningstar"] = CDLEVENINGSTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
        self.data["cdlgapsidesidewhite"] = CDLGAPSIDESIDEWHITE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLGRAVESTONEDOJI - Gravestone Doji
        self.data["cdlgravestonedoji"] = CDLGRAVESTONEDOJI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHAMMER - Hammer
        self.data["cdlhammer"] = CDLHAMMER(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHANGINGMAN - Hanging Man
        self.data["cdlhangingman"] = CDLHANGINGMAN(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHARAMI - Harami Pattern
        self.data["cdlharami"] = CDLHARAMI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHARAMICROSS - Harami Cross Pattern
        self.data["cdlharamicross"] = CDLHARAMICROSS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHIGHWAVE - High-Wave Candle
        self.data["cdlhighwave"] = CDLHIGHWAVE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHIKKAKE - Hikkake Pattern
        self.data["cdlhikkake"] = CDLHIKKAKE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHIKKAKEMOD - Modified Hikkake Pattern
        self.data["cdlhikkakemod"] = CDLHIKKAKEMOD(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLHOMINGPIGEON - Homing Pigeon
        self.data["cdlhomingpigeon"] = CDLHOMINGPIGEON(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLIDENTICAL3CROWS - Identical Three Crows
        self.data["cdlidentical3crows"] = CDLIDENTICAL3CROWS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLINNECK - In-Neck Pattern
        self.data["cdlinneck"] = CDLINNECK(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLINVERTEDHAMMER - Inverted Hammer
        self.data["cdlinvertedhammer"] = CDLINVERTEDHAMMER(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLKICKING - Kicking
        self.data["cdlkicking"] = CDLKICKING(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
        self.data["cdlkickingbylength"] = CDLKICKINGBYLENGTH(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLLADDERBOTTOM - Ladder Bottom
        self.data["cdlladderbottom"] = CDLLADDERBOTTOM(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLLONGLEGGEDDOJI - Long Legged Doji
        self.data["cdllongleggeddoji"] = CDLLONGLEGGEDDOJI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLLONGLINE - Long Line Candle
        self.data["cdllongline"] = CDLLONGLINE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLMARUBOZU - Marubozu
        self.data["cdlmarubozu"] = CDLMARUBOZU(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLMATCHINGLOW - Matching Low
        self.data["cdlmatchinglow"] = CDLMATCHINGLOW(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLMATHOLD - Mat Hold
        self.data["cdlmathold"] = CDLMATHOLD(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLMORNINGDOJISTAR - Morning Doji Star
        self.data["cdlmorningdojistar"] = CDLMORNINGDOJISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLMORNINGSTAR - Morning Star
        self.data["cdlmorningstar"] = CDLMORNINGSTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLONNECK - On-Neck Pattern
        self.data["cdlonneck"] = CDLONNECK(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLPIERCING - Piercing Pattern
        self.data["cdlpiercing"] = CDLPIERCING(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLRICKSHAWMAN - Rickshaw Man
        self.data["cdlrickshawman"] = CDLRICKSHAWMAN(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLRISEFALL3METHODS - Rising/Falling Three Methods
        self.data["cdlrisefall3methods"] = CDLRISEFALL3METHODS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSEPARATINGLINES - Separating Lines
        self.data["cdlseparatinglines"] = CDLSEPARATINGLINES(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSHOOTINGSTAR - Shooting Star
        self.data["cdlshootingstar"] = CDLSHOOTINGSTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSHORTLINE - Short Line Candle
        self.data["cdlshortline"] = CDLSHORTLINE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSPINNINGTOP - Spinning Top
        self.data["cdlspinningtop"] = CDLSPINNINGTOP(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSTALLEDPATTERN - Stalled Pattern
        self.data["cdlstalledpattern"] = CDLSTALLEDPATTERN(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLSTICKSANDWICH - Stick Sandwich
        self.data["cdlsticksandwich"] = CDLSTICKSANDWICH(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
        self.data["cdltakuri"] = CDLTAKURI(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLTASUKIGAP - Tasuki Gap
        self.data["cdltasukigap"] = CDLTASUKIGAP(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLTHRUSTING - Thrusting Pattern
        self.data["cdlthrusting"] = CDLTHRUSTING(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLTRISTAR - Tristar Pattern
        self.data["cdltristar"] = CDLTRISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLUNIQUE3RIVER - Unique 3 River
        self.data["cdlunique3river"] = CDLUNIQUE3RIVER(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
        self.data["cdlupsidegap2crows"] = CDLUPSIDEGAP2CROWS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
        self.data["cdlxsidegap3methods"] = CDLXSIDEGAP3METHODS(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )

        # Price Transform
        # -----------------------------------------
        # AVGPRICE - Average Price
        self.data["avgprice"] = AVGPRICE(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
        )  # / self.data["close"]

        # MEDPRICE - Median Price
        self.data["medprice"] = MEDPRICE(
            self.data["high"], self.data["low"]
        )  # / self.data["close"]

        # TYPPRICE - Typical Price
        self.data["typprice"] = TYPPRICE(
            self.data["high"], self.data["low"], self.data["close"]
        )  # / self.data["close"]

        # WCLPRICE - Weighted Close Price
        self.data["wclprice"] = WCLPRICE(
            self.data["high"], self.data["low"], self.data["close"]
        )  # / self.data["close"]

        # ДАЛЕЕ УМНОЖИМ ВСЕ ПАРАМЕТРЫ ПЕРИОДА НА 24, т.к. работаем с часами
        # simple moving average
        for timeperiod in [
            2 * 24,
            3 * 24,
            5 * 24,
            8 * 24,
            13 * 24,
            21 * 24,
            34 * 24,
            55 * 24,
            89 * 24,
            144 * 24,
        ]:
            self.data[f"sma{timeperiod}"] = SMA(
                self.data["close"], timeperiod=timeperiod
            )  # / self.data["close"]
        # Загрушки вместо самых длинных, В дальнейшем убрать
        for timeperiod in [55 * 24, 89 * 24, 144 * 24]:
            self.data[f"sma{timeperiod}"] = [0] * len(
                self.data
            )  # / self.data["close"]

        # Momentum Indicators
        # -----------------------------------------
        # ADX - Average Directional Movement Index
        self.data["adx_x24"] = ADX(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ADXR - Average Directional Movement Index Rating
        self.data["adxr_x24"] = ADXR(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # APO - Absolute Price Oscillator
        self.data["apo_x24"] = APO(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # AROON - Aroon
        self.data["aroon_aroondown_x24"], self.data["aroon_aroonup_x24"] = (
            AROON(
                self.data["high"],
                self.data["low"],
                fastperiod=12 * 24,
                slowperiod=26 * 24,
                signalperiod=9 * 24,
            )
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # AROONOSC - Aroon Oscillator
        self.data["aroonosc_x24"] = AROONOSC(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CCI - Commodity Channel Index
        self.data["cci_x24"] = CCI(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CMO - Chande Momentum Oscillator
        self.data["cmo_x24"] = CMO(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # DX - Directional Movement Index
        self.data["dx_x24"] = DX(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACD - Moving Average Convergence/Divergence
        (
            self.data["macd_macd_x24"],
            self.data["macd_macdsignal_x24"],
            self.data["macd_macdhist_x24"],
        ) = MACD(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACDEXT - MACD with controllable MA type
        (
            self.data["macdext_macd_x24"],
            self.data["macdext_macdsignal_x24"],
            self.data["macdext_macdhist_x24"],
        ) = MACDEXT(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        (
            self.data["macdfix_macd_x24"],
            self.data["macdfix_macdsignal_x24"],
            self.data["macdfix_macdhist_x24"],
        ) = MACDFIX(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MFI - Money Flow Index
        self.data["mfi_x24"] = MFI(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            self.data["volume"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MINUS_DI - Minus Directional Indicator
        self.data["minus_di_x24"] = MINUS_DI(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MINUS_DM - Minus Directional Movement
        self.data["minus_dm_x24"] = MINUS_DM(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MOM - Momentum
        self.data["mom_x24"] = MOM(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PLUS_DI - Plus Directional Indicator
        self.data["plus_di_x24"] = PLUS_DI(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PLUS_DM - Plus Directional Movement
        self.data["plus_dm_x24"] = PLUS_DM(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # PPO - Percentage Price Oscillator
        self.data["ppo_x24"] = PPO(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROC - Rate of change : ((price/prevPrice)-1)*100
        self.data["roc_x24"] = ROC(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        self.data["rocp_x24"] = ROCP(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCR - Rate of change ratio: (price/prevPrice)
        self.data["rocr_x24"] = ROCR(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        self.data["rocr100_x24"] = ROCR100(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # RSI - Relative Strength Index
        self.data["rsi_x24"] = RSI(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCH - Stochastic
        self.data["stoch_slowk_x24"], self.data["stoch_slowd_x24"] = STOCH(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCHF - Stochastic Fast
        self.data["stochf_fastk_x24"], self.data["stochf_fastd_x24"] = STOCHF(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # STOCHRSI - Stochastic Relative Strength Index
        self.data["stochrsi_fastk_x24"], self.data["stochrsi_fastd_x24"] = (
            STOCHRSI(
                self.data["close"],
                fastperiod=12 * 24,
                slowperiod=26 * 24,
                signalperiod=9 * 24,
            )
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        self.data["trix_x24"] = TRIX(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # ULTOSC - Ultimate Oscillator
        self.data["ultosc_x24"] = ULTOSC(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # WILLR - Williams' %R
        self.data["willr_x24"] = WILLR(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # Overlap Studies
        # -----------------------------------------
        # BBANDS - Bollinger Bands
        (
            self.data["bbands_upperband_x24"],
            self.data["bbands_middleband_x24"],
            self.data["bbands_lowerband_x24"],
        ) = BBANDS(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # DEMA - Double Exponential Moving Average
        self.data["dema_x24"] = DEMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # EMA - Exponential Moving Average
        self.data["ema_x24"] = EMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # KAMA - Kaufman Adaptive Moving Average
        self.data["kama_x24"] = KAMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MA - Moving average
        self.data["ma_x24"] = MA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MAMA - MESA Adaptive Moving Average
        self.data["mama_mama_x24"], self.data["mama_fama_x24"] = MAMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MIDPOINT - MidPoint over period
        self.data["midpoint_x24"] = MIDPOINT(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # MIDPRICE - Midpoint Price over period
        self.data["midprice_x24"] = MIDPRICE(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SAR - Parabolic SAR
        self.data["sar_x24"] = SAR(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SAREXT - Parabolic SAR - Extended
        self.data["sarext_x24"] = SAREXT(
            self.data["high"],
            self.data["low"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # SMA - Simple Moving Average
        self.data["sma_x24"] = SMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # T3 - Triple Exponential Moving Average (T3)
        self.data["t3_x24"] = T3(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TEMA - Triple Exponential Moving Average
        self.data["tema_x24"] = TEMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # TRIMA - Triangular Moving Average
        self.data["trima_x24"] = TRIMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # WMA - Weighted Moving Average
        self.data["wma_x24"] = WMA(
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # Pattern Recognition
        # -----------------------------------------
        # CDLABANDONEDBABY - Abandoned Baby
        self.data["cdlabandonedbaby_x24"] = CDLABANDONEDBABY(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLDARKCLOUDCOVER - Dark Cloud Cover
        self.data["cdldarkcloudcover_x24"] = CDLDARKCLOUDCOVER(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLEVENINGDOJISTAR - Evening Doji Star
        self.data["cdleveningdojistar_x24"] = CDLEVENINGDOJISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLEVENINGSTAR - Evening Star
        self.data["cdleveningstar_x24"] = CDLEVENINGSTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLMATHOLD - Mat Hold
        self.data["cdlmathold_x24"] = CDLMATHOLD(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLMORNINGDOJISTAR - Morning Doji Star
        self.data["cdlmorningdojistar_x24"] = CDLMORNINGDOJISTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # CDLMORNINGSTAR - Morning Star
        self.data["cdlmorningstar_x24"] = CDLMORNINGSTAR(
            self.data["open"],
            self.data["high"],
            self.data["low"],
            self.data["close"],
            fastperiod=12 * 24,
            slowperiod=26 * 24,
            signalperiod=9 * 24,
        )  # params: fastperiod = 12, slowperiod = 26, signalperiod = 9

        # Price Transform
        # -----------------------------------------
        # AVGPRICE - Average Price
        self.data["avgprice_div_close"] = (
            AVGPRICE(
                self.data["open"],
                self.data["high"],
                self.data["low"],
                self.data["close"],
            )
            / self.data["close"]
        )

        # MEDPRICE - Median Price
        self.data["medprice_div_close"] = (
            MEDPRICE(self.data["high"], self.data["low"]) / self.data["close"]
        )

        # TYPPRICE - Typical Price
        self.data["typprice_div_close"] = (
            TYPPRICE(self.data["high"], self.data["low"], self.data["close"])
            / self.data["close"]
        )

        # WCLPRICE - Weighted Close Price
        self.data["wclprice_div_close"] = (
            WCLPRICE(self.data["high"], self.data["low"], self.data["close"])
            / self.data["close"]
        )
        self.data.sort_values(by=["date"], ascending=True, inplace=True)
        return self.data.dropna().copy()
