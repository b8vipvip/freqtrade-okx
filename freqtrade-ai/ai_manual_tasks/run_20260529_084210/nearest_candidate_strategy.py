from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.vendor.qtpylib import indicators as qtpylib


class MultiCoin_AI_Strategy_20260529_061342_v003(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False

    minimal_roi = {
        "0": 0.012,
        "15": 0.008,
        "35": 0.005,
        "60": 0.003,
        "120": 0.001,
    }

    stoploss = -0.025

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    process_only_new_candles = True
    startup_candle_count = 220

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 10,
                "stop_duration_candles": 6,
                "max_allowed_drawdown": 0.02,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 2,
                "stop_duration_candles": 8,
                "only_per_pair": True,
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_histogram"] = macd["macdhist"]

        dataframe["volume_mean_20"] = dataframe["volume"].rolling(window=20, min_periods=20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        long_condition = (
            (dataframe["ema_50"] > dataframe["ema_200"]) &
            (dataframe["ema_12"] > dataframe["ema_26"]) &
            (qtpylib.crossed_above(dataframe["macd_histogram"], 0)) &
            (dataframe["rsi_14"] > 45) &
            (dataframe["rsi_14"] < 70) &
            (dataframe["adx"] > 20) &
            (dataframe["volume"] > dataframe["volume_mean_20"] * 2.2) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[long_condition, ["enter_long", "enter_tag"]] = (
            1,
            "macd_cross_momentum",
        )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
