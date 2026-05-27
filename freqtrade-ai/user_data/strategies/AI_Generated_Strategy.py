# -*- coding: utf-8 -*-
"""Relaxed long-only spot strategy with 5m execution and optional 1h trend context."""

from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import talib.abstract as ta


class AI_Generated_Strategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 240

    minimal_roi = {
        "0": 0.03,
        "45": 0.015,
        "120": 0.008,
        "240": 0,
    }

    stoploss = -0.06
    process_only_new_candles = True

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 1h filter: keep only one relaxed condition
                (
                    (dataframe["close_1h"] > dataframe["ema200_1h"])
                    | (dataframe["rsi_1h"] > 45)
                )
                & (dataframe["close"] > dataframe["ema20"])
                & (dataframe["rsi"] > 35)
                & (dataframe["rsi"] < 72)
                & (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["adx"] > 10)
                & (dataframe["atr_pct"] > 0.001)
                & (dataframe["atr_pct"] < 0.06)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["rsi"] > 78)
                    | ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 45))
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
