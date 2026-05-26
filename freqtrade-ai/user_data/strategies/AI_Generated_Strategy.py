# -*- coding: utf-8 -*-
"""Default placeholder strategy. This file can be overwritten by ai_tools/generate_strategy.py."""

from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta


class AI_Generated_Strategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 80

    minimal_roi = {
        "0": 0.03,
        "45": 0.015,
        "120": 0.008,
        "240": 0
    }

    stoploss = -0.06
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema20"] > dataframe["ema50"])
                & (dataframe["rsi"] >= 40)
                & (dataframe["rsi"] <= 62)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > 72)
                | (dataframe["ema20"] < dataframe["ema50"])
            )
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
