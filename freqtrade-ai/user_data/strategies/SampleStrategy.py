# -*- coding: utf-8 -*-
"""Sample spot-only strategy for dry-run/backtesting."""

from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta


class SampleStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 60

    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "90": 0.01,
        "180": 0
    }

    stoploss = -0.08

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
                & (dataframe["rsi"] >= 35)
                & (dataframe["rsi"] <= 65)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > 75)
                | (dataframe["ema20"] < dataframe["ema50"])
            )
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
