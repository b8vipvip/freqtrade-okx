# -*- coding: utf-8 -*-
"""Multi-coin spot long-only strategy (5m + 1h context).

Targets:
- BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT

Design constraints:
- Spot long only, no short, no leverage.
- No martingale / no infinite position add-ons.
- No OpenAI API calls.
- No manual trade-history reads.
- Shared indicator logic across all pairs.
"""

from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import talib.abstract as ta


class MultiCoin_AI_Strategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = False
    startup_candle_count = 240
    process_only_new_candles = True

    # Keep ROI-based exits available.
    minimal_roi = {
        "0": 0.030,
        "60": 0.018,
        "180": 0.010,
        "360": 0.0,
    }

    # Slightly wider stop to avoid tiny-pullback shakeouts.
    stoploss = -0.07

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3,
        }
    ]

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 1h context: trend-friendly or at least not weak
                ((dataframe["close_1h"] > dataframe["ema200_1h"]) | (dataframe["rsi_1h"] > 46))
                # 5m momentum/trend alignment (relaxed for more opportunities)
                & (dataframe["close"] > dataframe["ema20"])
                & (dataframe["rsi"] >= 34)
                & (dataframe["rsi"] <= 72)
                & (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    # Confirmed weakness instead of tiny pullback
                    ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 40))
                    | ((dataframe["ema20"] < dataframe["ema50"]) & (dataframe["macd"] < dataframe["macdsignal"]))
                    | ((dataframe["close"] < dataframe["ema100"]) & (dataframe["rsi"] < 45))
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
