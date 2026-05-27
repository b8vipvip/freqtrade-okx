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

    # Exits rely primarily on ROI/stoploss to avoid loss-heavy signal exits.
    use_exit_signal = False
    stoploss = -0.065

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3,
        }
    ]

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 1h context: trade only with medium/long trend tailwind
                (dataframe["close_1h"] > dataframe["ema_200_1h"])
                & (dataframe["rsi_1h"] > 50)
                # 5m alignment: reduce low-quality churn entries
                & (dataframe["close"] > dataframe["ema_50"])
                & (dataframe["ema_20"] > dataframe["ema_50"])
                & (dataframe["rsi"] >= 45)
                & (dataframe["rsi"] <= 68)
                & (dataframe["macd"] > dataframe["macdsignal"])
                & (dataframe["macd"] > 0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    # Extreme trend break only (kept as fallback if use_exit_signal is enabled)
                    (dataframe["close"] < dataframe["ema_200"])
                    | ((dataframe["close"] < dataframe["ema_100"]) & (dataframe["rsi"] < 35))
                    | (
                        (dataframe["ema_20"] < dataframe["ema_50"])
                        & (dataframe["ema_50"] < dataframe["ema_100"])
                        & (dataframe["macd"] < dataframe["macdsignal"])
                    )
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
