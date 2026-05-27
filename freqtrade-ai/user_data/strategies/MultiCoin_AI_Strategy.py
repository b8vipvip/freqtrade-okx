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
        "0": 0.032,
        "90": 0.020,
        "240": 0.012,
        "360": 0.0,
    }

    # Exits rely primarily on ROI/stoploss to avoid loss-heavy signal exits.
    use_exit_signal = False
    stoploss = -0.045

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2,
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
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # Core gate: keep strong 1h bias and positive liquidity.
        base_guard = (
            (dataframe["close_1h"] > dataframe["ema_200_1h"])
            & (dataframe["rsi_1h"] > 48)
            & (dataframe["close"] > dataframe["ema_50"])
            & (dataframe["ema_20"] > dataframe["ema_50"])
            & (dataframe["rsi"] >= 42)
            & (dataframe["rsi"] <= 66)
            & (dataframe["adx"] > 15)
            & (dataframe["atr_pct"] > 0.0018)
            & (dataframe["atr_pct"] < 0.028)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["volume"] > 0)
        )

        # Continuation setup: slightly looser to raise total trade count.
        continuation_entry = (
            base_guard
            & (dataframe["close"] > dataframe["ema_100"])
            & (dataframe["ema_50"] > dataframe["ema_100"])
            & (dataframe["macd"] > 0)
        )

        # Pullback-recovery setup: buy dip recoveries inside uptrend, avoiding weak tape.
        pullback_entry = (
            base_guard
            & (dataframe["close"] > dataframe["ema_200"])
            & (dataframe["close"] > (dataframe["ema_20"] * 0.995))
            & (dataframe["rsi"] >= 44)
            & (dataframe["rsi"] <= 58)
            & (dataframe["adx"] > 17)
            & (dataframe["macd"] > (dataframe["macdsignal"] * 0.985))
        )

        entry_mask = continuation_entry | pullback_entry

        # DOGE underperformed, so apply stricter filter to cut weak/flat entries.
        if "DOGE/USDT" in pair:
            doge_filter = (
                (dataframe["close_1h"] > (dataframe["ema_200_1h"] * 1.002))
                & (dataframe["rsi_1h"] > 52)
                & (dataframe["adx"] > 20)
                & (dataframe["atr_pct"] > 0.0022)
                & (dataframe["rsi"] >= 46)
            )
            entry_mask = entry_mask & doge_filter

        dataframe.loc[entry_mask, "enter_long"] = 1

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
