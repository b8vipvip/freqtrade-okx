# -*- coding: utf-8 -*-
"""BTC/USDT spot long-only strategy (5m + relaxed 1h context).

Design goals:
- Spot long-only (no short, no leverage logic).
- No martingale / no infinite position add-ons.
- Indicators based only on OHLCV + technical indicators.
- Keep trade count moderate and avoid noisy exits.
"""

from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import talib.abstract as ta


class BTC_Only_AI_Strategy(IStrategy):
    INTERFACE_VERSION = 3

    # Main execution timeframe
    timeframe = "5m"

    # Explicitly disallow shorting
    can_short = False

    # Enough candles for 1h EMA200 and 5m indicators
    startup_candle_count = 240

    # Keep ROI exits as a main profit-taking source
    minimal_roi = {
        "0": 0.030,
        "60": 0.018,
        "180": 0.010,
        "360": 0.0,
    }

    # Risk control for spot long entries
    stoploss = -0.06
    process_only_new_candles = True

    # Reduce overtrading a bit (not HFT-like)
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 6,
        }
    ]

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """1h context: loose bullish / neutral-strong filter."""
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """5m execution indicators."""
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter only BTC/USDT under relaxed trend + momentum confirmation."""

        # This strategy is intentionally designed only for BTC/USDT.
        if metadata.get("pair") != "BTC/USDT":
            dataframe["enter_long"] = 0
            return dataframe

        dataframe.loc[
            (
                # 1h relaxed environment: either above EMA200 or RSI not weak
                (
                    (dataframe["close_1h"] > dataframe["ema200_1h"])
                    | (dataframe["rsi_1h"] > 48)
                )
                # 5m structure: above EMA50 to avoid weak downtrend entries
                & (dataframe["close"] > dataframe["ema50"])
                # Requested RSI zone: avoid oversold knife-catching and overbought chasing
                & (dataframe["rsi"] >= 38)
                & (dataframe["rsi"] <= 68)
                # Momentum confirmation
                & (dataframe["macd"] > dataframe["macdsignal"])
                # Basic market validity
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit only when trend damage is clear (avoid tiny-pullback exits)."""

        if metadata.get("pair") != "BTC/USDT":
            dataframe["exit_long"] = 0
            return dataframe

        dataframe.loc[
            (
                (
                    # Stronger damage signal than close < ema20
                    ((dataframe["close"] < dataframe["ema50"]) & (dataframe["rsi"] < 42))
                    # Clear medium-term weakness
                    | (dataframe["close"] < dataframe["ema100"])
                    # Bearish alignment + momentum confirmation
                    | (
                        (dataframe["ema20"] < dataframe["ema50"])
                        & (dataframe["macd"] < dataframe["macdsignal"])
                    )
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
