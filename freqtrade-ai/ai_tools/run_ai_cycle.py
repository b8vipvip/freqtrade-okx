# -*- coding: utf-8 -*-
"""Run end-to-end AI optimization cycle for Freqtrade strategy."""

from __future__ import annotations

from analyze_backtest import analyze_backtest
from generate_strategy import generate_strategy


def run_cycle() -> None:
    print("[1/2] 开始分析回测结果...")
    analysis = analyze_backtest()
    print(f"[1/2] 分析完成，内容长度: {len(analysis)}")

    print("[2/2] 开始生成新策略...")
    code = generate_strategy()
    print(f"[2/2] 生成完成，代码长度: {len(code)}")
    print("AI 循环完成。")


if __name__ == "__main__":
    try:
        run_cycle()
    except Exception as exc:  # noqa: BLE001
        print(f"AI 循环失败: {exc}")
        raise
