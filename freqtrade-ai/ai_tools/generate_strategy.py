# -*- coding: utf-8 -*-
"""Generate Freqtrade strategy code from AI analysis."""

from __future__ import annotations

import argparse

from openai import OpenAI

from utils import (
    ensure_file_exists,
    extract_python_code,
    load_settings,
    read_text_file,
    write_text_file,
)


def _validate_strategy(code: str) -> None:
    required = [
        "class AI_Generated_Strategy",
        "IStrategy",
        "populate_indicators",
        "populate_entry_trend",
        "populate_exit_trend",
    ]
    missing = [item for item in required if item not in code]
    if missing:
        raise ValueError(f"生成策略缺少必要内容: {missing}")


def generate_strategy(analysis_path: str | None = None, output_path: str | None = None) -> str:
    settings = load_settings()
    src = analysis_path or settings["AI_ANALYSIS_FILE"]
    dst = output_path or settings["AI_STRATEGY_FILE"]

    ensure_file_exists(src, "AI analysis file not found")
    api_key = settings["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")

    analysis = read_text_file(src)
    prompt = (
        "请基于以下回测分析，生成完整的 Freqtrade Python 策略代码。要求："
        "1. 类名必须是 AI_Generated_Strategy；2. 必须继承 IStrategy；3. 默认 OKX 现货；"
        "4. timeframe='5m'；5. 包含 minimal_roi；6. 包含 stoploss；7. 包含 startup_candle_count；"
        "8. 包含 populate_indicators；9. 包含 populate_entry_trend；10. 包含 populate_exit_trend；"
        "11-14 禁止马丁格尔/无限补仓/高杠杆/自动加仓；15 禁止未来函数；"
        "16 禁止在实时交易函数里调用 OpenAI API；17 只输出完整 Python 代码，不要解释。"
        "\n\n回测分析:\n"
        f"{analysis}"
    )

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=settings["OPENAI_MODEL"],
        input=prompt,
        temperature=0.3,
    )
    code = extract_python_code(resp.output_text)
    _validate_strategy(code)
    write_text_file(dst, code)
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", dest="analysis_path", default=None)
    parser.add_argument("--output", dest="output_path", default=None)
    args = parser.parse_args()

    try:
        code = generate_strategy(args.analysis_path, args.output_path)
        print(f"策略生成成功，输出长度: {len(code)} 字符")
    except Exception as exc:  # noqa: BLE001
        print(f"策略生成失败: {exc}")
        raise


if __name__ == "__main__":
    main()
