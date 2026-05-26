# -*- coding: utf-8 -*-
"""Analyze Freqtrade backtest result with OpenAI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils import ensure_file_exists, load_settings, write_text_file


def _compact_json_payload(data: Any, max_chars: int = 25000) -> str:
    compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= max_chars:
        return compact

    summary = {
        "note": "Input too large; truncated for token safety.",
        "keys": list(data.keys()) if isinstance(data, dict) else "non-dict",
        "head": compact[: max_chars - 300],
        "tail": compact[-200:],
    }
    return json.dumps(summary, ensure_ascii=False)


def analyze_backtest(input_path: str | None = None, output_path: str | None = None) -> str:
    settings = load_settings()
    backtest_file = input_path or settings["BACKTEST_FILE"]
    analysis_file = output_path or settings["AI_ANALYSIS_FILE"]

    ensure_file_exists(backtest_file, "Backtest result file not found")
    api_key = settings["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")

    raw_data = json.loads(Path(backtest_file).read_text(encoding="utf-8"))
    payload = _compact_json_payload(raw_data)

    prompt = (
        "你是资深量化交易策略审查员。请基于以下 Freqtrade 回测 JSON，输出中文分析报告，必须覆盖："
        "1)总收益率 2)最大回撤 3)胜率 4)盈亏比 5)交易次数是否过多 6)止损是否过紧 "
        "7)止盈是否过早 8)是否追高 9)哪些交易对表现好/差 10)下一版策略优化建议。"
        "\n严格禁止建议或使用：马丁格尔、无限补仓、高杠杆、合约高风险策略。"
        "\n请使用结构化小标题输出。\n\n回测数据:\n"
        f"{payload}"
    )

    client = OpenAI(api_key=settings["OPENAI_API_KEY"], base_url=settings.get("OPENAI_BASE_URL") or None)
    response = client.chat.completions.create(
        model=settings["OPENAI_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    result = response.choices[0].message.content
    analysis = (result or "").strip() + "\n"
    write_text_file(analysis_file, analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", default=None)
    parser.add_argument("--output", dest="output_path", default=None)
    args = parser.parse_args()

    try:
        result = analyze_backtest(args.input_path, args.output_path)
        print(f"分析完成，输出文件已生成。长度: {len(result)} 字符")
    except Exception as exc:  # noqa: BLE001
        print(f"分析失败: {exc}")
        raise


if __name__ == "__main__":
    main()
