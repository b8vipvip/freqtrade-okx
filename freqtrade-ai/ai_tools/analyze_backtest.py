# -*- coding: utf-8 -*-
"""Analyze Freqtrade backtest result with OpenAI."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils import load_settings, write_text_file

KEYWORDS = ("profit", "drawdown", "trade", "win", "loss", "pair", "exit_reason")
CORE_FIELDS = (
    "total_trades",
    "profit_total",
    "profit_total_abs",
    "final_balance",
    "starting_balance",
    "wins",
    "losses",
    "draws",
    "winrate",
    "profit_factor",
    "expectancy",
    "max_drawdown",
    "max_drawdown_abs",
    "max_drawdown_account",
    "drawdown_start",
    "drawdown_end",
    "results_per_pair",
    "exit_reason_summary",
    "enter_tag_results",
    "left_open_trades",
)


def _load_backtest_data(preferred_path: str) -> Any:
    preferred = Path(preferred_path)
    if preferred.exists() and preferred.is_file():
        return json.loads(preferred.read_text(encoding="utf-8"))

    results_dir = Path("user_data/backtest_results")
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(
            f"Backtest result file not found: {preferred}. Also missing directory: {results_dir}"
        )

    zip_files = sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(
            f"Backtest result file not found: {preferred}. No backtest-result-*.zip found in {results_dir}"
        )

    latest_zip = zip_files[0]
    with zipfile.ZipFile(latest_zip) as zf:
        json_members = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".json") and not name.lower().endswith(".meta.json")
        ]
        if not json_members:
            raise FileNotFoundError(f"No valid .json (excluding .meta.json) found in {latest_zip}")

        with zf.open(json_members[0]) as fp:
            return json.load(fp)


def _recursive_keyword_matches(data: Any, path: str = "") -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = str(key).lower()
            next_path = f"{path}.{key}" if path else str(key)
            if any(keyword in key_lower for keyword in KEYWORDS):
                if isinstance(value, (str, int, float, bool)) or value is None:
                    matches.append({"path": next_path, "value": value})
                elif isinstance(value, list):
                    matches.append({"path": next_path, "value": value[:20]})
                elif isinstance(value, dict):
                    matches.append({"path": next_path, "value": {k: value[k] for k in list(value)[:20]}})
            matches.extend(_recursive_keyword_matches(value, next_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            next_path = f"{path}[{idx}]"
            matches.extend(_recursive_keyword_matches(item, next_path))
    return matches


def _extract_strategy_summary(strategy_name: str, strategy_data: Any, strategy_comparison: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"strategy_name": strategy_name}
    if isinstance(strategy_data, dict):
        for field in CORE_FIELDS:
            if field in strategy_data:
                summary[field] = strategy_data[field]

        trades = strategy_data.get("trades")
        if isinstance(trades, list):
            summary["trades_sample_head"] = trades[:20]
            summary["trades_sample_tail"] = trades[-20:] if len(trades) > 20 else trades

        summary["keyword_matches"] = _recursive_keyword_matches(strategy_data)
    else:
        summary["note"] = "strategy data is not a dict"

    summary["strategy_comparison"] = strategy_comparison
    return summary


def _build_backtest_summary(raw_data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_type": "freqtrade_backtest",
        "top_level_keys": list(raw_data.keys()) if isinstance(raw_data, dict) else "non-dict",
    }
    if not isinstance(raw_data, dict):
        summary["raw_keyword_matches"] = _recursive_keyword_matches(raw_data)
        return summary

    strategy_block = raw_data.get("strategy")
    strategy_comparison = raw_data.get("strategy_comparison")
    summary["strategy_comparison"] = strategy_comparison

    strategy_summaries: dict[str, Any] = {}
    if isinstance(strategy_block, dict):
        for strategy_name, strategy_data in strategy_block.items():
            strategy_summaries[strategy_name] = _extract_strategy_summary(
                strategy_name=strategy_name,
                strategy_data=strategy_data,
                strategy_comparison=strategy_comparison,
            )
    else:
        strategy_summaries["__fallback__"] = {
            "note": "top-level `strategy` field missing or not a dict",
            "keyword_matches": _recursive_keyword_matches(raw_data),
        }

    summary["strategies"] = strategy_summaries
    summary["raw_keyword_matches"] = _recursive_keyword_matches(raw_data)
    return summary


def analyze_backtest(input_path: str | None = None, output_path: str | None = None) -> str:
    settings = load_settings()
    backtest_file = input_path or settings["BACKTEST_FILE"]
    analysis_file = output_path or settings["AI_ANALYSIS_FILE"]

    api_key = settings["OPENAI_API_KEY"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未配置，请先在 .env 中设置。")

    raw_data = _load_backtest_data(backtest_file)
    summary_data = _build_backtest_summary(raw_data)
    payload = json.dumps(summary_data, ensure_ascii=False)

    prompt = (
        "你是资深量化交易策略审查员。以下是结构化的 Freqtrade 回测结果 summary dict。"
        "这是主要分析对象，不要把 config 快照或无关字段当成主要分析对象。"
        "必须优先使用 summary dict 中的明确数值进行结论。"
        "如果 summary 中已有 profit_total 或 profit_total_abs，禁止说‘无法判断’、‘按推断’或自行推断总收益率。"
        "字段已存在时，必须直接引用该字段。\n"
        "请输出中文结构化报告，并必须包含："
        "1)总收益率 2)绝对收益 3)初始资金 4)最终资金 5)总交易数 6)胜率 7)Profit factor 8)最大回撤 "
        "9)每个交易对表现 10)exit_signal/roi/force_exit 等退出原因分析 11)下一版策略优化建议。"
        "\n严格禁止建议或使用：马丁格尔、无限补仓、高杠杆、合约高风险策略。"
        "\n\nFreqtrade 回测 summary:\n"
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
