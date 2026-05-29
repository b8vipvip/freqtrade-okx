# -*- coding: utf-8 -*-
"""自动化 Freqtrade 策略优化脚本（中文交互向导 + 防过拟合 + 多区间验证）。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT_DIR / "user_data" / "backtest_results" / "ai_optimization_runs"
GENERATED_DIR = ROOT_DIR / "user_data" / "strategies" / "generated"
STRATEGY_DIR = ROOT_DIR / "user_data" / "strategies"
MEMORY_FILE = ROOT_DIR / "user_data" / "ai_memory" / "strategy_memory.json"
BLACKLIST_FILE = ROOT_DIR / "user_data" / "ai_memory" / "strategy_blacklist.json"
LESSONS_FILE = ROOT_DIR / "user_data" / "ai_memory" / "strategy_lessons.json"
BEST_STRATEGY_FILE = ROOT_DIR / "user_data" / "ai_memory" / "best_strategy.json"
RESET_HISTORY_FILE = ROOT_DIR / "user_data" / "ai_memory" / "reset_history.json"
NEAREST_CANDIDATE_FILE = ROOT_DIR / "user_data" / "ai_memory" / "nearest_candidate.json"
LAST_RUN_SUMMARY_FILE = ROOT_DIR / "user_data" / "ai_memory" / "last_run_summary.json"
ITERATION_STATS_FILE_NAME = "iteration_stats.json"
MEMORY_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_memory.example.json"
BLACKLIST_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_blacklist.example.json"
LESSONS_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_lessons.example.json"
MODEL_CONFIG_FILE = ROOT_DIR / "ai_tools" / "model_config.json"
MODEL_CONFIG_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "model_config.example.json"
TIMERANGE_RE = re.compile(r"^\d{8}-\d{8}$")


ROLE_DISPLAY_NAMES = {
    "strategy_advisor": "策略顾问",
    "code_generator": "代码生成",
}


class Tee:
    """Write terminal output to multiple streams, similar to the shell tee command."""

    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, data: str) -> int:
        for file in self.files:
            file.write(data)
        return len(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()

    def isatty(self) -> bool:
        return any(getattr(file, "isatty", lambda: False)() for file in self.files)


@dataclass
class LogFileContext:
    run_log_path: Path
    global_log_path: Path
    latest_log_path: Path
    files: list[Any] = field(default_factory=list)
    original_stdout: Any = None
    original_stderr: Any = None


def _resolve_log_dir(log_dir: str) -> Path:
    path = Path(log_dir).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _update_latest_log(latest_log_path: Path, global_log_path: Path) -> None:
    latest_log_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_log_path.exists() or latest_log_path.is_symlink():
        latest_log_path.unlink()
    try:
        latest_log_path.symlink_to(global_log_path)
    except OSError:
        shutil.copy2(global_log_path, latest_log_path)


def setup_terminal_logging(run_dir: Path, args: argparse.Namespace) -> LogFileContext | None:
    if getattr(args, "no_log_file", False):
        return None

    run_id = run_dir.name
    run_log_path = run_dir / "run.log"
    global_log_dir = _resolve_log_dir(str(getattr(args, "log_dir", "user_data/logs")))
    global_log_path = global_log_dir / f"auto_optimize_{run_id}.log"
    latest_log_path = global_log_dir / "latest.log"

    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    global_log_path.parent.mkdir(parents=True, exist_ok=True)
    run_log_file = run_log_path.open("a", encoding="utf-8", buffering=1)
    global_log_file = global_log_path.open("a", encoding="utf-8", buffering=1)

    ctx = LogFileContext(
        run_log_path=run_log_path,
        global_log_path=global_log_path,
        latest_log_path=latest_log_path,
        files=[run_log_file, global_log_file],
        original_stdout=sys.stdout,
        original_stderr=sys.stderr,
    )
    sys.stdout = Tee(sys.__stdout__, run_log_file, global_log_file)
    sys.stderr = Tee(sys.__stderr__, run_log_file, global_log_file)
    _update_latest_log(latest_log_path, global_log_path)
    return ctx


def restore_terminal_logging(log_ctx: LogFileContext | None) -> None:
    if log_ctx is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = log_ctx.original_stdout or sys.__stdout__
    sys.stderr = log_ctx.original_stderr or sys.__stderr__
    for file in log_ctx.files:
        file.close()


def print_log_start_banner(log_ctx: LogFileContext | None) -> None:
    if log_ctx is None:
        return
    print("\n========== 日志文件 ==========")
    print("本次完整运行日志：")
    print(log_ctx.run_log_path)
    print("\n全局日志副本：")
    print(log_ctx.global_log_path)
    print("\n查看实时日志：")
    print(f"tail -f {log_ctx.run_log_path}")


def print_log_saved_summary(args: argparse.Namespace) -> None:
    log_ctx = getattr(args, "_log_context", None)
    if log_ctx is None:
        return
    print("\n========== 日志保存 ==========")
    print("完整运行日志：")
    print(log_ctx.run_log_path)
    print("\n全局日志：")
    print(log_ctx.global_log_path)
    print("\n最近一次日志：")
    print(log_ctx.latest_log_path)


def _format_elapsed_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


@dataclass
class AIRoleRuntime:
    role: str
    client: OpenAI
    model_pool: list[str]
    timeout_sec: int
    switch_on_error: bool = True
    max_attempts_per_call: int = 5
    attempts: list[dict[str, Any]] = field(default_factory=list)
    used_model: str = ""

    @property
    def display_name(self) -> str:
        return ROLE_DISPLAY_NAMES.get(self.role, self.role)

    def begin_call(self) -> None:
        self.attempts = []
        self.used_model = ""

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "model_pool": list(self.model_pool),
            "used_model": self.used_model,
            "attempts": list(self.attempts),
        }


@dataclass
class PeriodDef:
    name: str
    timerange: str
    weight: float
    kind: str


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "strategy_advisor": {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url_env": "CLAUDE_BASE_URL",
        "api_key_env": "CLAUDE_API_KEY",
        "model_env": "CLAUDE_MODEL",
        "model_pool_env": "CLAUDE_MODEL_POOL",
        "default_model": "claude-opus-4-7",
    },
    "code_generator": {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model_pool_env": "OPENAI_MODEL_POOL",
        "default_model": "gpt-5.5",
    },
    "code_repair": {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model_pool_env": "OPENAI_MODEL_POOL",
        "default_model": "gpt-5.5",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_runtime_json_file(path: Path, example_path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if example_path.exists():
        shutil.copy2(example_path, path)
    else:
        write_json(path, {"items": []})


def _read_json_list_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("items", [])
        return items if isinstance(items, list) else []
    return raw if isinstance(raw, list) else []


def _write_json_list_file(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_yes_no(value: str) -> bool | None:
    s = value.strip().lower()
    if s in {"y", "yes", "是", "true", "1"}:
        return True
    if s in {"n", "no", "否", "false", "0"}:
        return False
    return None


def ask_text(prompt: str, default: str) -> str:
    v = input(f"{prompt}（默认：{default}）：").strip()
    return v if v else default


def ask_bool(prompt: str, default: bool) -> bool:
    d = "是" if default else "否"
    while True:
        v = input(f"{prompt}（默认：{d}，输入 y/n）：").strip()
        if not v:
            return default
        p = parse_yes_no(v)
        if p is not None:
            return p
        print("输入无效，请输入 y 或 n。")


def ask_int(prompt: str, default: int) -> int:
    while True:
        v = input(f"{prompt}（默认：{default}）：").strip()
        if not v:
            return default
        if v.isdigit():
            return int(v)
        print("请输入整数。")


def ask_float(prompt: str, default: float) -> float:
    while True:
        v = input(f"{prompt}（默认：{default}）：").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            print("请输入数字（可带小数）。")


def ask_timerange(prompt: str, default: str) -> str:
    while True:
        v = input(f"{prompt}（默认：{default}）：").strip()
        if not v:
            return default
        if TIMERANGE_RE.match(v):
            return v
        print("时间区间格式错误，应为 YYYYMMDD-YYYYMMDD。")


def ensure_goal_file(goal_path: Path) -> None:
    if goal_path.exists():
        return
    example = ROOT_DIR / "ai_tools" / "optimization_goal.example.json"
    if not example.exists():
        raise FileNotFoundError(f"未找到目标文件，且示例文件不存在：{example}")
    goal_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(example, goal_path)
    print(f"未找到目标文件，已从示例生成：{goal_path}")


def ensure_model_config_files() -> dict[str, Any]:
    if not MODEL_CONFIG_EXAMPLE_FILE.exists():
        write_json(MODEL_CONFIG_EXAMPLE_FILE, DEFAULT_MODEL_CONFIG)
    if not MODEL_CONFIG_FILE.exists():
        write_json(MODEL_CONFIG_FILE, DEFAULT_MODEL_CONFIG)
    return read_json(MODEL_CONFIG_FILE)


def run_wizard(goal: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    print("\n===== 自动优化中文设置向导 =====")
    enter = input("是否进入交互式设置？（默认：是，回车=是，输入 n=跳过）：").strip()
    parsed = parse_yes_no(enter) if enter else True
    if parsed is False:
        return goal

    runtime = json.loads(json.dumps(goal))
    runtime.setdefault("language", "zh-CN")
    runtime["strategy_family"] = ask_text("策略家族名", str(runtime.get("strategy_family", args.base_strategy)))
    runtime["config"] = ask_text("配置文件路径", str(runtime.get("config", args.config)))
    cfg_path = ROOT_DIR / runtime["config"]
    if not cfg_path.exists():
        cont = ask_bool(f"配置文件不存在：{runtime['config']}，是否继续", False)
        if not cont:
            raise RuntimeError("用户取消：配置文件不存在")

    runtime["timeframe"] = ask_text("主周期 timeframe", str(runtime.get("timeframe", args.timeframe)))

    runtime.setdefault("train_period", {})
    default_train_timerange = (
        runtime.get("train_period", {}).get("timerange")
        or getattr(args, "timerange", None)
        or "20260501-20260525"
    )
    runtime["train_period"]["timerange"] = ask_timerange("训练区间", str(default_train_timerange))
    runtime["train_period"]["name"] = runtime["train_period"].get("name", "train")
    runtime["train_period"]["weight"] = float(runtime["train_period"].get("weight", 1.0))

    vals = runtime.get("validation_periods", [])
    print("当前验证区间列表：")
    for idx, item in enumerate(vals, start=1):
        print(f"  {idx}. {item.get('name', 'valid')} -> {item.get('timerange', '')}")
    use_default = input("验证区间：回车=使用默认；输入 n=手动输入：").strip().lower()
    if use_default == "n":
        new_vals: list[dict[str, Any]] = []
        print("请输入验证区间（每行一个 YYYYMMDD-YYYYMMDD，空行结束）：")
        cnt = 1
        while True:
            line = input(f"验证区间{cnt}：").strip()
            if not line:
                break
            if not TIMERANGE_RE.match(line):
                print("格式错误，跳过该行。")
                continue
            new_vals.append({"name": f"valid_{cnt:02d}", "timerange": line, "weight": 1.0})
            cnt += 1
        if new_vals:
            runtime["validation_periods"] = new_vals

    runtime.setdefault("data_download", {})
    runtime["data_download"]["download_timerange"] = ask_timerange(
        "数据下载区间", str(runtime["data_download"].get("download_timerange", runtime["train_period"]["timerange"]))
    )
    runtime["data_download"]["auto_download"] = ask_bool("是否自动下载数据", bool(runtime["data_download"].get("auto_download", True)))
    runtime["runtime_force_download"] = ask_bool("是否强制重新下载历史数据", bool(runtime.get("runtime_force_download", False)))

    default_iter = args.iterations if args.iterations is not None else int(runtime.get("max_iterations", 5))
    runtime["max_iterations"] = ask_int("最大迭代轮数", int(default_iter))

    auto_default = bool(args.auto_approve)
    runtime["runtime_auto_approve"] = ask_bool("是否全自动不中途确认", auto_default)

    runtime.setdefault("target", {})
    runtime["target"]["min_profit_total_pct"] = ask_float("目标最低收益率", float(runtime["target"].get("min_profit_total_pct", 0)))
    runtime["target"]["max_drawdown_pct"] = ask_float("最大允许回撤(%)", float(runtime["target"].get("max_drawdown_pct", 3)))
    runtime["target"]["min_profit_factor"] = ask_float("最低 Profit factor", float(runtime["target"].get("min_profit_factor", 1.0)))
    runtime["target"]["min_trades"] = ask_int("目标最小交易数", int(runtime["target"].get("min_trades", 25)))
    runtime["target"]["max_trades"] = ask_int("目标最大交易数", int(runtime["target"].get("max_trades", 80)))

    runtime.setdefault("overfit_guard", {})
    runtime["overfit_guard"]["enabled"] = ask_bool("是否启用防过拟合", bool(runtime["overfit_guard"].get("enabled", True)))

    print("提示：当前项目历史回测中 exit_signal 曾造成大量亏损，建议保持关闭。")
    runtime["target"]["prefer_exit_signal"] = ask_bool("是否允许 exit_signal", bool(runtime["target"].get("prefer_exit_signal", False)))
    runtime["runtime_reset_best"] = ask_bool("是否初始化历史最佳策略", bool(runtime.get("runtime_reset_best", False)))

    runtime.setdefault("baseline", {})
    b = runtime["baseline"]
    print("当前基准 baseline：")
    print(f"  profit_total_abs={b.get('profit_total_abs', 0)}")
    print(f"  profit_total_pct={b.get('profit_total_pct', 0)}")
    print(f"  profit_factor={b.get('profit_factor', 0)}")
    print(f"  max_drawdown_pct={b.get('max_drawdown_pct', 0)}")
    print(f"  total_trades={b.get('total_trades', 0)}")
    if ask_bool("是否手动修改 baseline", False):
        b["profit_total_abs"] = ask_float("baseline.profit_total_abs", float(b.get("profit_total_abs", 0)))
        b["profit_total_pct"] = ask_float("baseline.profit_total_pct", float(b.get("profit_total_pct", 0)))
        b["profit_factor"] = ask_float("baseline.profit_factor", float(b.get("profit_factor", 0)))
        b["max_drawdown_pct"] = ask_float("baseline.max_drawdown_pct", float(b.get("max_drawdown_pct", 0)))
        b["total_trades"] = ask_int("baseline.total_trades", int(b.get("total_trades", 0)))

    print("\n========== 本次自动优化设置 ==========")
    print(f"策略家族：{runtime.get('strategy_family')}")
    print(f"配置文件：{runtime.get('config')}")
    print(f"训练区间：{runtime.get('train_period', {}).get('timerange')}")
    print("验证区间：")
    for item in runtime.get("validation_periods", []):
        print(f"  - {item.get('name')} : {item.get('timerange')}")
    print(f"数据下载区间：{runtime.get('data_download', {}).get('download_timerange')}")
    print(f"自动下载数据：{runtime.get('data_download', {}).get('auto_download')}")
    print(f"迭代轮数：{runtime.get('max_iterations')}")
    print(f"全自动模式：{runtime.get('runtime_auto_approve')}")
    print(f"目标收益率：{runtime.get('target', {}).get('min_profit_total_pct')}")
    print(f"最大回撤：{runtime.get('target', {}).get('max_drawdown_pct')}")
    print(f"最低 Profit factor：{runtime.get('target', {}).get('min_profit_factor')}")
    print(f"目标交易数：{runtime.get('target', {}).get('min_trades')}~{runtime.get('target', {}).get('max_trades')}")
    print(f"防过拟合：{runtime.get('overfit_guard', {}).get('enabled')}")
    print(f"是否允许 exit_signal：{runtime.get('target', {}).get('prefer_exit_signal')}")
    print(f"当前基准：{runtime.get('baseline')}")

    start = input("是否开始自动优化？（回车/y=开始，n=取消）：").strip()
    if start and parse_yes_no(start) is False:
        raise RuntimeError("用户取消执行")

    return runtime


# 以下保留原有核心函数（精简）
def extract_python_code(content: str) -> str:
    m = re.search(r"```python\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else content).strip() + "\n"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("strategy_spec 为空")

    # 1) 优先处理 markdown fenced code block（```json ... ``` / ``` ... ```）
    fence_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in fence_patterns:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            payload = m.group(1).strip()
            obj = json.loads(payload)
            if isinstance(obj, dict):
                return obj
            raise ValueError("strategy_spec 顶层必须是 JSON object")

    # 2) 纯 JSON（直接是对象文本）
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3) 文本中混有前后缀说明，尝试提取首个 {...} 对象并解码
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("未能从 strategy_spec 文本中提取 JSON object")


def _list_backtest_zips(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime)


def _select_backtest_zip(results_dir: Path, before_set: set[Path], cmd_end_ts: float) -> Path:
    all_zips = _list_backtest_zips(results_dir)
    new_zips = [z for z in all_zips if z not in before_set]
    candidates = new_zips if new_zips else [z for z in all_zips if z.stat().st_mtime >= cmd_end_ts]
    if not candidates:
        raise FileNotFoundError("未找到本轮回测新增的 backtest-result-*.zip")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_backtest_from_zip(zip_path: Path, strategy_class: str) -> dict[str, Any]:
    print(f"正在解析 zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n for n in zf.namelist()
            if n.endswith('.json')
            and not n.endswith('.meta.json')
            and not n.endswith('_config.json')
        ]
        primary = [n for n in names if Path(n).name.startswith("backtest-result-")]
        if not primary:
            raise RuntimeError(f"zip 内未找到 backtest-result-*.json: {zip_path}")
        json_name = sorted(primary)[-1]
        print(f"正在读取 json: {json_name}")
        with zf.open(json_name) as fp:
            data = json.load(fp)

    strategy_data = data.get("strategy")
    if not isinstance(strategy_data, dict):
        raise RuntimeError("回测结果缺少 strategy 字段")
    if strategy_class not in strategy_data:
        raise RuntimeError(f"未在回测结果中找到当前策略 {strategy_class}")
    result = strategy_data[strategy_class]
    print(f"找到策略: {strategy_class}")
    return result


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def run_single_backtest_metrics(config: str, class_name: str, timeframe: str, timerange: str) -> dict[str, Any]:
    cmd = [
        "docker", "compose", "run", "--rm", "freqtrade", "backtesting",
        "--config", config, "--strategy", class_name, "--timeframe", timeframe,
        "--timerange", timerange, "--export", "trades", "--cache", "none",
    ]
    results_dir = ROOT_DIR / "user_data" / "backtest_results"
    before_zips = set(_list_backtest_zips(results_dir))
    start_ts = time.time()
    cp = run_cmd(cmd, ROOT_DIR)
    if cp.returncode != 0:
        raise RuntimeError(f"回测失败：{class_name}\n{cp.stderr}")
    result_zip = _select_backtest_zip(results_dir, before_zips, start_ts)
    return _extract_metrics(parse_backtest_from_zip(result_zip, class_name))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _print_round_table(version: str, interval: str, metrics: dict[str, Any]) -> None:
    trades = _safe_int(metrics.get("total_trades"))
    profit_pct = _safe_float(metrics.get("profit_total_pct"))
    profit_abs = _safe_float(metrics.get("profit_total_abs"))
    winrate = _safe_float(metrics.get("winrate")) * 100.0
    pf = _safe_float(metrics.get("profit_factor"))
    max_dd = _safe_float(metrics.get("max_drawdown")) * 100.0
    roi_profit = metrics.get("roi_profit_abs")
    stop_loss_abs = metrics.get("stop_loss_profit_abs")
    trailing_abs = metrics.get("trailing_stop_loss_profit_abs")
    force_exit_abs = metrics.get("force_exit_profit_abs")
    roi_text = f"{_safe_float(roi_profit):.4f}" if roi_profit is not None else "无法解析 exit reason 明细"
    stop_text = f"{_safe_float(stop_loss_abs):.4f}" if stop_loss_abs is not None else "无法解析 exit reason 明细"
    trailing_text = f"{_safe_float(trailing_abs):.4f}" if trailing_abs is not None else "无法解析 exit reason 明细"
    force_exit_text = f"{_safe_float(force_exit_abs):.4f}" if force_exit_abs is not None else "无法解析 exit reason 明细"
    print("版本 | 区间 | 交易数 | 收益率 | 收益USDT | 胜率 | PF | 最大回撤 | ROI收益USDT | 固定止损USDT | 移动止盈/止损USDT | 强制退出USDT")
    print(
        f"{version} | {interval} | {trades} | {_format_pct(profit_pct)} | {profit_abs:.4f} | "
        f"{_format_pct(winrate)} | {pf:.4f} | {_format_pct(max_dd)} | {roi_text} | {stop_text} | "
        f"{trailing_text} | {force_exit_text}"
    )


def parse_exit_reason_details(result: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "parsed": True,
        "roi_count": 0,
        "roi_profit_abs": 0.0,
        "stop_loss_count": 0,
        "stop_loss_profit_abs": 0.0,
        "trailing_stop_loss_count": 0,
        "trailing_stop_loss_profit_abs": 0.0,
        "force_exit_count": 0,
        "force_exit_profit_abs": 0.0,
        "exit_signal_count": 0,
        "exit_signal_profit_abs": 0.0,
    }
    alias_to_bucket = {
        "roi": "roi",
        "stop_loss": "stop_loss",
        "stoploss": "stop_loss",
        "stop_loss_on_exchange": "stop_loss",
        "trailing_stop_loss": "trailing_stop_loss",
        "force_exit": "force_exit",
        "exit_signal": "exit_signal",
    }

    def _accumulate(reason: str, count: Any, profit_abs: Any) -> bool:
        bucket = alias_to_bucket.get(str(reason).strip().lower())
        if not bucket:
            return False
        details[f"{bucket}_count"] += _safe_int(count)
        details[f"{bucket}_profit_abs"] += _safe_float(profit_abs)
        return True

    ers = result.get("exit_reason_summary")
    if isinstance(ers, list):
        for row in ers:
            if not isinstance(row, dict):
                continue
            _accumulate(row.get("key"), row.get("trades"), row.get("profit_total_abs"))
        return details

    trades = result.get("trades")
    if isinstance(trades, list):
        parsed_any = False
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            if _accumulate(trade.get("exit_reason"), 1, trade.get("profit_abs")):
                parsed_any = True
        details["parsed"] = parsed_any
        return details

    details["parsed"] = False
    print(f"[debug] exit_reason_summary 类型: {type(ers).__name__}")
    print(f"[debug] exit_reason_summary 前3项: {ers[:3] if isinstance(ers, list) else ers}")
    print(f"[debug] result 可用字段列表: {sorted(result.keys())}")
    print(f"[debug] trades 数量: {len(trades) if isinstance(trades, list) else 0}")
    return details





def _max_drawdown_pct(metrics: dict[str, Any]) -> float:
    if metrics.get("max_drawdown_pct") is not None:
        return _safe_float(metrics.get("max_drawdown_pct"))
    return _safe_float(metrics.get("max_drawdown")) * 100.0


def _format_signed_abs(value: float) -> str:
    return f"{value:+.4f}"


def _print_starting_champion_summary(goal: dict[str, Any], champion: dict[str, Any]) -> None:
    source = str(champion.get("source") or "baseline")
    meta = champion.get("meta", {}) or {}
    if source == "reset_empty":
        print("历史 best 不存在，当前使用 baseline 作为 champion。")
        source_label = "baseline"
    elif source == "historical_best":
        source_label = "historical_best"
    elif source == "baseline":
        if not BEST_STRATEGY_FILE.exists():
            print("历史 best 不存在，当前使用 baseline 作为 champion。")
        else:
            print("历史 best 无效，当前使用 baseline 作为 champion。")
        source_label = "baseline"
    else:
        source_label = source or "无"

    tm = meta.get("train_metrics", {}) or {}
    av = dict(meta.get("avg_validation_metrics", {}) or {})
    validation_metrics = meta.get("validation_metrics", []) or []
    if validation_metrics:
        avg_profit, avg_pf, max_dd = _aggregate_validation_metrics(validation_metrics)
        av.setdefault("profit_total_pct", avg_profit)
        av.setdefault("profit_factor", avg_pf)
        av.setdefault("max_drawdown_pct", max_dd)

    advantages = meta.get("why_best") or ("无历史 best，作为基准对照。" if source_label == "baseline" else "综合指标为当前历史最优。")
    risks = meta.get("risk_summary") or meta.get("main_risks") or "仍需在更多验证区间复验稳健性。"
    print("\n========== 当前历史最佳策略简述 ==========")
    print(f"来源：{source_label}")
    print(f"策略名：{meta.get('strategy_class', '-')}")
    print(f"策略文件：{meta.get('strategy_file', '-')}")
    print(f"训练区间：{goal.get('train_period', {}).get('timerange', '-')}")
    print(f"交易数：{_safe_int(tm.get('total_trades'))}")
    print(f"收益USDT：{_safe_float(tm.get('profit_total_abs')):.4f}")
    print(f"收益率：{_safe_float(tm.get('profit_total_pct')):.2f}%")
    print(f"胜率：{_safe_float(tm.get('winrate')) * 100:.2f}%")
    print(f"Profit Factor：{_safe_float(tm.get('profit_factor')):.4f}")
    print(f"最大回撤：{_max_drawdown_pct(tm):.2f}%")
    print(f"验证区间平均收益率：{_safe_float(av.get('profit_total_pct')):.2f}%")
    print(f"验证区间平均 PF：{_safe_float(av.get('profit_factor')):.4f}")
    print(f"验证区间最大回撤：{_safe_float(av.get('max_drawdown_pct')):.2f}%")
    print(f"是否疑似过拟合：{'是' if meta.get('is_overfit') else '否'}")
    print(f"主要优势：{advantages}")
    print(f"主要风险：{risks}")


def _validation_profit_states(validation_metrics: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    profitable: list[str] = []
    losing: list[str] = []
    for item in validation_metrics:
        label = str(item.get("timerange") or item.get("period") or item.get("period_name") or "validation")
        profit = _safe_float((item.get("metrics", {}) or {}).get("profit_total_abs"))
        if profit > 0:
            profitable.append(label)
        elif profit < 0:
            losing.append(label)
    return profitable, losing


def _build_not_best_reason_detail(
    train_metrics: dict[str, Any],
    validation_metrics: list[dict[str, Any]],
    final_score: float,
    champion_meta: dict[str, Any] | None,
    target_cfg: dict[str, Any],
    invalid_reason: str | None,
) -> list[str]:
    details: list[str] = []
    champion_train = (champion_meta or {}).get("train_metrics", {}) or {}
    champion_name = (champion_meta or {}).get("strategy_class") or "historical_best"
    train_profit = _safe_float(train_metrics.get("profit_total_pct"))
    champ_profit = _safe_float(champion_train.get("profit_total_pct"))
    if champion_train and train_profit < champ_profit:
        details.append(f"训练区间收益低于 {champion_name}：{train_profit:.2f}% < {champ_profit:.2f}%")
    train_pf = _safe_float(train_metrics.get("profit_factor"))
    champ_pf = _safe_float(champion_train.get("profit_factor"))
    if champion_train and train_pf < champ_pf:
        details.append(f"训练区间 PF 低于 {champion_name}：{train_pf:.4f} < {champ_pf:.4f}")
    train_dd = _max_drawdown_pct(train_metrics)
    champ_dd = _max_drawdown_pct(champion_train)
    if champion_train and champ_dd > 0 and train_dd > champ_dd:
        details.append(f"训练区间回撤高于 {champion_name}：{train_dd:.2f}% > {champ_dd:.2f}%")

    profitable, losing = _validation_profit_states(validation_metrics)
    if profitable and losing:
        details.append(f"验证区间表现不稳定：{','.join(profitable)} 盈利，但 {','.join(losing)} 亏损")
    elif validation_metrics and not profitable and losing:
        details.append("验证区间全部亏损")
    worst_pf = min((_safe_float((item.get("metrics", {}) or {}).get("profit_factor")) for item in validation_metrics), default=None)
    if worst_pf is not None and worst_pf < _safe_float(target_cfg.get("min_profit_factor", 1.0)):
        details.append(f"最差验证 PF 仅 {worst_pf:.4f}")

    roi_profit_abs = _safe_float(train_metrics.get("roi_profit_abs"))
    stop_loss_profit_abs = _safe_float(train_metrics.get("stop_loss_profit_abs"))
    if roi_profit_abs > 0 and abs(stop_loss_profit_abs) > roi_profit_abs * 1.2:
        details.append(f"固定止损亏损吞噬 ROI 收益：训练 ROI {roi_profit_abs:+.4f}，固定止损 {stop_loss_profit_abs:.4f}")

    min_pf = _safe_float(target_cfg.get("min_profit_factor", 1.0))
    if train_pf < min_pf:
        details.append(f"训练区间 PF 低于目标：{train_pf:.4f} < {min_pf:.4f}")
    max_dd_target = _safe_float(target_cfg.get("max_drawdown_pct"))
    if max_dd_target > 0 and train_dd > max_dd_target:
        details.append(f"训练区间回撤超标：{train_dd:.2f}% > {max_dd_target:.2f}%")
    if final_score <= 0:
        details.append(f"final_score={final_score:.4f}，未通过评分门槛")
    elif invalid_reason:
        details.append(str(invalid_reason))
    return details or [str(invalid_reason or "综合评分未优于当前 best")]


def _build_common_failure_patterns(rows: list[dict[str, Any]], target_cfg: dict[str, Any]) -> list[str]:
    if not rows:
        return []
    total = len(rows)
    majority = total // 2 + 1
    min_trades = _safe_int(target_cfg.get("min_trades", 25))
    max_trades = _safe_int(target_cfg.get("max_trades", 80))
    min_pf = _safe_float(target_cfg.get("min_profit_factor", 1.0))
    max_dd = _safe_float(target_cfg.get("max_drawdown_pct", 3.0))
    patterns: list[str] = []

    if sum(_safe_int(r.get("total_trades")) > max_trades for r in rows) >= majority:
        patterns.append("交易数过高")
    if sum(0 < _safe_int(r.get("total_trades")) < min_trades for r in rows) >= majority:
        patterns.append("交易数偏低")

    validation_state_rows = [r for r in rows if r.get("validation_metrics")]
    if validation_state_rows:
        all_validation_losses = True
        any_mixed = False
        for row in validation_state_rows:
            profits = [_safe_float((item.get("metrics", {}) or {}).get("profit_total_abs")) for item in row.get("validation_metrics", [])]
            if any(p > 0 for p in profits):
                all_validation_losses = False
            if any(p > 0 for p in profits) and any(p < 0 for p in profits):
                any_mixed = True
        if all_validation_losses:
            patterns.append("验证区间全部亏损")
        elif any_mixed:
            patterns.append("验证区间表现不稳定")

        severe_month_drag = 0
        for row in validation_state_rows:
            profits = [_safe_float((item.get("metrics", {}) or {}).get("profit_total_abs")) for item in row.get("validation_metrics", [])]
            if len(profits) >= 2 and min(profits) < 0 and max(profits) > 0 and abs(min(profits)) > max(abs(p) for p in profits if p != min(profits)) * 1.2:
                severe_month_drag += 1
        if severe_month_drag > 0:
            patterns.append("最差验证月份拖累整体表现")

    stoploss_swallow = sum(
        _safe_float(r.get("roi_profit_abs")) > 0
        and abs(_safe_float(r.get("stop_loss_profit_abs"))) > _safe_float(r.get("roi_profit_abs")) * 1.2
        for r in rows
    )
    if stoploss_swallow >= majority or stoploss_swallow > 0:
        patterns.append("固定止损吞噬 ROI")
    if sum(_safe_float(r.get("profit_factor")) < min_pf for r in rows) >= majority:
        patterns.append("PF 低")
    if max_dd > 0 and sum(_safe_float(r.get("max_drawdown_pct")) > max_dd for r in rows) >= majority:
        patterns.append("回撤超标")

    if not patterns:
        reasons = [str(r.get("invalid_reason") or r.get("failure_reason") or "综合评分不达标") for r in rows]
        patterns = sorted({r for r in reasons if r})[:5]
    return patterns


def _build_why_nearest(row: dict[str, Any], target_cfg: dict[str, Any], champion_meta: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    trades = _safe_int(row.get("total_trades"))
    min_trades = _safe_int(target_cfg.get("min_trades", 25))
    max_trades = _safe_int(target_cfg.get("max_trades", 80))
    if min_trades <= trades <= max_trades:
        reasons.append("训练交易数在目标范围内")
    elif trades < min_trades:
        reasons.append("训练交易数略低于目标下限，仅作为候选参考")
    elif trades <= int(max_trades * 1.5):
        reasons.append("训练交易数略高于目标上限，仅作为候选参考")

    validation_metrics = row.get("validation_metrics", []) or []
    if any(_safe_float((item.get("metrics", {}) or {}).get("profit_total_abs")) > 0 and _safe_float((item.get("metrics", {}) or {}).get("profit_factor")) > 1 for item in validation_metrics):
        reasons.append("有一个验证区间盈利且 PF > 1")
    profitable, losing = _validation_profit_states(validation_metrics)
    champion_train = (champion_meta or {}).get("train_metrics", {}) or {}
    if champion_train and (
        _safe_float(row.get("train_profit_pct")) < _safe_float(champion_train.get("profit_total_pct"))
        or _safe_float(row.get("profit_factor")) < _safe_float(champion_train.get("profit_factor"))
        or losing
    ):
        reasons.append("但训练区间和其他验证区间不如 historical_best")
    elif losing:
        reasons.append("但仍存在亏损验证区间")
    reasons.append("仅作为后续优化参考，不覆盖正式 best")
    return reasons


def _build_nearest_advisor_notes(nearest_candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not nearest_candidate:
        return {}
    train = nearest_candidate.get("train_metrics", {}) or {}
    validations = nearest_candidate.get("validation_metrics", []) or []
    positives: list[str] = []
    problems: list[str] = []
    for item in validations:
        label = str(item.get("timerange") or item.get("label") or item.get("period") or "验证区间")
        profit_abs = _safe_float(item.get("profit_total_abs"))
        pf = _safe_float(item.get("profit_factor"))
        dd = _safe_float(item.get("max_drawdown_pct"))
        trades = _safe_int(item.get("total_trades"))
        if profit_abs > 0:
            positives.append(f"{label} 验证区间盈利 {_format_signed_abs(profit_abs)} USDT")
            positives.append(f"{label} PF {pf:.4f}")
            positives.append(f"{label} 回撤仅 {dd:.2f}%")
            positives.append(f"{label} 交易数 {trades}，可作为目标区间参考")
        elif profit_abs < 0:
            problems.append(f"{label} 亏损 {profit_abs:.4f} USDT")
    train_profit_abs = _safe_float(train.get("profit_total_abs"))
    if train_profit_abs < 0:
        problems.insert(0, f"训练区间亏损 {train_profit_abs:.4f} USDT")
    if _safe_float(train.get("profit_factor")) > 0:
        problems.append("训练 PF 低于 historical_best 或目标时，不得覆盖正式 best")
    if _safe_float(train.get("roi_profit_abs")) > 0 and abs(_safe_float(train.get("stop_loss_profit_abs"))) > _safe_float(train.get("roi_profit_abs")) * 1.2:
        problems.append("固定止损吞噬 ROI")
    return {
        "本轮_nearest_可取之处": positives,
        "本轮_nearest_问题": problems,
        "下一轮建议": [
            "不要扩大交易数",
            "不要扩大 stoploss",
            "保持 45~65 笔交易",
            "分析 nearest_candidate 在盈利验证区间有效、但在亏损验证区间失效的原因",
            "优先做 pair_specific_filter 或 tag_specific_filter",
            "减少导致亏损验证区间固定止损的入场",
        ],
    }

def _memory_presence(path: Path) -> str:
    return "存在" if path.exists() else "不存在"


def _load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return None


def _is_reset_best(data: dict[str, Any] | None) -> bool:
    return bool(data and data.get("source") == "reset")


def _strategy_file_valid(path_raw: str | None) -> str:
    if not path_raw:
        return "不存在"
    sf = Path(path_raw)
    if not sf.is_absolute():
        sf = ROOT_DIR / sf
    return "有效" if sf.exists() and sf.is_file() else "无效"


def _normalize_validation_metric(item: dict[str, Any], fallback_label: str = "validation") -> dict[str, Any]:
    m = item.get("metrics", item) if isinstance(item, dict) else {}
    return {
        "label": item.get("period") or item.get("period_name") or fallback_label,
        "timerange": item.get("timerange", ""),
        "total_trades": _safe_int(m.get("total_trades")),
        "profit_total_abs": _safe_float(m.get("profit_total_abs")),
        "profit_total_pct": _safe_float(m.get("profit_total_pct")),
        "profit_factor": _safe_float(m.get("profit_factor")),
        "max_drawdown_pct": _safe_float(m.get("max_drawdown_pct")),
        "winrate": _safe_float(m.get("winrate")),
        "roi_profit_abs": _safe_float(m.get("roi_profit_abs")),
        "stop_loss_profit_abs": _safe_float(m.get("stop_loss_profit_abs")),
        "trailing_stop_loss_profit_abs": _safe_float(m.get("trailing_stop_loss_profit_abs")),
        "force_exit_profit_abs": _safe_float(m.get("force_exit_profit_abs")),
    }

def _build_baseline_best(goal: dict[str, Any]) -> dict[str, Any]:
    baseline = goal.get("baseline", {}) or {}
    return {
        "strategy_class": "baseline",
        "strategy_file": "baseline",
        "source_run_id": "baseline",
        "version": "baseline",
        "train_metrics": baseline,
        "validation_metrics": [],
        "avg_validation_metrics": {},
        "score_breakdown": {},
        "final_score": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "why_best": "无历史 best，使用 baseline。",
    }


def _load_champion(runtime_goal: dict[str, Any]) -> dict[str, Any]:
    if BEST_STRATEGY_FILE.exists():
        try:
            data = read_json(BEST_STRATEGY_FILE)
        except json.JSONDecodeError:
            print("警告：best_strategy.json 不是合法 JSON，已回退到 baseline champion。")
            return {"meta": _build_baseline_best(runtime_goal), "code": "", "source": "baseline"}
        if data.get("source") == "reset":
            print("历史最佳策略处于 reset 状态，已回退到 baseline champion。")
            return {"meta": _build_baseline_best(runtime_goal), "code": "", "source": "reset_empty"}

        strategy_file_raw = str(data.get("strategy_file", "") or "").strip()
        if not strategy_file_raw:
            return {"meta": _build_baseline_best(runtime_goal), "code": "", "source": "baseline"}

        sf = Path(strategy_file_raw)
        if not sf.is_absolute():
            sf = ROOT_DIR / sf
        code = ""
        if sf.exists() and sf.is_file():
            code = sf.read_text(encoding="utf-8")
        else:
            print("历史最佳策略文件无效或不是文件，已仅加载指标，不加载代码：")
            print(f"strategy_file={strategy_file_raw}")
        return {"meta": data, "code": code, "source": "historical_best"}
    return {"meta": _build_baseline_best(runtime_goal), "code": "", "source": "baseline"}


def _append_reset_history(reason: str) -> None:
    items: list[dict[str, Any]] = []
    if RESET_HISTORY_FILE.exists():
        try:
            raw = json.loads(RESET_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                items = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
                items = [x for x in raw.get("items", []) if isinstance(x, dict)]
        except json.JSONDecodeError:
            items = []
    items.append({"source": "reset", "reset_at": datetime.utcnow().isoformat(), "reason": reason})
    RESET_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESET_HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_champion_source(champion: dict[str, Any]) -> None:
    source = champion.get("source", "baseline")
    sf = str((champion.get("meta", {}) or {}).get("strategy_file", "") or "").strip() or "无"
    print(f"当前 champion 来源：{source}")
    print(f"当前 champion 策略文件：{sf}")


def print_current_best_summary(goal: dict[str, Any], memory: dict[str, Any] | None, current_best: dict[str, Any] | None) -> None:
    source = "baseline"
    data = current_best or memory or _build_baseline_best(goal)
    if current_best:
        source = "本轮新策略"
    elif memory:
        source = "历史 best"
    tm = data.get("train_metrics", {}) or {}
    av = data.get("avg_validation_metrics", {}) or {}
    validation_metrics = data.get("validation_metrics", []) or []
    if validation_metrics:
        avg_profit, avg_pf, max_dd = _aggregate_validation_metrics(validation_metrics)
        if "profit_total_pct" not in av:
            av["profit_total_pct"] = avg_profit
        if "profit_factor" not in av:
            av["profit_factor"] = avg_pf
        if "max_drawdown_pct" not in av:
            av["max_drawdown_pct"] = max_dd
    print("\n========== 当前最佳策略简述 ==========")
    print(f"来源：{source}")
    print(f"策略名：{data.get('strategy_class', '-')}")
    print(f"策略文件：{data.get('strategy_file', '-')}")
    print(f"训练区间：{goal.get('train_period', {}).get('timerange', '-')}")
    print(f"交易数：{_safe_int(tm.get('total_trades'))}")
    print(f"收益USDT：{_safe_float(tm.get('profit_total_abs')):.4f}")
    print(f"收益率：{_safe_float(tm.get('profit_total_pct')):.2f}%")
    print(f"胜率：{_safe_float(tm.get('winrate')) * 100:.2f}%")
    print(f"Profit Factor：{_safe_float(tm.get('profit_factor')):.4f}")
    print(f"最大回撤：{_safe_float(tm.get('max_drawdown_pct') or _safe_float(tm.get('max_drawdown')) * 100):.2f}%")
    print(f"验证区间平均收益率：{_safe_float(av.get('profit_total_pct')):.2f}%")
    print(f"验证区间平均 PF：{_safe_float(av.get('profit_factor')):.4f}")
    print(f"验证区间最大回撤：{_safe_float(av.get('max_drawdown_pct')):.2f}%")
    print(f"是否疑似过拟合：{'是' if data.get('is_overfit') else '否'}")
    print(f"主要优势：{data.get('why_best', '综合指标相对更优。')}")
    print("主要风险：仍需更多区间复验。")


def _aggregate_validation_metrics(validation_metrics: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not validation_metrics:
        return 0.0, 0.0, 0.0
    rows = [(item.get("metrics", {}) or {}) for item in validation_metrics]
    avg_profit = sum(_safe_float(x.get("profit_total_pct")) for x in rows) / len(rows)
    avg_pf = sum(_safe_float(x.get("profit_factor")) for x in rows) / len(rows)
    max_dd = max(
        (_safe_float(x.get("max_drawdown_pct")) if x.get("max_drawdown_pct") is not None else _safe_float(x.get("max_drawdown")) * 100.0)
        for x in rows
    )
    return avg_profit, avg_pf, max_dd



def _new_round_defaults() -> dict[str, Any]:
    """Return per-iteration defaults for values written to summaries.

    Keep this in one place so skipped/failed branches can still write a
    complete summary without referencing variables that are only assigned by
    later backtest branches.
    """
    return {
        "train_metrics": None,
        "validation_metrics": [],
        "validation_status": "not_run",
        "validation_skip_reason": "",
        "holdout_metrics": [],
        "holdout_status": "not_run",
        "holdout_reason": "",
        "pair_metrics": [],
        "entry_tag_metrics": [],
        "similarity_report": None,
        "final_score": 0,
        "score_breakdown": {},
        "invalid_reason": "",
        "is_valid": False,
        "is_best": False,
        "trade_under_min": False,
        "cannot_be_official_best_unless_validation_strong": False,
        "validation_strong": False,
        "trade_count_warning": "",
    }


def _apply_train_hard_constraint_status(
    state: dict[str, Any],
    hard_invalid_reason: str,
    validation_skip_reason: str | None = None,
) -> None:
    """Mark validation/holdout as skipped after a train hard constraint."""
    state["validation_metrics"] = []
    state["validation_status"] = "skipped"
    state["validation_skip_reason"] = validation_skip_reason or hard_invalid_reason
    state["holdout_metrics"] = []
    state["holdout_status"] = "skipped"
    state["holdout_reason"] = "训练区间触发硬约束，跳过验证和 holdout"
    state["is_valid"] = False
    state["is_best"] = False
    state["invalid_reason"] = hard_invalid_reason
    state["final_score"] = 0


def _ensure_holdout_not_run_reason(state: dict[str, Any]) -> None:
    """Populate the standard holdout not-run reason for non-best rounds."""
    if state.get("holdout_status") == "not_run" and not state.get("holdout_reason"):
        state["holdout_reason"] = "未达到候选 best 条件，未执行 holdout"


def _minimal_round_summary(
    *,
    version: str,
    strategy_class: str,
    strategy_file: str,
    state: dict[str, Any],
    mutation_type: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    """Build a minimal but schema-stable per-round summary."""
    _ensure_holdout_not_run_reason(state)
    return {
        "version": version,
        "strategy_class": strategy_class,
        "strategy_file": strategy_file,
        "mutation_type": mutation_type,
        "train_metrics": state.get("train_metrics"),
        "validation_metrics": state.get("validation_metrics", []),
        "validation_status": state.get("validation_status", "not_run"),
        "validation_skip_reason": state.get("validation_skip_reason", ""),
        "holdout_metrics": state.get("holdout_metrics", []),
        "holdout_status": state.get("holdout_status", "not_run"),
        "holdout_reason": state.get("holdout_reason", ""),
        "pair_metrics": state.get("pair_metrics", []),
        "entry_tag_metrics": state.get("entry_tag_metrics", []),
        "score_breakdown": state.get("score_breakdown", {}),
        "final_score": state.get("final_score", 0),
        "failure_reason": failure_reason,
        "is_best": bool(state.get("is_best", False)),
        "is_valid": bool(state.get("is_valid", False)),
        "invalid_reason": state.get("invalid_reason", ""),
        "similarity_report": state.get("similarity_report"),
        "trade_under_min": bool(state.get("trade_under_min", False)),
        "cannot_be_official_best_unless_validation_strong": bool(state.get("cannot_be_official_best_unless_validation_strong", False)),
        "validation_strong": bool(state.get("validation_strong", False)),
        "trade_count_warning": state.get("trade_count_warning", ""),
    }


def _is_validation_strong(
    validation_metrics: list[dict[str, Any]],
    baseline_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
) -> bool:
    """Return whether validation is strong enough for a near-min-trades candidate."""
    if not validation_metrics:
        return False
    avg_profit_pct, avg_profit_factor, max_drawdown_pct = _aggregate_validation_metrics(validation_metrics)
    baseline_profit_pct = _safe_float(baseline_cfg.get("profit_total_pct"))
    baseline_profit_factor = _safe_float(baseline_cfg.get("profit_factor"))
    max_drawdown_target = _safe_float(target_cfg.get("max_drawdown_pct", 3.0))
    return (
        avg_profit_pct >= max(0.0, baseline_profit_pct)
        and avg_profit_factor >= max(1.0, baseline_profit_factor)
        and (max_drawdown_target <= 0 or max_drawdown_pct <= max_drawdown_target)
    )

def extract_strategy_features(strategy_code: str) -> dict[str, Any]:
    lc = strategy_code.lower()

    def _extract(pattern: str) -> str | None:
        m = re.search(pattern, strategy_code, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    def _norm_list(tokens: list[str]) -> list[str]:
        return sorted(set(t.strip().lower() for t in tokens if t and t.strip()))

    indicators = _norm_list(re.findall(r"\b(rsi|ema\d*|macd|bbands|bollinger|adx|atr|volume|sma\d*)\b", lc))
    entry_conditions = _norm_list(re.findall(r"\((df\[.*?\]\s*[<>=!]+\s*[^\)]+)\)", strategy_code, flags=re.DOTALL))
    entry_tags = _norm_list(re.findall(r"[\"']([A-Za-z0-9_\-]+)[\"']\s*(?:,|\)|\])", _extract(r"entry_tag\s*=\s*(.*)") or ""))

    return {
        "minimal_roi": _extract(r"minimal_roi\s*=\s*(\{.*?\})"),
        "stoploss": _extract(r"stoploss\s*=\s*([-\d\.]+)"),
        "trailing_stop": _extract(r"trailing_stop\s*=\s*(True|False)"),
        "use_exit_signal": _extract(r"use_exit_signal\s*=\s*(True|False)"),
        "timeframe": _extract(r"timeframe\s*=\s*[\"']([^\"']+)[\"']"),
        "indicators": indicators,
        "entry_conditions": entry_conditions[:20],
        "entry_tags": entry_tags[:8],
    }


def _feature_signature(features: dict[str, Any]) -> str:
    key = {
        "minimal_roi": features.get("minimal_roi"),
        "stoploss": features.get("stoploss"),
        "trailing_stop": features.get("trailing_stop"),
        "entry_conditions": features.get("entry_conditions", []),
        "entry_tags": features.get("entry_tags", []),
        "indicators": features.get("indicators", []),
    }
    return hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def strategy_similarity(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, list[str]]:
    weights = {"entry_conditions": 0.30, "entry_tags": 0.15, "stoploss": 0.15, "minimal_roi": 0.15, "trailing_stop": 0.10, "indicators": 0.15}
    score = 0.0
    reasons: list[str] = []
    for k in ["stoploss", "minimal_roi", "trailing_stop"]:
        if str(a.get(k)) == str(b.get(k)) and a.get(k) is not None:
            score += weights[k]
            reasons.append(f"{k} 相同")
    for k in ["entry_conditions", "entry_tags", "indicators"]:
        sa, sb = set(a.get(k, [])), set(b.get(k, []))
        if sa and sb:
            j = len(sa & sb) / max(1, len(sa | sb))
            score += weights[k] * j
            if j >= 0.66:
                reasons.append(f"{k} 相似")
    return min(1.0, score), reasons


def build_compact_strategy_context(memory: list[dict[str, Any]], baseline: dict[str, Any], max_items: int = 5, max_chars: int = 2500) -> str:
    failed = [x for x in memory if not x.get("is_valid")]
    failed = failed[-max_items:]
    lines = [
        "当前最佳基准：",
        f"- profit_total_pct={_safe_float(baseline.get('profit_total_pct')):.2f}",
        f"- profit_factor={_safe_float(baseline.get('profit_factor')):.4f}",
        f"- max_drawdown_pct={_safe_float(baseline.get('max_drawdown_pct')):.2f}",
        f"- total_trades={_safe_int(baseline.get('total_trades'))}",
        "最近失败策略摘要：",
    ]
    for item in failed[-max_items:]:
        tm = item.get("train_metrics", {})
        avoid_next = item.get("avoid_next", [])
        if isinstance(avoid_next, str):
            avoid_next = [avoid_next]
        lines.append(
            f"- {item.get('version')}：交易数{_safe_int(tm.get('total_trades'))}，收益率{_safe_float(tm.get('profit_total_pct')):.2f}%"
            f"，PF{_safe_float(tm.get('profit_factor')):.2f}，回撤{_safe_float(tm.get('max_drawdown_pct')):.2f}%，失败={item.get('failure_reason', '未知')}，avoid_next={avoid_next}"
        )
    lines.extend([
        "禁止重复：",
        "- 不要生成与失败策略相同的 stoploss / ROI / trailing_stop 组合",
        "- 不要继续生成 300+ 交易的高频策略",
        "- 不要继续宽松 RSI/EMA/momentum 高频结构",
        "- 不要为了满足 min_trades 而过度放宽入场",
        "- 优先减少 stop_loss 损失和验证区间回撤",
    ])
    text = "\n".join(lines)
    return text[:max_chars]


def _score_zero_reason(
    final_score: float,
    train_metrics: dict[str, Any],
    validation_metrics: list[dict[str, Any]],
    validation_score: float,
) -> str | None:
    if abs(final_score) > 1e-12:
        return None
    if _safe_int(train_metrics.get("total_trades")) == 0:
        return "训练区间无交易，导致训练分数为0。"
    if not validation_metrics:
        return "没有可用的验证区间结果，验证分数按0处理。"
    if abs(validation_score) <= 1e-12:
        return "验证分数为0（收益、PF、胜率与回撤综合后接近0）。"
    return "综合评分公式计算结果为0。"


def load_project_env() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def validate_strategy_class_name(strategy_file: Path, class_name: str) -> None:
    content = strategy_file.read_text(encoding="utf-8")
    pattern = rf"class\s+{re.escape(class_name)}\s*\(\s*IStrategy\s*\)\s*:"
    if not re.search(pattern, content):
        raise RuntimeError(f"策略文件类名校验失败：{strategy_file.name} 中未找到 class {class_name}(IStrategy):")


class AIRequestFailed(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AIModelPoolExhaustedError(AIRequestFailed):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        attempts: list[dict[str, Any]] | None = None,
        used_model: str = "",
    ):
        super().__init__(message, status_code=status_code)
        self.attempts = attempts or []
        self.used_model = used_model


def _format_ai_error(exc: Exception) -> tuple[str, int | None]:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    detail = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("code") or "")
    if not detail:
        detail = str(exc)
    msg = f"{status_code} {detail}".strip() if status_code else detail
    return msg.strip(), status_code


def _is_403_provider_tos_block(error_message: str, status_code: int | None) -> bool:
    msg = (error_message or "").lower()
    return status_code == 403 and any(k in msg for k in ["terms of service", "prohibited", "guardrail", "moderation"])


def _is_retriable_ai_error(error_message: str, status_code: int | None, exc: Exception | None = None) -> bool:
    msg = (error_message or "").lower()
    retry_status_codes = {429, 500, 502, 503, 504}
    retry_keywords = [
        "system_cpu_overloaded",
        "system cpu overloaded",
        "auth_unavailable",
        "no auth available",
        "provider unavailable",
        "model unavailable",
        "upstream error",
        "gateway timeout",
        "rate limit",
        "timeout",
    ]
    if status_code in retry_status_codes:
        return True
    if exc is not None and isinstance(exc, (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError, TimeoutError)):
        return True
    return any(k in msg for k in retry_keywords)


def _print_auth_unavailable_hint(error_message: str) -> None:
    msg = (error_message or "").lower()
    if "auth_unavailable" in msg or "no auth available" in msg or "providers=codex" in msg:
        print("检测到中转站 provider 鉴权/通道不可用，这通常不是本地代码错误。")
        print("将自动切换同角色模型池中的下一个模型；如持续失败，请检查模型池、BASE_URL 或中转站分组。")


def _ai_backoff_seconds(failure_index: int, tos_block: bool = False) -> int:
    if tos_block:
        return 5
    return min(60, 10 * (2 ** max(0, failure_index - 1)))


def _role_pool_failed_message(role_name: str, attempts: list[dict[str, Any]], fallback_error: str) -> str:
    display = ROLE_DISPLAY_NAMES.get(role_name, role_name)
    failed_attempts = [a for a in attempts if a.get("status") == "failed"]
    if failed_attempts and all(int(a.get("status_code") or 0) == 403 and a.get("tos_blocked") for a in failed_attempts):
        return f"{display}模型池全部失败：403 provider TOS blocked"
    return f"{display}模型池全部失败：{fallback_error}"


def safe_ask_ai(
    role_runtime: AIRoleRuntime,
    messages: list[dict[str, str]],
    state: dict[str, Any],
) -> str:
    role_runtime.begin_call()
    if role_runtime.role not in {"strategy_advisor", "code_generator"}:
        raise ValueError(f"safe_ask_ai 仅支持 strategy_advisor/code_generator，收到：{role_runtime.role}")
    model_pool = [m for m in role_runtime.model_pool if m]
    if not model_pool:
        raise AIModelPoolExhaustedError(f"{role_runtime.display_name}模型池为空", attempts=[])

    max_attempts = max(1, int(role_runtime.max_attempts_per_call or 1))
    timeout_sec = max(1, int(role_runtime.timeout_sec or 1))
    last_error_message = "未知错误"
    last_status_code: int | None = None

    for attempt_idx in range(max_attempts):
        model_idx = attempt_idx % len(model_pool) if role_runtime.switch_on_error else 0
        model = model_pool[model_idx]
        next_model = model_pool[(model_idx + 1) % len(model_pool)] if len(model_pool) > 1 else model
        now = time.time()
        last_call = float(state.get("last_ai_call_time", 0.0) or 0.0)
        cooldown = max(0.0, float(state.get("ai_call_cooldown_seconds", 0.0) or 0.0))
        elapsed_since_last = now - last_call if last_call > 0 else -1.0
        wait_before_call = max(0.0, cooldown - elapsed_since_last) if elapsed_since_last >= 0 else 0.0
        print("准备调用 AI：")
        print(f"角色：{role_runtime.role}")
        print(f"模型池：{', '.join(model_pool)}")
        print(f"当前模型：{model}")
        print(f"尝试次数：{attempt_idx + 1}/{max_attempts}")
        if elapsed_since_last < 0:
            print("距离上次 AI 请求：首次调用")
        else:
            print(f"距离上次 AI 请求：{elapsed_since_last:.1f} 秒")
        print(f"本次请求前等待：{wait_before_call:.1f} 秒")
        if wait_before_call > 0:
            time.sleep(wait_before_call)

        start_ts = time.time()
        stop_event = threading.Event()

        def _heartbeat() -> None:
            while not stop_event.wait(10):
                waited = int(time.time() - start_ts)
                print(f"AI 正在生成策略中，已等待 {waited} 秒……")
                if waited > timeout_sec:
                    print(f"AI 调用超过 {timeout_sec} 秒，可能是中转站响应过慢。")

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                res = role_runtime.client.chat.completions.create(model=model, messages=messages, temperature=0.2)
            except (InternalServerError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, TimeoutError, Exception) as exc:
                error_message, status_code = _format_ai_error(exc)
                last_error_message = error_message or exc.__class__.__name__
                last_status_code = status_code
                tos_block = _is_403_provider_tos_block(last_error_message, status_code)
                retriable = tos_block or _is_retriable_ai_error(last_error_message, status_code, exc)
                role_runtime.attempts.append({
                    "model": model,
                    "status": "failed",
                    "error": last_error_message,
                    "status_code": status_code,
                    "tos_blocked": tos_block,
                })
                if status_code == 503 and "system cpu overloaded" in last_error_message.lower():
                    print("提示：这是模型服务/中转站负载过高，不是本地代码错误。")
                _print_auth_unavailable_hint(last_error_message)
                if not retriable:
                    raise AIModelPoolExhaustedError(
                        _role_pool_failed_message(role_runtime.role, role_runtime.attempts, last_error_message),
                        status_code=status_code,
                        attempts=role_runtime.attempts,
                    ) from exc
                wait_sec = _ai_backoff_seconds(attempt_idx + 1, tos_block=tos_block)
                print("AI 调用失败：")
                print(f"角色：{role_runtime.role}")
                print(f"模型：{model}")
                print(f"错误：{last_error_message}")
                if attempt_idx + 1 < max_attempts:
                    print(f"将在 {wait_sec} 秒后切换到下一个模型：{next_model}")
                    print(f"尝试次数：{attempt_idx + 1}/{max_attempts}")
                    time.sleep(wait_sec)
                    continue
                raise AIModelPoolExhaustedError(
                    _role_pool_failed_message(role_runtime.role, role_runtime.attempts, last_error_message),
                    status_code=status_code,
                    attempts=role_runtime.attempts,
                ) from exc
            finally:
                state["last_ai_call_time"] = time.time()
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1)

        content = (res.choices[0].message.content or "").strip()
        elapsed = int(time.time() - start_ts)
        role_runtime.used_model = model
        role_runtime.attempts.append({"model": model, "status": "success"})
        print("AI 调用成功：")
        print(f"角色：{role_runtime.role}")
        print(f"实际使用模型：{model}")
        print(f"用时：{elapsed} 秒")
        print(f"返回字符数：{len(content)}")
        return content

    raise AIModelPoolExhaustedError(
        _role_pool_failed_message(role_runtime.role, role_runtime.attempts, last_error_message),
        status_code=last_status_code,
        attempts=role_runtime.attempts,
    )


def _parse_model_pool(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _build_ai_role_runtime(
    model_cfg: dict[str, Any],
    role_name: str,
    timeout_sec: int,
    max_attempts_per_call: int,
    switch_on_error: bool,
) -> AIRoleRuntime:
    base_url = (os.getenv(str(model_cfg.get("base_url_env", ""))) or "").strip() or None
    api_key = (os.getenv(str(model_cfg.get("api_key_env", ""))) or "").strip()
    default_pool_env_by_role = {"strategy_advisor": "CLAUDE_MODEL_POOL", "code_generator": "OPENAI_MODEL_POOL"}
    pool_env_name = str(model_cfg.get("model_pool_env") or default_pool_env_by_role.get(role_name, ""))
    legacy_model_env = str(model_cfg.get("model_env", ""))
    model_pool = _parse_model_pool(os.getenv(pool_env_name)) if pool_env_name else []
    if not model_pool:
        legacy_model = (os.getenv(legacy_model_env) or str(model_cfg.get("default_model", ""))).strip()
        model_pool = [legacy_model] if legacy_model else []
    if not api_key:
        raise RuntimeError(f"未检测到 {model_cfg.get('api_key_env')}，无法初始化 {role_name} 模型池。")
    if not model_pool:
        raise RuntimeError(f"未检测到 {pool_env_name or legacy_model_env}，无法初始化 {role_name} 模型池。")
    return AIRoleRuntime(
        role=role_name,
        client=OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec),
        model_pool=model_pool,
        timeout_sec=timeout_sec,
        switch_on_error=switch_on_error,
        max_attempts_per_call=max_attempts_per_call,
    )


def _print_ai_model_pool_config(advisor_runtime: AIRoleRuntime, code_runtime: AIRoleRuntime, cooldown_seconds: float) -> None:
    print("\n========== AI 模型池配置 ==========")
    print("策略顾问模型池：")
    for idx, model in enumerate(advisor_runtime.model_pool, start=1):
        print(f"{idx}. {model}")
    print("\n代码生成模型池：")
    for idx, model in enumerate(code_runtime.model_pool, start=1):
        print(f"{idx}. {model}")
    print(f"\n模型失败自动轮换：{'开启' if advisor_runtime.switch_on_error and code_runtime.switch_on_error else '关闭'}")
    print(f"单次 AI 调用最大尝试次数：{max(advisor_runtime.max_attempts_per_call, code_runtime.max_attempts_per_call)}")
    print(f"AI 请求超时：{max(advisor_runtime.timeout_sec, code_runtime.timeout_sec)} 秒")
    print(f"AI 请求冷却：{cooldown_seconds:g} 秒")

def _strategy_spec_prompt(class_name: str, runtime_goal: dict[str, Any], baseline_cfg: dict[str, Any], compact_memory: str, previous_failure_reason: str | None) -> str:
    target_cfg = runtime_goal.get("target", {}) or {}
    min_trades = int(target_cfg.get("min_trades", 25))
    max_trades = int(target_cfg.get("max_trades", 80))
    min_trades_grace_ratio = float(target_cfg.get("min_trades_grace_ratio", 0.8))
    min_trades_grace_floor = int(min_trades * min_trades_grace_ratio)
    return (
        f"你是策略顾问模型。只输出 mutation_spec JSON，不要输出 Python 代码。建议 strategy_name={class_name}\n"
        f"goal={json.dumps(runtime_goal.get('target', {}), ensure_ascii=False)}\n"
        f"baseline={json.dumps(baseline_cfg, ensure_ascii=False)}\n"
        f"recent_failed={compact_memory}\n"
        f"last_failure={previous_failure_reason or '无'}\n"
        "必须只选择一个 mutation_type，允许值：add_entry_filter,tighten_entry_trigger,remove_bad_entry_condition,pair_specific_filter,tag_specific_filter,adjust_roi,adjust_stoploss,reduce_trade_frequency,disable_or_adjust_trailing,tighten_volume_filter,cooldown_or_protection。\n"
        "JSON 必须包含: mutation_type,reason,expected_effect,changes,do_not_change。\n"
        f"硬约束：本轮训练区间总交易数目标是 {min_trades}~{max_trades}（不是单币种）。低于 {min_trades_grace_floor} 会直接跳过验证；{min_trades_grace_floor}~{min_trades - 1} 会继续验证但仅作候选参考。超过 {max_trades} 不能成为 best，超过 {int(max_trades * 1.5)} 会直接跳过验证。\n"
        "重点：减少固定止损吞噬 ROI，不是增加交易数量。禁止/不推荐：increase_trade_frequency,loosen_entry,enable_trailing,adjust_stoploss_only,widen_stoploss。优先：add_entry_filter,tighten_entry_trigger,remove_bad_entry_condition,pair_specific_filter,tag_specific_filter。"
    )


def maybe_reset_best_strategy(reset_best: bool) -> bool:
    if not reset_best:
        return False
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = BEST_STRATEGY_FILE.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if BEST_STRATEGY_FILE.exists():
        backup_path = backup_dir / f"best_strategy.{now}.json"
        shutil.copy2(BEST_STRATEGY_FILE, backup_path)
        print(f"已备份历史 best_strategy.json -> {backup_path}")
    if BEST_STRATEGY_FILE.exists():
        BEST_STRATEGY_FILE.unlink()
    _append_reset_history("用户选择初始化历史最佳策略")
    print("已初始化历史最佳策略：当前 run 将从空 champion 开始。")
    return True


def _extract_exit_profit_fields(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "roi_profit_abs": _safe_float(metrics.get("roi_profit_abs")),
        "stop_loss_profit_abs": _safe_float(metrics.get("stop_loss_profit_abs")),
        "trailing_stop_loss_profit_abs": _safe_float(metrics.get("trailing_stop_loss_profit_abs")),
        "force_exit_profit_abs": _safe_float(metrics.get("force_exit_profit_abs")),
    }


def _is_true_or_one(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in {1, True}
    if isinstance(node, ast.NameConstant):
        return node.value in {1, True}
    if isinstance(node, ast.Tuple):
        return any(_is_true_or_one(el) for el in node.elts)
    if isinstance(node, ast.List):
        return any(_is_true_or_one(el) for el in node.elts)
    return False


def _target_contains_enter_long(target: ast.AST) -> bool:
    if isinstance(target, ast.Subscript):
        segment = ast.unparse(target)
        return "enter_long" in segment
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_contains_enter_long(el) for el in target.elts)
    return False


def check_entry_long_static(strategy_file: Path) -> tuple[bool, str | None]:
    content = strategy_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return False, f"静态检查失败：策略代码语法解析失败（{exc.msg}）"

    target_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "populate_entry_trend":
            target_func = node
            break
    if target_func is None:
        return False, "静态检查失败：缺少 populate_entry_trend 函数"

    body_text = "\n".join(ast.unparse(stmt) for stmt in target_func.body)
    if "enter_long" not in body_text:
        return False, "静态检查失败：populate_entry_trend 函数体未出现 enter_long"

    for stmt in ast.walk(target_func):
        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            value = stmt.value if hasattr(stmt, "value") else None
            if value is None:
                continue
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            else:
                targets = [stmt.target]
            if any(_target_contains_enter_long(t) for t in targets) and _is_true_or_one(value):
                return True, None

    return False, "静态检查失败：populate_entry_trend 中未检测到 enter_long 被赋值为 1 或 True"


def _validate_round(train_metrics: dict[str, Any], validation_metrics: list[dict[str, Any]], final_score: float) -> tuple[bool, str | None]:
    train_trades = _safe_int(train_metrics.get("total_trades"))
    if train_trades == 0:
        return False, "训练区间无交易"
    if not validation_metrics:
        return False, "验证区间结果缺失"
    if all(_safe_int(item.get("metrics", {}).get("total_trades")) == 0 for item in validation_metrics):
        return False, "所有验证区间无交易"
    if final_score <= 0:
        return False, "final_score<=0"
    return True, None


def _baseline_gate_and_penalty(
    train_metrics: dict[str, Any],
    runtime_goal: dict[str, Any],
) -> tuple[bool, str | None, float]:
    baseline = runtime_goal.get("baseline", {}) or {}
    target = runtime_goal.get("target", {}) or {}
    if not baseline:
        return True, None, 0.0

    profit_abs = _safe_float(train_metrics.get("profit_total_abs"))
    profit_pct = _safe_float(train_metrics.get("profit_total_pct"))
    pf = _safe_float(train_metrics.get("profit_factor"))
    dd_pct = _safe_float(train_metrics.get("max_drawdown")) * 100.0
    trades = _safe_int(train_metrics.get("total_trades"))

    b_profit_abs = _safe_float(baseline.get("profit_total_abs"))
    b_profit_pct = _safe_float(baseline.get("profit_total_pct"))
    b_pf = _safe_float(baseline.get("profit_factor"))
    b_dd_pct = _safe_float(baseline.get("max_drawdown_pct"))
    b_trades = _safe_int(baseline.get("total_trades"))
    target_max_dd = _safe_float(target.get("max_drawdown_pct"))

    if trades <= 0:
        return False, "交易数<=0", 0.0
    if target_max_dd > 0 and dd_pct > target_max_dd:
        return False, f"最大回撤超出目标({dd_pct:.2f}%>{target_max_dd:.2f}%)", 0.0
    if profit_abs < b_profit_abs - 1e-9:
        return False, "profit_total_abs 低于 baseline", 0.0

    better_count = 0
    if profit_abs >= b_profit_abs:
        better_count += 1
    if profit_pct >= b_profit_pct:
        better_count += 1
    if pf >= b_pf:
        better_count += 1
    if dd_pct <= b_dd_pct:
        better_count += 1
    if trades >= b_trades:
        better_count += 1
    if better_count < 3:
        return False, "综合表现不优于 baseline", 0.0

    dd_penalty = max(0.0, dd_pct - b_dd_pct) * 2.0
    return True, None, dd_penalty

def _build_periods(runtime_goal: dict[str, Any]) -> tuple[PeriodDef, list[PeriodDef]]:
    train_cfg = runtime_goal.get("train_period", {})
    train = PeriodDef(
        name=str(train_cfg.get("name", "train")),
        timerange=str(train_cfg.get("timerange", "")),
        weight=float(train_cfg.get("weight", 1.0)),
        kind="train",
    )
    validations: list[PeriodDef] = []
    for idx, item in enumerate(runtime_goal.get("validation_periods", []), start=1):
        validations.append(
            PeriodDef(
                name=str(item.get("name", f"valid_{idx:02d}")),
                timerange=str(item.get("timerange", "")),
                weight=float(item.get("weight", 1.0)),
                kind="validation",
            )
        )
    return train, validations


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    required = ["total_trades", "profit_total_abs", "profit_total", "profit_factor", "max_drawdown_account"]
    for key in required:
        if key not in result:
            raise RuntimeError(f"回测结果缺少字段 {key}")

    total_trades = int(result["total_trades"])
    profit_total_abs = float(result["profit_total_abs"])
    profit_total = float(result["profit_total"])
    profit_total_pct = profit_total * 100.0
    profit_factor = float(result["profit_factor"])
    max_drawdown = float(result["max_drawdown_account"])
    max_drawdown_pct = max_drawdown * 100.0

    print(f"total_trades: {total_trades}")
    print(f"profit_total_abs: {profit_total_abs}")
    print(f"profit_total_pct: {profit_total_pct}")
    print(f"profit_factor: {profit_factor}")
    print(f"max_drawdown_pct: {max_drawdown_pct}")

    exit_reason_details = parse_exit_reason_details(result)
    if not exit_reason_details.get("parsed", False):
        print("无法解析 exit reason 明细。")
    return {
        "total_trades": total_trades,
        "profit_total_abs": profit_total_abs,
        "profit_total": profit_total,
        "profit_total_pct": profit_total_pct,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "winrate": float(result.get("winrate", 0.0) or 0.0),
        **exit_reason_details,
        "pairs": result.get("results_per_pair", []),
        "entry_tags": result.get("results_per_enter_tag", []),
    }


def _normalize_pair_metrics(raw_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw_pairs or []:
        pair = str(item.get("key") or item.get("pair") or "")
        if not pair:
            continue
        trades = _safe_int(item.get("trades") or item.get("total_trades"))
        profit_abs = _safe_float(item.get("profit_total_abs"))
        profit_pct = _safe_float(item.get("profit_total")) * 100.0 if item.get("profit_total") is not None else _safe_float(item.get("profit_total_pct"))
        out.append({"pair": pair, "trades": trades, "profit_total_abs": profit_abs, "profit_total_pct": profit_pct, "profit_factor": _safe_float(item.get("profit_factor")), "winrate": _safe_float(item.get("winrate")), "max_drawdown_pct": _safe_float(item.get("max_drawdown_pct"))})
    return out


def _normalize_entry_tag_metrics(raw_tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw_tags or []:
        tag = str(item.get("key") or item.get("enter_tag") or "")
        if not tag:
            continue
        out.append({"enter_tag": tag, "trades": _safe_int(item.get("trades") or item.get("total_trades")), "profit_total_abs": _safe_float(item.get("profit_total_abs")), "profit_total_pct": (_safe_float(item.get("profit_total")) * 100.0 if item.get("profit_total") is not None else _safe_float(item.get("profit_total_pct"))), "profit_factor": _safe_float(item.get("profit_factor")), "winrate": _safe_float(item.get("winrate"))})
    return out


def _compute_final_score(train_score: float, validation_score: float, train_metrics: dict[str, Any], validation_metrics: list[dict[str, Any]], target_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    max_dd_target = _safe_float(target_cfg.get("max_drawdown_pct", 3.0))
    max_trades = _safe_int(target_cfg.get("max_trades", 80))
    worst_val_score = min((_score(v["metrics"], PeriodDef(v["period"], v["timerange"], 1.0, "validation")) for v in validation_metrics), default=0.0)
    penalty = 0.0
    penalty_reasons: list[str] = []
    for item in validation_metrics:
        m = item.get("metrics", {}) or {}
        if _safe_float(m.get("profit_total_pct")) < -2.5:
            penalty += 40.0
            penalty_reasons.append(f"{item.get('period')} profit_total_pct<-2.5%")
        if _safe_float(m.get("profit_factor")) < 0.45:
            penalty += 30.0
            penalty_reasons.append(f"{item.get('period')} PF<0.45")
        if max_dd_target > 0 and _safe_float(m.get("max_drawdown_pct")) > max_dd_target * 0.9:
            penalty += 12.0
            penalty_reasons.append(f"{item.get('period')} DD接近上限")
        if _safe_int(m.get("total_trades")) > max_trades:
            penalty += 16.0
            penalty_reasons.append(f"{item.get('period')} trades>{max_trades}")
    roi_abs = max(0.0, _safe_float(train_metrics.get("roi_profit_abs")))
    stop_loss_abs = abs(_safe_float(train_metrics.get("stop_loss_profit_abs")))
    if stop_loss_abs > roi_abs * 1.2:
        penalty += 30.0
        penalty_reasons.append("固定止损亏损>ROI*1.2")
    score = train_score * 0.4 + validation_score * 0.4 + worst_val_score * 0.2 - penalty
    return score, {"worst_validation_score": worst_val_score, "penalty_total": penalty, "penalty_reasons": penalty_reasons}


def _score(metrics: dict[str, Any], period: PeriodDef) -> float:
    return (
        metrics["profit_total_pct"] * 1.0
        + metrics["profit_factor"] * 8.0
        + metrics["winrate"] * 20.0
        - metrics["max_drawdown"] * 40.0
    ) * period.weight




def _parse_timerange(timerange: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not TIMERANGE_RE.match(timerange):
        raise ValueError(f"无效时间区间格式: {timerange}")
    start_s, end_s = timerange.split("-", 1)
    start = pd.to_datetime(start_s, format="%Y%m%d", utc=True)
    end = pd.to_datetime(end_s, format="%Y%m%d", utc=True)
    return start, end


def _pair_candidates(pair: str) -> list[str]:
    base = pair.replace(":", "").replace("/", "_")
    return [base, base.lower(), base.upper()]


def _find_data_file(data_dir: Path, pair: str, timeframe: str) -> Path | None:
    exts = ("feather", "parquet", "json", "json.gz")
    for pair_key in _pair_candidates(pair):
        for ext in exts:
            matches = sorted(data_dir.glob(f"**/{pair_key}-{timeframe}.{ext}"))
            if matches:
                return matches[0]
    return None


@lru_cache(maxsize=256)
def _read_data_coverage(file_path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    p = Path(file_path)
    try:
        if p.suffix == ".feather":
            df = pd.read_feather(p, columns=["date"])
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p, columns=["date"])
        else:
            df = pd.read_json(p, compression="infer")
    except Exception:
        return None
    if "date" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min(), dates.max()


def check_local_data_coverage(goal: dict[str, Any]) -> dict[str, Any]:
    config_path = ROOT_DIR / str(goal.get("config", ""))
    config_data = read_json(config_path) if config_path.exists() else {}
    exchange = str(config_data.get("exchange", {}).get("name") or goal.get("exchange") or "okx").lower()
    pair_whitelist = config_data.get("exchange", {}).get("pair_whitelist", [])
    if not isinstance(pair_whitelist, list):
        pair_whitelist = []
    timeframes = goal.get("data_download", {}).get("timeframes") or [goal.get("timeframe", "5m"), "1h"]
    timeframes = [str(tf) for tf in timeframes]

    needed_ranges = [goal.get("data_download", {}).get("download_timerange"), goal.get("train_period", {}).get("timerange")]
    needed_ranges.extend([x.get("timerange") for x in goal.get("validation_periods", []) if isinstance(x, dict)])
    needed_ranges = [x for x in needed_ranges if isinstance(x, str) and TIMERANGE_RE.match(x)]

    min_start: pd.Timestamp | None = None
    max_end: pd.Timestamp | None = None
    for tr in needed_ranges:
        s, e = _parse_timerange(tr)
        min_start = s if min_start is None else min(min_start, s)
        max_end = e if max_end is None else max(max_end, e)

    data_dir = ROOT_DIR / "user_data" / "data" / exchange
    missing: list[dict[str, str]] = []
    insufficient: list[dict[str, str]] = []

    for pair in pair_whitelist:
        for tf in timeframes:
            data_file = _find_data_file(data_dir, pair, tf)
            if data_file is None:
                missing.append({"pair": pair, "timeframe": tf})
                continue
            coverage = _read_data_coverage(str(data_file))
            if coverage is None or min_start is None or max_end is None:
                continue
            local_start, local_end = coverage
            if local_start > min_start or local_end < max_end:
                insufficient.append({
                    "pair": pair,
                    "timeframe": tf,
                    "local": f"{local_start.strftime('%Y%m%d')}-{local_end.strftime('%Y%m%d')}",
                    "required": f"{min_start.strftime('%Y%m%d')}-{max_end.strftime('%Y%m%d')}",
                })

    return {
        "exchange": exchange,
        "pair_whitelist": pair_whitelist,
        "timeframes": timeframes,
        "download_timerange": goal.get("data_download", {}).get("download_timerange"),
        "required_range": None if min_start is None or max_end is None else f"{min_start.strftime('%Y%m%d')}-{max_end.strftime('%Y%m%d')}",
        "missing": missing,
        "insufficient": insufficient,
        "is_covered": not missing and not insufficient,
    }


def maybe_download_data(runtime_goal: dict[str, Any], args: argparse.Namespace, train_timerange: str) -> None:
    if args.skip_download:
        print("已启用 --skip-download：直接跳过数据下载。")
        return

    dcfg = runtime_goal.setdefault("data_download", {})
    if not dcfg.get("auto_download", True):
        print("配置 data_download.auto_download=false，跳过数据下载。")
        return

    if not dcfg.get("timeframes"):
        dcfg["timeframes"] = [runtime_goal.get("timeframe", args.timeframe), "1h"]

    if args.force_download:
        print("已启用 --force-download：强制重新下载历史数据。")
    else:
        coverage = check_local_data_coverage(runtime_goal)
        if coverage["is_covered"]:
            print("本地历史数据已存在，覆盖目标区间，跳过下载。")
            return
        if coverage["missing"]:
            print("缺少数据：")
            for item in coverage["missing"]:
                print(f"- {item['pair']} {item['timeframe']}")
        for item in coverage["insufficient"]:
            print(
                f"{item['pair']} {item['timeframe']} 本地数据范围为 {item['local']}，"
                f"但目标需要 {item['required']}，将执行补充下载。"
            )
        print("将执行 download-data 补齐。")

    dtr = str(dcfg.get("download_timerange", train_timerange))
    exchange = str(read_json(ROOT_DIR / runtime_goal["config"]).get("exchange", {}).get("name", "okx")).lower()
    tf_list = [str(x) for x in dcfg.get("timeframes", [])]
    cmd = [
        "docker", "compose", "run", "--rm", "freqtrade", "download-data",
        "--config", str(runtime_goal.get("config", args.config)), "--exchange", exchange,
        "--timeframes", *tf_list, "--timerange", dtr, "--prepend",
    ]
    cp = run_cmd(cmd, ROOT_DIR)
    print(cp.stdout)
    if cp.returncode != 0:
        print(cp.stderr)
        raise RuntimeError("下载历史数据失败。")

def run_auto_optimization(runtime_goal: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> None:
    config = str(runtime_goal.get("config", args.config))
    timeframe = str(runtime_goal.get("timeframe", args.timeframe))
    strategy_family = str(runtime_goal.get("strategy_family", args.base_strategy))
    iterations = int(runtime_goal.get("max_iterations", args.iterations))
    train, validations = _build_periods(runtime_goal)

    if not train.timerange:
        raise RuntimeError("缺少训练区间 train_period.timerange，无法继续。")

    maybe_download_data(runtime_goal, args, train.timerange)

    model_config = ensure_model_config_files()
    advisor_cfg = model_config.get("strategy_advisor", {})
    generator_cfg = model_config.get("code_generator", {})
    max_attempts_per_call = max(1, int(args.ai_model_max_attempts_per_call))
    switch_on_error = bool(args.ai_model_switch_on_error)
    advisor_runtime = _build_ai_role_runtime(
        advisor_cfg,
        "strategy_advisor",
        args.ai_timeout,
        max_attempts_per_call,
        switch_on_error,
    )
    code_runtime = _build_ai_role_runtime(
        generator_cfg,
        "code_generator",
        args.ai_timeout,
        max_attempts_per_call,
        switch_on_error,
    )
    code_repair_runtime = AIRoleRuntime(
        role="code_generator",
        client=code_runtime.client,
        model_pool=list(code_runtime.model_pool),
        timeout_sec=code_runtime.timeout_sec,
        switch_on_error=code_runtime.switch_on_error,
        max_attempts_per_call=code_runtime.max_attempts_per_call,
    )
    _print_ai_model_pool_config(advisor_runtime, code_runtime, float(args.ai_call_cooldown_seconds))

    best: dict[str, Any] | None = None
    session_best: dict[str, Any] | None = None
    reset_best_used = bool(getattr(args, "reset_best", False) or runtime_goal.get("runtime_reset_best", False))
    champion = {"meta": _build_baseline_best(runtime_goal), "code": "", "source": "reset_empty"} if reset_best_used else _load_champion(runtime_goal)
    _print_champion_source(champion)
    _print_starting_champion_summary(runtime_goal, champion)
    nearest_candidate: dict[str, Any] | None = None
    historical_best_mem = _load_json_or_none(BEST_STRATEGY_FILE)
    nearest_mem = _load_json_or_none(NEAREST_CANDIDATE_FILE)
    last_run_summary_mem = _load_json_or_none(LAST_RUN_SUMMARY_FILE)
    print("========== 记忆加载状态 ==========")
    h_status = "reset" if _is_reset_best(historical_best_mem) else ("存在" if historical_best_mem else "不存在")
    print(f"historical_best：{h_status}")
    print(f"historical_best 策略文件：{_strategy_file_valid((historical_best_mem or {}).get('strategy_file')) if historical_best_mem else '不存在'}")
    print(f"nearest_candidate：{_memory_presence(NEAREST_CANDIDATE_FILE)}")
    print(f"nearest_candidate 策略文件：{_strategy_file_valid((nearest_mem or {}).get('strategy_file')) if nearest_mem else '不存在'}")
    print(f"last_run_summary：{_memory_presence(LAST_RUN_SUMMARY_FILE)}")
    used_failed_mutations: set[str] = set()
    run_id = run_dir.name.replace("run_", "")
    ensure_runtime_json_file(MEMORY_FILE, MEMORY_EXAMPLE_FILE)
    ensure_runtime_json_file(BLACKLIST_FILE, BLACKLIST_EXAMPLE_FILE)
    ensure_runtime_json_file(LESSONS_FILE, LESSONS_EXAMPLE_FILE)

    memory_items = _read_json_list_file(MEMORY_FILE)
    blacklist_items = _read_json_list_file(BLACKLIST_FILE)
    lessons_items = _read_json_list_file(LESSONS_FILE)
    print(f"strategy_memory：{_memory_presence(MEMORY_FILE)}")
    print(f"strategy_lessons：{_memory_presence(LESSONS_FILE)}")
    print(f"strategy_blacklist：{_memory_presence(BLACKLIST_FILE)}")
    memory_cfg = runtime_goal.get("memory", {}) or {}
    memory_enabled = bool(memory_cfg.get("enabled", True))
    memory_max_items = int(memory_cfg.get("max_items", 5))
    memory_max_chars = int(memory_cfg.get("max_prompt_chars", 2500))
    avoid_similar = bool(memory_cfg.get("avoid_similar_failed_strategies", True))
    prev_train_trades: int | None = None
    previous_failure_reason: str | None = None
    zero_trade_streak = 0
    leaderboard: list[dict[str, Any]] = []
    best_summary_path: Path | None = None
    stop_on_ai_error = bool(runtime_goal.get("stop_on_ai_error", False))
    ai_runtime_state = {"last_ai_call_time": 0.0, "ai_call_cooldown_seconds": float(args.ai_call_cooldown_seconds)}
    iteration_stats_path = run_dir / ITERATION_STATS_FILE_NAME
    version_statuses: list[dict[str, Any]] = []
    iteration_stats: dict[str, Any] = {
        "planned_iterations": int(runtime_goal.get("max_iterations", args.iterations)),
        "advisor_success_count": 0,
        "codegen_success_count": 0,
        "generated_versions_count": 0,
        "train_backtest_count": 0,
        "validation_backtest_count": 0,
        "validation_backtest_total_count": 0,
        "skipped_validation_count": 0,
        "valid_strategy_count": 0,
        "invalid_strategy_count": 0,
        "new_best_update_count": 0,
        "current_iteration_version": "",
        "history_strategy_total_count": 0,
        "version_statuses": version_statuses,
    }

    def flush_iteration_stats() -> None:
        iteration_stats["history_strategy_total_count"] = len(_read_json_list_file(MEMORY_FILE)) if MEMORY_FILE.exists() else 0
        write_json(iteration_stats_path, iteration_stats)
    for i in range(1, iterations + 1):
        ver = f"v{i:03d}"
        iteration_stats["current_iteration_version"] = ver
        class_name = f"{strategy_family}_{run_id}_{ver}"
        strategy_file = STRATEGY_DIR / f"{class_name}.py"
        version_dir = run_dir / ver
        version_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "version": ver,
            "strategy_class": class_name,
            "advisor_status": "未执行",
            "codegen_status": "未执行",
            "syntax_check_status": "未执行",
            "static_check_status": "未启用",
            "train_backtest_status": "未执行",
            "validation_backtest_status": "未执行",
            "is_valid": False,
            "is_best": False,
            "invalid_reason": "",
            "final_score": 0.0,
        }
        version_statuses.append(status)
        advisor_runtime.begin_call()
        code_runtime.begin_call()
        code_repair_runtime.begin_call()
        round_state = _new_round_defaults()
        train_metrics = round_state["train_metrics"]
        validation_metrics = round_state["validation_metrics"]
        validation_status = round_state["validation_status"]
        validation_skip_reason = round_state["validation_skip_reason"]
        holdout_metrics = round_state["holdout_metrics"]
        holdout_status = round_state["holdout_status"]
        holdout_reason = round_state["holdout_reason"]
        pair_metrics = round_state["pair_metrics"]
        entry_tag_metrics = round_state["entry_tag_metrics"]
        similarity_report = round_state["similarity_report"]
        final_score = round_state["final_score"]
        score_breakdown = round_state["score_breakdown"]
        invalid_reason = round_state["invalid_reason"]
        is_valid = round_state["is_valid"]
        is_best = round_state["is_best"]
        mutation_type = ""
        spec_hash = ""
        code_hash = ""
        features: dict[str, Any] = {}
        failure_reason = ""
        trade_under_min = bool(round_state["trade_under_min"])
        cannot_be_official_best_unless_validation_strong = bool(round_state["cannot_be_official_best_unless_validation_strong"])
        validation_strong = bool(round_state["validation_strong"])
        trade_count_warning = str(round_state["trade_count_warning"])
        summary_write_failed = False

        def current_ai_models_used() -> dict[str, Any]:
            return {
                "strategy_advisor": advisor_runtime.usage_snapshot(),
                "code_generator": code_runtime.usage_snapshot(),
            }

        def enrich_leaderboard_entry(entry: dict[str, Any]) -> dict[str, Any]:
            usage = current_ai_models_used()
            advisor_usage = usage.get("strategy_advisor", {}) or {}
            codegen_usage = usage.get("code_generator", {}) or {}
            entry.setdefault("advisor_model_used", advisor_usage.get("used_model", ""))
            entry.setdefault("codegen_model_used", codegen_usage.get("used_model", ""))
            entry.setdefault("advisor_attempt_count", len(advisor_usage.get("attempts", []) or []))
            entry.setdefault("codegen_attempt_count", len(codegen_usage.get("attempts", []) or []))
            return entry

        def sync_round_state() -> None:
            round_state.update({
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "validation_status": validation_status,
                "validation_skip_reason": validation_skip_reason,
                "holdout_metrics": holdout_metrics,
                "holdout_status": holdout_status,
                "holdout_reason": holdout_reason,
                "pair_metrics": pair_metrics,
                "entry_tag_metrics": entry_tag_metrics,
                "similarity_report": similarity_report,
                "final_score": final_score,
                "score_breakdown": score_breakdown,
                "invalid_reason": invalid_reason,
                "is_valid": is_valid,
                "is_best": is_best,
                "trade_under_min": trade_under_min,
                "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
                "validation_strong": validation_strong,
                "trade_count_warning": trade_count_warning,
            })

        def write_iteration_summary(extra: dict[str, Any] | None = None) -> Path:
            nonlocal is_valid, is_best, invalid_reason, summary_write_failed
            sync_round_state()
            summary_data = _minimal_round_summary(
                version=ver,
                strategy_class=class_name,
                strategy_file=str(strategy_file),
                state=round_state,
                mutation_type=mutation_type,
                failure_reason=failure_reason,
            )
            summary_data["ai_models_used"] = current_ai_models_used()
            if extra:
                summary_data.update(extra)
            summary_path_local = version_dir / "summary.json"
            try:
                write_json(summary_path_local, summary_data)
            except Exception as exc:  # noqa: BLE001 - keep the optimizer running between rounds.
                print(f"写入 summary.json 失败：{exc}。将写入最小 summary 并继续下一轮。")
                summary_write_failed = True
                is_valid = False
                is_best = False
                invalid_reason = f"summary 写入失败：{exc}"
                round_state["is_valid"] = False
                round_state["is_best"] = False
                round_state["invalid_reason"] = f"summary 写入失败：{exc}"
                minimal_summary = _minimal_round_summary(
                    version=ver,
                    strategy_class=class_name,
                    strategy_file=str(strategy_file),
                    state=round_state,
                    mutation_type=mutation_type,
                    failure_reason=failure_reason,
                )
                try:
                    summary_path_local.write_text(json.dumps(minimal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as second_exc:  # noqa: BLE001
                    print(f"写入最小 summary 仍失败：{second_exc}。继续下一轮。")
            return summary_path_local

        print(f"\n========== 第 {i} 轮 / {ver} ==========")
        print("1. 正在生成 mutation_spec（冠军-挑战者小步改动）……")
        print(f"当前策略顾问模型池：{', '.join(advisor_runtime.model_pool)}")
        print(f"当前代码生成模型池：{', '.join(code_runtime.model_pool)}")
        target_cfg = runtime_goal.get("target", {}) or {}
        baseline_cfg = runtime_goal.get("baseline", {}) or {}
        min_trades = int(target_cfg.get("min_trades", 25))
        max_trades = int(target_cfg.get("max_trades", 80))
        min_trades_grace_ratio = float(target_cfg.get("min_trades_grace_ratio", 0.8))
        allow_near_min_trades_best = bool(
            getattr(args, "allow_near_min_trades_best", False)
            or runtime_goal.get("allow_near_min_trades_best", False)
            or target_cfg.get("allow_near_min_trades_best", False)
        )
        zero_trade_hint = (
            "上一轮失败原因：训练区间和所有验证区间均为 0 交易，请放宽入场条件。"
            if prev_train_trades == 0 else
            "优先保证训练区间有稳定交易，不要把过滤条件堆得过严。"
        )
        failure_context = f"上一轮失败原因：{previous_failure_reason}\n" if previous_failure_reason else ""
        failure_context += "最近失败策略共同原因通常不是没有盈利单，而是固定止损或 trailing_stop_loss 吃掉 ROI 收益。\n"
        compact_memory = build_compact_strategy_context(memory_items, baseline_cfg, memory_max_items, memory_max_chars) if memory_enabled else ""
        session_parent_candidates = {
            "historical_best": historical_best_mem if historical_best_mem and not _is_reset_best(historical_best_mem) else None,
            "nearest_candidate": nearest_mem,
            "baseline": {"strategy_class": "baseline", "train_metrics": baseline_cfg},
        }
        official_champion_name = "historical_best" if session_parent_candidates["historical_best"] else "baseline"
        print(f"当前正式 champion：{official_champion_name}")
        print("当前 session_parent 候选：")
        print(f"- historical_best: {(session_parent_candidates['historical_best'] or {}).get('strategy_class', '无')}")
        print(f"- nearest_candidate: {(session_parent_candidates['nearest_candidate'] or {}).get('strategy_class', '无')}")
        spec_prompt = _strategy_spec_prompt(class_name, runtime_goal, baseline_cfg, compact_memory, previous_failure_reason)
        if last_run_summary_mem:
            spec_prompt += "\nlast_run_summary=" + json.dumps(last_run_summary_mem, ensure_ascii=False)[:4000] + "\n"
        spec_prompt += f"\nchampion_strategy_class={champion.get('meta', {}).get('strategy_class', 'baseline')}\n"
        spec_prompt += f"\n已失败 mutation_type（避免重复）={sorted(used_failed_mutations)}\n"
        (version_dir / "advisor_prompt.txt").write_text(spec_prompt, encoding="utf-8")
        print("正在调用策略顾问模型生成 mutation_spec……")
        try:
            spec_text = safe_ask_ai(
                advisor_runtime,
                [{"role": "user", "content": spec_prompt}],
                state=ai_runtime_state,
            )
            print("策略顾问模型返回完成。")
            status["advisor_status"] = "成功"
            iteration_stats["advisor_success_count"] += 1
            delay_sec = max(0.0, float(args.advisor_to_codegen_delay_seconds))
            if delay_sec > 0:
                print(f"策略顾问模型完成，将等待 {int(delay_sec)} 秒后调用代码生成模型，避免中转站请求过快。")
                time.sleep(delay_sec)
        except AIRequestFailed as exc:
            spec_text = ""
            err_msg = str(exc)
            previous_failure_reason = f"策略顾问模型调用失败：{err_msg}"
            invalid_reason = previous_failure_reason
            write_iteration_summary()
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            status["advisor_status"] = "AI 调用失败"
            status["invalid_reason"] = invalid_reason
            iteration_stats["invalid_strategy_count"] += 1
            print(previous_failure_reason)
            flush_iteration_stats()
            if stop_on_ai_error:
                print("已设置 stop_on_ai_error=true，停止后续轮次。")
                break
            print(f"8. 第 {i} 轮完成：无效，原因：{invalid_reason}")
            continue
        (version_dir / "strategy_spec.raw.txt").write_text(spec_text or "", encoding="utf-8")
        if not spec_text:
            previous_failure_reason = "strategy_advisor 生成 mutation_spec 失败。"
            invalid_reason = "mutation_spec JSON 解析失败"
            write_iteration_summary()
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            print(f"strategy_spec 原始返回已保存：{version_dir / 'strategy_spec.raw.txt'}")
            print(f"8. 第 {i} 轮完成：无效，原因：{invalid_reason}")
            continue
        try:
            strategy_spec = extract_json_object(spec_text)
        except (ValueError, json.JSONDecodeError):
            previous_failure_reason = "mutation_spec 不是有效 JSON object。"
            invalid_reason = "mutation_spec JSON 解析失败"
            write_iteration_summary()
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            print(f"strategy_spec 解析失败，请检查：{version_dir / 'strategy_spec.raw.txt'}")
            print(f"8. 第 {i} 轮完成：无效，原因：{invalid_reason}")
            continue
        write_json(version_dir / "mutation_spec.json", strategy_spec)
        print(f"2. mutation_spec 已保存：{version_dir / 'mutation_spec.json'}")
        spec_hash = hashlib.sha256(json.dumps(strategy_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        mutation_type = str(strategy_spec.get("mutation_type", "") or "")
        prompt = (
            f"请基于 champion 策略代码进行一次最小修改生成 challenger，只输出 Python 代码。类名必须为 {class_name}，继承 IStrategy，"
            f"timeframe='{timeframe}'，并实现 populate_indicators/populate_entry_trend/populate_exit_trend。\n"
            f"champion_strategy_code=\n{champion.get('code', '')}\n"
            f"mutation_spec={json.dumps(strategy_spec, ensure_ascii=False)}\n"
            f"当前失败摘要={previous_failure_reason or '无'}\n"
            "硬性约束：\n"
            "1) 仅允许在 champion 基础上一次小改动，不允许完全重写。\n"
            "2) 策略必须在训练区间产生合理交易，严禁生成完全无交易策略。\n"
            f"2) 训练区间总交易数目标为 {min_trades}~{max_trades}（不是单币种）。低于 {int(min_trades * min_trades_grace_ratio)} 笔直接跳过验证；{int(min_trades * min_trades_grace_ratio)}~{min_trades - 1} 笔继续验证但仅作候选参考。超过 {max_trades} 不能成为 best；超过 {int(max_trades * 1.5)} 直接跳过验证。\n"
            "3) 禁止连续状态型宽松入场；优先 crossed_above/crossed_below 事件触发；每个策略最多 1~2 个 entry_tag。\n"
            "4) 不允许多个 OR 条件堆叠造成高频；不允许为了增加交易数而放宽入场。\n"
            f"3) {zero_trade_hint}\n"
            "4) 如果上一轮 total_trades=0，本轮必须大幅放宽入场条件，并确保训练区间产生交易。\n"
            "5) 目标训练区间交易数至少 25 笔，理想目标 25~80 笔。\n"
            "6) use_exit_signal 必须为 False，不允许改为 True。\n"
            "7) 仅现货 long only：不做空、不杠杆、不马丁格尔、不无限补仓，不允许 conditions_short。\n"
            "8) 不调用外部 API，不读取手动交易记录。\n"
            "9) 禁止使用过强过滤的全 AND 叠加（如 close>ema200_1h、rsi_1h>55、ema20>ema50>ema100、volume>rolling_mean*1.5 同时成立）。\n"
            "10) 入场逻辑可更宽松，鼓励用 OR 组合：RSI 回调反弹 / EMA 短周期金叉 / 布林带下轨反弹 / MACD 转强 / 成交量不极低。\n"
            "11) 不允许生成完全无交易策略，也不允许 200+ 训练交易。\n"
            "12) 目标不是追求 0 回撤，而是在足够交易数下综合表现优于 baseline。\n"
            f"{failure_context}"
            f"{compact_memory}\n"
            "当前 baseline：\n"
            f"- 总收益(USDT)：{baseline_cfg.get('profit_total_abs', -7.43)}\n"
            f"- 收益率(%)：{baseline_cfg.get('profit_total_pct', -0.74)}\n"
            f"- Profit Factor：{baseline_cfg.get('profit_factor', 0.63)}\n"
            f"- 最大回撤(%)：{baseline_cfg.get('max_drawdown_pct', 1.45)}\n"
            f"- 交易数：{baseline_cfg.get('total_trades', 47)}\n"
            "输出要求：\n"
            "- 只输出可运行的完整 Python 策略代码，不要解释。\n"
            "- 避免把入场条件写成几乎永远不触发的苛刻组合。\n"
        )
        (version_dir / "codegen_prompt.txt").write_text(prompt, encoding="utf-8")
        print("3. 正在调用代码生成模型池生成 Freqtrade 策略代码……")
        response_text = ""
        try:
            response_text = safe_ask_ai(
                code_runtime,
                [{"role": "user", "content": prompt}],
                state=ai_runtime_state,
            )
        except AIRequestFailed as exc:
            previous_failure_reason = f"代码生成模型调用失败：{str(exc)}"
            invalid_reason = previous_failure_reason
            print("本轮停止：代码生成模型多次失败，跳过本轮回测。")
            if response_text:
                (version_dir / "codegen.raw.txt").write_text(response_text, encoding="utf-8")
            write_iteration_summary()
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            continue
        (version_dir / "codegen.raw.txt").write_text(response_text, encoding="utf-8")
        status["codegen_status"] = "成功"
        iteration_stats["codegen_success_count"] += 1
        code = extract_python_code(response_text)
        features = extract_strategy_features(code)
        signature = _feature_signature(features)
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        similarity_threshold = float(args.similarity_threshold)
        similarity_report = {"is_similar": False, "similarity_score": 0.0, "similar_to": "", "similar_type": "none", "reasons": [], "decision": "auto_continue", "user_prompt_required": False}
        parent_pool = [
            ("session_parent", champion.get("meta") if champion.get("meta") else None),
            ("historical_best", session_parent_candidates.get("historical_best")),
            ("nearest_candidate", session_parent_candidates.get("nearest_candidate")),
        ]
        best_match = {"score": 0.0, "type": "none", "item": None, "reasons": []}
        for t, item in parent_pool:
            if not isinstance(item, dict):
                continue
            score, reasons = strategy_similarity(features, item.get("features", {}))
            if score > best_match["score"]:
                best_match = {"score": score, "type": t, "item": item, "reasons": reasons}
        for item in blacklist_items[-120:]:
            score, reasons = strategy_similarity(features, item.get("features", {}))
            if score > best_match["score"]:
                best_match = {"score": score, "type": "failed_blacklist", "item": item, "reasons": reasons}
        if best_match["score"] >= similarity_threshold:
            similarity_report.update({
                "is_similar": True,
                "similarity_score": round(float(best_match["score"]), 4),
                "similar_to": (best_match["item"] or {}).get("strategy_class") or (best_match["item"] or {}).get("version") or "unknown",
                "similar_type": best_match["type"],
                "reasons": best_match["reasons"],
            })
        print("新策略相似度检测：")
        print(f"- 相似对象：{similarity_report['similar_to'] or '无'}")
        print(f"- 相似对象类型：{similarity_report['similar_type']}")
        print(f"- 相似度：{float(similarity_report['similarity_score']):.2f}")
        print(f"- 相似原因：{'、'.join(similarity_report['reasons']) if similarity_report['reasons'] else '无'}")
        if similarity_report["similar_type"] in {"session_parent", "historical_best", "nearest_candidate"} and args.auto_continue_if_similar_to_parent:
            print(f"新策略与 {similarity_report['similar_type']} 相似，这是小步迭代预期行为，继续回测。")
            print("相似对象为当前 session_parent，属于预期小步修改，自动继续回测。")
            similarity_report["decision"] = "auto_continue"
        elif avoid_similar and similarity_report["similar_type"] == "failed_blacklist":
            if args.auto_approve and args.auto_reject_failed_similarity:
                similarity_report["decision"] = "auto_reject"
                write_json(version_dir / "similarity_report.json", similarity_report)
                invalid_reason = "自动拒绝：与失败黑名单高度相似"
                write_iteration_summary({"similarity_report": similarity_report})
                leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": "自动拒绝：与失败黑名单高度相似"})
                continue
            if not args.auto_approve:
                similarity_report["user_prompt_required"] = True
                similarity_report["decision"] = "user_prompt"
                print("新策略与历史失败黑名单高度相似，且不是当前 session_parent。")
                print("建议重新生成。")
                yn = input("是否仍然执行回测？(y/n)\n").strip()
                if parse_yes_no(yn) is not True:
                    similarity_report["decision"] = "user_reject"
                    write_json(version_dir / "similarity_report.json", similarity_report)
                    invalid_reason = "用户拒绝回测相似失败策略。"
                    write_iteration_summary({"similarity_report": similarity_report})
                    leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": "用户拒绝回测相似失败策略。"})
                    continue
                similarity_report["decision"] = "user_override_continue"
        print(f"- 是否允许自动继续：{'是' if similarity_report['decision'] in {'auto_continue', 'user_override_continue'} else '否'}")
        write_json(version_dir / "similarity_report.json", similarity_report)
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        strategy_file.write_text(code, encoding="utf-8")
        shutil.copy2(strategy_file, version_dir / "strategy.py")
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(strategy_file, GENERATED_DIR / strategy_file.name)
        iteration_stats["generated_versions_count"] += 1
        print(f"4. 策略代码已保存：{version_dir / 'strategy.py'}")

        print("5. 正在检查 Python 语法……")
        pyc = run_cmd([sys.executable, "-m", "py_compile", str(strategy_file)], ROOT_DIR)
        if pyc.returncode != 0:
            print("策略代码语法错误，尝试修复一次。")
            repair_prompt = f"请修复以下策略代码，仅输出可运行 Python：\n错误信息:\n{pyc.stderr}\n代码:\n{code}"
            try:
                repaired = safe_ask_ai(
                    code_repair_runtime,
                    [{"role": "user", "content": repair_prompt}],
                    state=ai_runtime_state,
                )
            except AIRequestFailed as exc:
                repaired = ""
                print(f"代码修复模型调用失败：{exc}")
            if repaired:
                code = extract_python_code(repaired)
                strategy_file.write_text(code, encoding="utf-8")
                pyc = run_cmd([sys.executable, "-m", "py_compile", str(strategy_file)], ROOT_DIR)
            if pyc.returncode != 0:
                print(pyc.stderr)
                print(f"第 {i} 轮语法检查失败，跳过。")
                continue
        status["syntax_check_status"] = "成功"
        validate_strategy_class_name(strategy_file, class_name)
        static_ok, static_reason = check_entry_long_static(strategy_file)
        if not static_ok:
            status["static_check_status"] = "失败"
            invalid_reason = static_reason or "静态检查失败"
            previous_failure_reason = invalid_reason
            card = {
                "version": ver,
                "run_id": run_id,
                "strategy_class": class_name,
                "strategy_file": str(strategy_file),
                "code_hash": code_hash,
                "created_at": datetime.utcnow().isoformat(),
                "features": features,
                "failure_reason": invalid_reason,
                "avoid_next": "确保 populate_entry_trend 对 enter_long 赋值为 1/True。",
                "final_score": 0.0,
                "train_profit_pct": 0.0,
                "avg_validation_profit_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "is_overfit": False,
                "is_best": False,
                "is_valid": False,
                "invalid_reason": invalid_reason,
            }
            leaderboard.append(card)
            memory_items.append(card)
            print(f"第 {i} 轮无效：{invalid_reason}")
            print(f"策略文件路径：{strategy_file}")
            print(f"策略类名：{class_name}")
            print(f"失败原因：{invalid_reason}")
            print("本轮策略未通过静态检查，没有执行回测。")
            print("生成策略已保存，可手动查看：")
            print(f"sed -n '1,260p' {strategy_file}")
            print("建议手动回测命令：")
            print("docker compose run --rm freqtrade backtesting \\")
            print(f"  --config {config} \\")
            print(f"  --strategy {class_name} \\")
            print(f"  --timeframe {timeframe} \\")
            print(f"  --timerange {train.timerange} \\")
            print("  --export trades \\")
            print("  --cache none")
            continue
        status["static_check_status"] = "成功"

        print(f"6. 正在回测训练区间：{train.timerange}")
        train_cmd = [
            "docker", "compose", "run", "--rm", "freqtrade", "backtesting",
            "--config", config, "--strategy", class_name, "--timeframe", timeframe,
            "--timerange", train.timerange, "--export", "trades", "--cache", "none",
        ]
        results_dir = ROOT_DIR / "user_data" / "backtest_results"
        train_before_zips = set(_list_backtest_zips(results_dir))
        train_start_ts = time.time()
        train_cp = run_cmd(train_cmd, ROOT_DIR)
        (version_dir / "backtest_logs.txt").write_text(
            f"[Train {train.timerange}]\nSTDOUT:\n{train_cp.stdout}\n\nSTDERR:\n{train_cp.stderr}\n",
            encoding="utf-8",
        )
        if train_cp.returncode != 0:
            print(train_cp.stderr)
            if "Impossible to load Strategy" in (train_cp.stdout + train_cp.stderr):
                raise RuntimeError(f"第 {i} 轮回测失败：Impossible to load Strategy（{class_name}）。已停止后续轮次。")
            continue
        print("正在解析回测结果……")
        train_zip = _select_backtest_zip(results_dir, train_before_zips, train_start_ts)
        train_result = parse_backtest_from_zip(train_zip, class_name)
        train_metrics = _extract_metrics(train_result)
        pair_metrics = _normalize_pair_metrics(train_metrics.get("pairs", []))
        entry_tag_metrics = _normalize_entry_tag_metrics(train_metrics.get("entry_tags", []))
        status["train_backtest_status"] = "已训练回测"
        iteration_stats["train_backtest_count"] += 1
        prev_train_trades = int(train_metrics.get("total_trades", 0) or 0)
        write_json(version_dir / "train_metrics.json", train_metrics)
        train_score = _score(train_metrics, train)
        _print_round_table(ver, train.timerange, train_metrics)
        train_trades = _safe_int(train_metrics.get("total_trades"))
        severe_trade_excess = train_trades > max_trades * 1.5
        mild_trade_excess = max_trades < train_trades <= max_trades * 1.5
        high_freq_risk = train_trades > max_trades

        val_scores = []
        validation_metrics = []
        validation_status = "not_run"
        validation_skip_reason = ""
        hard_invalid_reason: str | None = None
        min_trades_grace_floor = min_trades * min_trades_grace_ratio
        if train_trades == 0:
            hard_invalid_reason = "训练区间无交易"
            validation_skip_reason = "训练区间无交易"
        elif train_trades < min_trades_grace_floor:
            hard_invalid_reason = "训练区间交易数低于目标下限"
            validation_skip_reason = "训练区间交易数低于目标下限"
        elif train_trades < min_trades:
            trade_under_min = True
            cannot_be_official_best_unless_validation_strong = True
            trade_count_warning = "训练交易数略低于目标，但表现接近打平，建议下一轮在保持质量基础上略微增加信号。"
            print(f"训练区间交易数 {train_trades} 低于目标下限 {min_trades}，但在宽限范围内，继续验证，仅作为候选参考。")
        elif severe_trade_excess:
            hard_invalid_reason = "训练区间交易数严重超过目标上限"
            validation_skip_reason = "训练区间交易数严重超过目标上限"
        if hard_invalid_reason:
            print(f"训练区间触发硬约束：{hard_invalid_reason}，跳过所有验证区间回测。")
            _apply_train_hard_constraint_status(round_state, hard_invalid_reason, validation_skip_reason)
            validation_metrics = round_state["validation_metrics"]
            validation_status = round_state["validation_status"]
            validation_skip_reason = round_state["validation_skip_reason"]
            holdout_metrics = round_state["holdout_metrics"]
            holdout_status = round_state["holdout_status"]
            holdout_reason = round_state["holdout_reason"]
            status["validation_backtest_status"] = "跳过验证"
            iteration_stats["skipped_validation_count"] += 1
        else:
            if mild_trade_excess:
                print(f"训练区间交易数 {train_trades} 超过目标上限 {max_trades}，允许继续验证但本轮不可成为有效 best。")
            for p in validations:
                print(f"7. 正在回测验证区间：{p.timerange}")
                vcmd = train_cmd.copy()
                vcmd[vcmd.index("--timerange") + 1] = p.timerange
                val_before_zips = set(_list_backtest_zips(results_dir))
                val_start_ts = time.time()
                val_cp = run_cmd(vcmd, ROOT_DIR)
                with (version_dir / "backtest_logs.txt").open("a", encoding="utf-8") as logf:
                    logf.write(f"\n[Validation {p.name} {p.timerange}]\nSTDOUT:\n{val_cp.stdout}\n\nSTDERR:\n{val_cp.stderr}\n")
                if val_cp.returncode != 0:
                    print(val_cp.stderr)
                    continue
                print("正在解析回测结果……")
                val_zip = _select_backtest_zip(results_dir, val_before_zips, val_start_ts)
                vm = _extract_metrics(parse_backtest_from_zip(val_zip, class_name))
                validation_metrics.append({"period": p.name, "timerange": p.timerange, "metrics": vm})
                iteration_stats["validation_backtest_total_count"] += 1
                _print_round_table(ver, p.timerange, vm)
                val_scores.append(_score(vm, p))
            if validation_metrics:
                validation_status = "completed"
                status["validation_backtest_status"] = "已验证回测"
                iteration_stats["validation_backtest_count"] += 1
            else:
                validation_status = "failed"
        validation_score = sum(val_scores) / len(val_scores) if val_scores else 0.0
        write_json(
            version_dir / "validation_metrics.json",
            {
                "periods": validation_metrics,
                "average_score": validation_score,
            },
        )
        overfit_penalty = max(0.0, train_score - validation_score) * 0.3
        baseline_ok, baseline_reason, baseline_dd_penalty = _baseline_gate_and_penalty(train_metrics, runtime_goal)
        final_score, score_penalty_detail = _compute_final_score(
            train_score=train_score,
            validation_score=validation_score,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            target_cfg=target_cfg,
        )
        final_score = final_score - overfit_penalty - baseline_dd_penalty
        if hard_invalid_reason:
            final_score = 0
        if train_trades > max_trades:
            final_score = min(final_score, 0.0)
        is_overfit = train_score > validation_score * 1.3 if validation_score else True
        zero_reason = _score_zero_reason(final_score, train_metrics, validation_metrics, validation_score)
        if zero_reason:
            print(f"第 {i} 轮 final_score 为 0，原因：{zero_reason}")

        all_validation_zero = bool(validation_metrics) and all(
            _safe_int(item.get("metrics", {}).get("total_trades")) == 0 for item in validation_metrics
        )
        if _safe_int(train_metrics.get("total_trades")) == 0 and all_validation_zero:
            zero_trade_streak += 1
            previous_failure_reason = "训练区间和所有验证区间均为 0 交易，请放宽入场条件。"
        else:
            zero_trade_streak = 0

        validation_strong = _is_validation_strong(validation_metrics, baseline_cfg, target_cfg)
        is_valid, invalid_reason = _validate_round(train_metrics, validation_metrics, final_score)
        if hard_invalid_reason:
            is_valid = False
            invalid_reason = hard_invalid_reason
        elif trade_under_min and not validation_strong:
            is_valid = False
            invalid_reason = "训练区间交易数略低于目标下限，验证强度不足"
        elif trade_under_min and not allow_near_min_trades_best:
            is_valid = False
            invalid_reason = "训练区间交易数略低于目标下限，仅作为候选参考"
        elif mild_trade_excess:
            is_valid = False
            invalid_reason = "训练区间交易数超过目标上限"
        if is_valid and not baseline_ok:
            is_valid = False
            invalid_reason = baseline_reason
        if not is_valid:
            previous_failure_reason = invalid_reason
        failure_reasons = []
        if _safe_float(train_metrics.get("profit_total_pct")) < _safe_float((runtime_goal.get("baseline", {}) or {}).get("profit_total_pct")):
            failure_reasons.append("训练区间亏损超过 baseline")
        if _safe_float(train_metrics.get("profit_factor")) < _safe_float((runtime_goal.get("baseline", {}) or {}).get("profit_factor")):
            failure_reasons.append("Profit factor 低于 baseline")
        if _safe_float(train_metrics.get("max_drawdown_pct")) > _safe_float((runtime_goal.get("target", {}) or {}).get("max_drawdown_pct")):
            failure_reasons.append("最大回撤超过目标")
        if _safe_int(train_metrics.get("total_trades")) > int((runtime_goal.get("target", {}) or {}).get("max_trades", 80)) * 1.5:
            failure_reasons.append("交易数超过目标上限")
        roi_profit_abs = _safe_float(train_metrics.get("roi_profit_abs"))
        stop_loss_profit_abs = _safe_float(train_metrics.get("stop_loss_profit_abs"))
        trailing_stop_loss_profit_abs = _safe_float(train_metrics.get("trailing_stop_loss_profit_abs"))
        if abs(stop_loss_profit_abs) > max(0.0, roi_profit_abs) * 1.2:
            failure_reasons.append("固定止损亏损吞噬 ROI 收益。")
        if trailing_stop_loss_profit_abs < -20:
            failure_reasons.append("移动止盈/止损结构造成大额亏损。")
        if high_freq_risk:
            failure_reasons.append("高频风险：交易数超过目标上限 1.5 倍")
        if not failure_reasons and invalid_reason:
            failure_reasons.append(invalid_reason)
        failure_reason = "；".join(failure_reasons) if failure_reasons else ("通过" if is_valid else "综合评分不达标")
        round_data = {
            "iteration": i, "class_name": class_name, "strategy_file": str(strategy_file),
            "train_metrics": train_metrics, "train_score": train_score, "validation_score": validation_score,
            "overfit_penalty": overfit_penalty, "final_score": final_score, "is_overfit": is_overfit,
            "is_valid": is_valid, "invalid_reason": invalid_reason,
        }
        write_json(run_dir / f"round_{i:03d}.json", round_data)
        is_best = is_valid and final_score > 0 and (best is None or final_score > float(best["final_score"]))
        reason_detail = [] if is_best else _build_not_best_reason_detail(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            final_score=final_score,
            champion_meta=champion.get("meta", {}) or {},
            target_cfg=target_cfg,
            invalid_reason=invalid_reason,
        )

        status["is_valid"] = bool(is_valid)
        status["is_best"] = bool(is_best)
        status["invalid_reason"] = str(invalid_reason or "")
        status["final_score"] = float(final_score)
        if validation_metrics:
            status["validation_backtest_status"] = "已完成"
        elif "跳过" in str(validation_status or "") or _safe_int(train_metrics.get("total_trades")) == 0:
            status["validation_backtest_status"] = "跳过"
        if is_valid:
            iteration_stats["valid_strategy_count"] += 1
        else:
            iteration_stats["invalid_strategy_count"] += 1
        if session_best is None or final_score > float(session_best.get("final_score", -1e18)):
            session_best = {"version": ver, "class_name": class_name, "final_score": final_score, "is_valid": is_valid, "invalid_reason": invalid_reason}
        score_breakdown = {
            "train_score": train_score,
            "validation_score": validation_score,
            "overfit_penalty": overfit_penalty,
            "baseline_dd_penalty": baseline_dd_penalty,
            "formula": "final_score = train_score*0.4 + validation_score*0.4 + worst_validation_score*0.2 - penalties - overfit_penalty - baseline_dd_penalty",
            "penalty_detail": score_penalty_detail,
            "zero_score_reason": zero_reason,
            "baseline_check": {
                "passed": baseline_ok,
                "reason": baseline_reason,
            },
        }
        champion_metrics = champion.get("meta", {}).get("train_metrics", {}) or {}
        improvement_vs_champion = {
            "profit_total_pct_delta": _safe_float(train_metrics.get("profit_total_pct")) - _safe_float(champion_metrics.get("profit_total_pct")),
            "profit_factor_delta": _safe_float(train_metrics.get("profit_factor")) - _safe_float(champion_metrics.get("profit_factor")),
            "drawdown_pct_delta": _safe_float(train_metrics.get("max_drawdown_pct")) - _safe_float(champion_metrics.get("max_drawdown_pct")),
            "trades_delta": _safe_int(train_metrics.get("total_trades")) - _safe_int(champion_metrics.get("total_trades")),
        }
        summary = {
            "strategy_class": class_name,
            "strategy_file": str(strategy_file),
            "parent_strategy": champion.get("meta", {}).get("strategy_class", "baseline"),
            "mutation_type": mutation_type,
            "official_champion": official_champion_name,
            "historical_best": session_parent_candidates.get("historical_best"),
            "nearest_candidate_used": session_parent_candidates.get("nearest_candidate"),
            "session_parent_choice": strategy_spec.get("session_parent_choice", "baseline"),
            "session_parent_reason": strategy_spec.get("session_parent_reason", ""),
            "advisor_prompt_file": str(version_dir / "advisor_prompt.txt"),
            "codegen_prompt_file": str(version_dir / "codegen_prompt.txt"),
            "changed_items": strategy_spec.get("changes", []),
            "train_metrics": train_metrics,
            "exit_reason_details": {
                "train": {k: v for k, v in train_metrics.items() if k.startswith(("roi_", "stop_loss_", "trailing_stop_loss_", "force_exit_", "exit_signal_")) or k == "parsed"},
                "validation": [
                    {
                        "period_name": item.get("period_name"),
                        "details": {
                            k: v for k, v in (item.get("metrics", {}) or {}).items()
                            if k.startswith(("roi_", "stop_loss_", "trailing_stop_loss_", "force_exit_", "exit_signal_")) or k == "parsed"
                        },
                    }
                    for item in validation_metrics
                ],
            },
            "validation_metrics": validation_metrics,
            "validation_status": validation_status,
            "validation_skip_reason": validation_skip_reason,
            "holdout_metrics": holdout_metrics,
            "holdout_status": holdout_status,
            "holdout_reason": holdout_reason,
            "pair_metrics": pair_metrics,
            "entry_tag_metrics": entry_tag_metrics,
            "score_breakdown": score_breakdown,
            "overfit_result": {
                "is_overfit": is_overfit,
                "train_score": train_score,
                "validation_score": validation_score,
            },
            "final_score": final_score,
            "improvement_vs_champion": improvement_vs_champion,
            "failure_reason": failure_reason,
            "reason_detail": reason_detail,
            "is_best": is_best,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
            "similarity_report": read_json(version_dir / "similarity_report.json") if (version_dir / "similarity_report.json").exists() else {},
            "trade_under_min": trade_under_min,
            "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
            "validation_strong": validation_strong,
            "allow_near_min_trades_best": allow_near_min_trades_best,
            "min_trades_grace_ratio": min_trades_grace_ratio,
            "trade_count_warning": trade_count_warning,
        }
        summary_path = write_iteration_summary(summary)
        avg_validation_profit_pct, avg_validation_profit_factor, max_validation_drawdown_pct = _aggregate_validation_metrics(validation_metrics)
        leaderboard_entry = {
            "version": ver,
            "run_id": run_id,
            "strategy_class": class_name,
            "strategy_file": str(strategy_file),
            "code_hash": code_hash,
            "created_at": datetime.utcnow().isoformat(),
            "final_score": final_score,
            "train_profit_pct": _safe_float(train_metrics.get("profit_total_pct")),
            "train_profit_abs": _safe_float(train_metrics.get("profit_total_abs")),
            "avg_validation_profit_pct": avg_validation_profit_pct,
            "avg_validation_profit_factor": avg_validation_profit_factor,
            "max_validation_drawdown_pct": max_validation_drawdown_pct,
            "validation_metrics": validation_metrics,
            "trade_under_min": trade_under_min,
            "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
            "validation_strong": validation_strong,
            "trade_count_warning": trade_count_warning,
            "profit_factor": _safe_float(train_metrics.get("profit_factor")),
            "max_drawdown_pct": _safe_float(train_metrics.get("max_drawdown")) * 100.0,
            "total_trades": _safe_int(train_metrics.get("total_trades")),
            "is_overfit": is_overfit,
            "is_best": False,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
            "overfit_result": {"is_overfit": is_overfit},
            "features": features,
            "spec_hash": spec_hash,
            "failure_reason": failure_reason,
            "reason_detail": reason_detail,
            "avoid_next": "降低高频宽松入场，控制回撤与止损亏损。",
            **_extract_exit_profit_fields(train_metrics),
        }
        leaderboard.append(leaderboard_entry)
        memory_items.append({
            **leaderboard_entry,
            "validation_metrics": validation_metrics,
            "avg_validation_metrics": {
                "profit_total_pct": avg_validation_profit_pct,
                "profit_factor": avg_validation_profit_factor,
                "max_drawdown_pct": max_validation_drawdown_pct,
            },
            "train_metrics": train_metrics,
            **_extract_exit_profit_fields(train_metrics),
        })
        if (final_score <= 0 or _safe_float(train_metrics.get("max_drawdown_pct")) > _safe_float((runtime_goal.get("target", {}) or {}).get("max_drawdown_pct"))
                or avg_validation_profit_pct < _safe_float((runtime_goal.get("baseline", {}) or {}).get("profit_total_pct"))
                or _safe_int(train_metrics.get("total_trades")) > int((runtime_goal.get("target", {}) or {}).get("max_trades", 80)) * 1.5):
            if mutation_type:
                used_failed_mutations.add(mutation_type)
            blacklist_items.append({
                "code_hash": code_hash,
                "feature_signature": signature,
                "features": features,
                "failure_reason": failure_reason,
                "avoid_next": "避免重复高频宽松且亏损放大的结构。",
            })
            lessons_items.append({
                "version": ver,
                "failure_reason": failure_reason,
                "avoid_next": "减少 stoploss 损失，验证区间优先稳健。",
            })
        holdout_failed = False
        holdout_ranges = list(runtime_goal.get("holdout_ranges", []) or [])
        if is_best and holdout_ranges:
            holdout_status = "completed"
            holdout_reason = ""
            for idx, h in enumerate(holdout_ranges, start=1):
                h_label = str(h.get("label") or f"holdout_{idx:02d}")
                h_timerange = str(h.get("timerange") or "")
                if not TIMERANGE_RE.match(h_timerange):
                    continue
                hcmd = train_cmd.copy()
                hcmd[hcmd.index("--timerange") + 1] = h_timerange
                before = set(_list_backtest_zips(results_dir))
                ts = time.time()
                hcp = run_cmd(hcmd, ROOT_DIR)
                if hcp.returncode != 0:
                    continue
                hzip = _select_backtest_zip(results_dir, before, ts)
                hm = _extract_metrics(parse_backtest_from_zip(hzip, class_name))
                holdout_metrics.append({"label": h_label, "timerange": h_timerange, "metrics": hm})
                if _safe_float(hm.get("profit_total_pct")) < -2.5 or _safe_float(hm.get("profit_factor")) < 0.45 or _safe_float(hm.get("max_drawdown_pct")) > 3.0:
                    holdout_failed = True
            if holdout_failed:
                holdout_status = "failed"
                holdout_reason = "holdout 复验未通过"
                is_best = False
                is_valid = False
                invalid_reason = "holdout 复验未通过，降级为 nearest_candidate"
                status["is_best"] = False
                status["is_valid"] = False
                status["invalid_reason"] = invalid_reason
            elif not holdout_metrics:
                holdout_status = "not_run"
                holdout_reason = "未找到可执行 holdout 区间，未执行 holdout"
        elif holdout_status == "not_run" and not holdout_reason:
            holdout_reason = "未达到候选 best 条件，未执行 holdout"
        if not is_best and not reason_detail:
            reason_detail = _build_not_best_reason_detail(
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                final_score=final_score,
                champion_meta=champion.get("meta", {}) or {},
                target_cfg=target_cfg,
                invalid_reason=invalid_reason,
            )
        summary.update({
            "holdout_metrics": holdout_metrics,
            "holdout_status": holdout_status,
            "holdout_reason": holdout_reason,
            "is_best": is_best,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
            "reason_detail": reason_detail,
            "trade_under_min": trade_under_min,
            "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
            "validation_strong": validation_strong,
            "trade_count_warning": trade_count_warning,
        })
        summary_path = write_iteration_summary(summary)
        if summary_write_failed:
            status["is_valid"] = False
            status["is_best"] = False
            status["invalid_reason"] = invalid_reason
            flush_iteration_stats()
            continue
        if is_best:
            best = round_data
            iteration_stats["new_best_update_count"] += 1
            best_summary_path = summary_path
            write_json(run_dir / "best_strategy.json", best)
            shutil.copy2(strategy_file, GENERATED_DIR / f"BEST_{strategy_family}.py")
            champion = {"meta": {"strategy_class": class_name, "strategy_file": str(strategy_file), "train_metrics": train_metrics}, "code": code}
            historical_best_mem = {
                "strategy_class": class_name,
                "strategy_file": str(strategy_file),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "final_score": final_score,
            }
            print(f"本轮 {ver} 成为新 best，下一轮将以 {ver} 作为 session_parent 候选。")
        print(f"8. 第 {i} 轮完成：{'有效' if is_valid else '无效'}，原因：{invalid_reason or '通过'}")
        print(f"是否成为新最佳：{'是' if is_best else '否'}")
        if not is_best:
            print("未成为 best 原因：")
            for detail in reason_detail:
                print(f"- {detail}")
        flush_iteration_stats()
        if zero_trade_streak >= 3:
            print("连续 3 轮无交易，可能是 AI prompt 或策略模板过于保守，请检查生成策略代码。")
            break

    for row in leaderboard:
        summary_file = run_dir / str(row.get("version", "")) / "summary.json"
        usage = {}
        if summary_file.exists():
            try:
                usage = read_json(summary_file).get("ai_models_used", {}) or {}
            except Exception:
                usage = {}
        advisor_usage = usage.get("strategy_advisor", {}) or {}
        codegen_usage = usage.get("code_generator", {}) or {}
        row.setdefault("advisor_model_used", advisor_usage.get("used_model", ""))
        row.setdefault("codegen_model_used", codegen_usage.get("used_model", ""))
        row.setdefault("advisor_attempt_count", len(advisor_usage.get("attempts", []) or []))
        row.setdefault("codegen_attempt_count", len(codegen_usage.get("attempts", []) or []))

    leaderboard_sorted = sorted(leaderboard, key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)
    best_version = None
    valid_rows = [row for row in leaderboard_sorted if row.get("is_valid")]
    invalid_rows = [row for row in leaderboard_sorted if not row.get("is_valid")]
    generated_rows = [f"{row.get('version')}:{row.get('strategy_class')}" for row in leaderboard_sorted]
    print("\n===== 本次运行摘要 =====")
    print(f"运行目录：{run_dir}")
    print("生成策略：" + ("、".join(generated_rows) if generated_rows else "无"))
    print("有效策略：" + ("、".join(f"{row['version']}:{row['strategy_class']}" for row in valid_rows) if valid_rows else "无"))
    print("无效策略：" + ("、".join(f"{row['version']}:{row['strategy_class']}({row.get('invalid_reason')})" for row in invalid_rows) if invalid_rows else "无"))
    print(f"当前最佳策略是否更新：{'是' if best else '否'}")
    target_cfg = (runtime_goal.get("target", {}) or {})
    baseline_cfg = (runtime_goal.get("baseline", {}) or {})
    max_drawdown_target = _safe_float(target_cfg.get("max_drawdown_pct", 3.0))
    max_trades_target = _safe_int(target_cfg.get("max_trades", 80))
    baseline_profit_pct = _safe_float(baseline_cfg.get("profit_total_pct"))

    def _is_eligible_nearest(row: dict[str, Any], allow_oversized: bool) -> bool:
        total_trades = _safe_int(row.get("total_trades"))
        if total_trades <= 0:
            return False
        min_trades_target = _safe_int(target_cfg.get("min_trades", 25))
        if total_trades < min_trades_target and not (row.get("trade_under_min") and row.get("validation_strong")):
            return False
        if total_trades > int(max_trades_target * 1.5):
            return False
        if total_trades > max_trades_target and not allow_oversized:
            return False
        return not row.get("is_valid")

    def _nearest_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        profit_pct = _safe_float(row.get("train_profit_pct"))
        drawdown_pct = _safe_float(row.get("max_drawdown_pct"))
        profit_factor = _safe_float(row.get("profit_factor"))
        total_trades = _safe_int(row.get("total_trades"))

        profit_gap = abs(profit_pct - baseline_profit_pct)
        drawdown_penalty = max(0.0, drawdown_pct - max_drawdown_target)
        min_trades_target = _safe_int(target_cfg.get("min_trades", 25))
        trade_penalty = max(0, total_trades - max_trades_target) + max(0, min_trades_target - total_trades)
        return (profit_gap, drawdown_penalty, -profit_factor, float(trade_penalty))

    nearest_failed_candidates = [row for row in leaderboard_sorted if _is_eligible_nearest(row, allow_oversized=False)]
    oversized_fallback = False
    if not nearest_failed_candidates:
        nearest_failed_candidates = [row for row in leaderboard_sorted if _is_eligible_nearest(row, allow_oversized=True)]
        oversized_fallback = True
    closest_failed = sorted(nearest_failed_candidates, key=_nearest_sort_key)[0] if nearest_failed_candidates else None
    nearest_candidate = None
    if closest_failed:
        nearest_validation_metrics: list[dict[str, Any]] = []
        nearest_summary = run_dir / str(closest_failed.get("version")) / "summary.json"
        if nearest_summary.exists():
            sum_data = read_json(nearest_summary)
            nearest_validation_metrics = [_normalize_validation_metric(x, "validation") for x in (sum_data.get("validation_metrics") or [])]
        nearest_candidate = {
            "strategy_class": closest_failed.get("strategy_class"),
            "strategy_file": closest_failed.get("strategy_file"),
            "why_nearest": _build_why_nearest(closest_failed, target_cfg, champion.get("meta", {}) or historical_best_mem),
            "train_metrics": {
                "profit_total_abs": closest_failed.get("train_profit_abs"),
                "profit_total_pct": closest_failed.get("train_profit_pct"),
                "profit_factor": closest_failed.get("profit_factor"),
                "max_drawdown_pct": closest_failed.get("max_drawdown_pct"),
                "total_trades": closest_failed.get("total_trades"),
                "roi_profit_abs": closest_failed.get("roi_profit_abs"),
                "stop_loss_profit_abs": closest_failed.get("stop_loss_profit_abs"),
            },
            "validation_metrics": nearest_validation_metrics,
            "trade_under_min": bool(closest_failed.get("trade_under_min", False)),
            "validation_strong": bool(closest_failed.get("validation_strong", False)),
            "trade_count_warning": closest_failed.get("trade_count_warning", ""),
            "improvement_vs_baseline": {
                "profit_total_pct_delta": _safe_float(closest_failed.get("train_profit_pct")) - baseline_profit_pct,
                "profit_factor_delta": _safe_float(closest_failed.get("profit_factor")) - _safe_float(baseline_cfg.get("profit_factor")),
            },
        }
        if not nearest_validation_metrics:
            nearest_candidate["validation_metrics_missing_reason"] = "训练区间触发硬约束，跳过验证"
        if bool(closest_failed.get("trade_under_min", False)):
            nearest_candidate["trade_under_min"] = True
            nearest_candidate.setdefault("why_nearest", []).append("训练交易数略低于目标，但验证表现较强，仅作候选参考")
        if oversized_fallback and _safe_int(closest_failed.get("total_trades")) > max_trades_target:
            nearest_candidate["trade_over_limit"] = True
            nearest_candidate.setdefault("why_nearest", []).append("本轮无目标交易数范围候选，使用轻度超标候选仅作参考")
        write_json(NEAREST_CANDIDATE_FILE, nearest_candidate)

    historical_best = read_json(BEST_STRATEGY_FILE) if BEST_STRATEGY_FILE.exists() else None
    current_best_saved = None
    if best:
        best_version = f"v{int(best['iteration']):03d}"
    for row in leaderboard_sorted:
        row["is_best"] = row["version"] == best_version
    write_json(run_dir / "leaderboard.json", {"items": leaderboard_sorted})
    _write_json_list_file(MEMORY_FILE, memory_items[-200:])
    _write_json_list_file(BLACKLIST_FILE, blacklist_items[-200:])
    _write_json_list_file(LESSONS_FILE, lessons_items[-200:])

    best_pair_metrics: list[dict[str, Any]] = []
    best_entry_tag_metrics: list[dict[str, Any]] = []
    if best_version:
        best_sum = run_dir / best_version / "summary.json"
        if best_sum.exists():
            bsd = read_json(best_sum)
            best_pair_metrics = list(bsd.get("pair_metrics", []) or [])
            best_entry_tag_metrics = list(bsd.get("entry_tag_metrics", []) or [])
    trade_count_warning = next((str(r.get("trade_count_warning")) for r in leaderboard_sorted if r.get("trade_count_warning")), "")
    common_failure_patterns = _build_common_failure_patterns(invalid_rows, target_cfg)
    nearest_advisor_notes = _build_nearest_advisor_notes(nearest_candidate)
    last_run_summary = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "target": target_cfg,
        "official_best": current_best_saved if 'current_best_saved' in locals() else None,
        "historical_best": historical_best_mem,
        "nearest_candidate": nearest_candidate,
        "session_best": session_best,
        "trade_count_warning": trade_count_warning,
        "failed_versions": [r.get("version") for r in invalid_rows],
        "common_failure_patterns": common_failure_patterns,
        "nearest_advisor_notes": nearest_advisor_notes,
        "recommended_next_mutation_types": ["add_entry_filter", "tighten_entry_trigger", "remove_bad_entry_condition", "pair_specific_filter", "tag_specific_filter"],
        "forbidden_next_mutation_types": sorted(used_failed_mutations),
        "worst_pairs": sorted(best_pair_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")))[:5],
        "best_pairs": sorted(best_pair_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")), reverse=True)[:5],
        "worst_entry_tags": sorted(best_entry_tag_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")))[:5],
        "best_entry_tags": sorted(best_entry_tag_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")), reverse=True)[:5],
        "lessons_for_next_run": [
            "不要重新生成完全不同策略",
            "优先围绕 nearest_candidate 和 historical_best 做单点小步调整",
            "目标总交易数是 25~80，不是单币种 25~80",
            "如果 nearest_candidate 交易数略超标，例如 98 笔，下一轮目标是压到 60~80 笔",
            "不要放宽入场",
            "不要启用 exit_signal",
            "不要启用或扩大 trailing",
            "不要扩大 stoploss",
            "优先减少固定止损亏损",
            "优先削减验证区间持续拖累的 pair / entry_tag，而不是扩大止损",
            "避免与历史失败策略相似",
        ],
    }
    write_json(LAST_RUN_SUMMARY_FILE, last_run_summary)

    if best:
        best_strategy_file = run_dir / best_version / "strategy.py" if best_version else Path(best["strategy_file"])
        print("自动优化完成")
        print(f"- 最佳策略: {best['class_name']}")
        print(f"- 最佳得分: {best['final_score']:.4f}")
        print(f"- 最佳策略文件路径: {best_strategy_file}")
        print(f"- leaderboard.json 路径: {run_dir / 'leaderboard.json'}")
        if best_summary_path:
            print(f"- summary.json 路径: {best_summary_path}")
        print("========== 最佳策略简述 ==========")
        print(f"策略名：{best['class_name']}")
        print(f"策略文件：{best_strategy_file}")
        print(f"训练区间收益：{_safe_float(best['train_metrics'].get('profit_total_pct')):.2f}%")
        avg_val_profit = next((r.get('avg_validation_profit_pct', 0.0) for r in leaderboard_sorted if r.get('version') == best_version), 0.0)
        print(f"验证区间平均收益：{_safe_float(avg_val_profit):.2f}%")
        print(f"交易数：{_safe_int(best['train_metrics'].get('total_trades'))}")
        print(f"胜率：{_safe_float(best['train_metrics'].get('winrate')) * 100:.2f}%")
        print(f"Profit Factor：{_safe_float(best['train_metrics'].get('profit_factor')):.4f}")
        print(f"最大回撤：{_safe_float(best['train_metrics'].get('max_drawdown')) * 100:.2f}%")
        print(f"是否疑似过拟合：{'是' if best.get('is_overfit') else '否'}")
        print("主要优势：训练与验证综合评分最高。")
        print("主要风险：仍需更多样本周期验证稳健性。")
        print("为什么成为 best：在满足有效性约束下 final_score 最高。")
        best_leaderboard_entry = next((r for r in leaderboard_sorted if r.get("version") == best_version), {}) or {}
        best_validation_metrics = best_leaderboard_entry.get("validation_metrics", []) or []
        cb_avg_profit, cb_avg_pf, cb_max_dd = _aggregate_validation_metrics(best_validation_metrics)
        current_best_saved = {"strategy_class": best["class_name"], "strategy_file": str(best_strategy_file), "source_run_id": run_id, "version": best_version, "train_metrics": best["train_metrics"], "validation_metrics": best_validation_metrics, "avg_validation_metrics": {"profit_total_pct": cb_avg_profit, "profit_factor": cb_avg_pf, "max_drawdown_pct": cb_max_dd}, "score_breakdown": {}, "final_score": best["final_score"], "created_at": datetime.utcnow().isoformat(), "why_best": "本轮 final_score 最高。", "is_overfit": bool(best.get("is_overfit"))}
        write_json(BEST_STRATEGY_FILE, current_best_saved)
    else:
        if args.force_session_best and session_best:
            write_json(run_dir / "session_best.json", session_best)
        print("本次没有找到有效新策略，当前最佳策略保持不变。")
        print("本轮失败策略共同原因：")
        if common_failure_patterns:
            for pattern in common_failure_patterns:
                print(f"- {pattern}")
        else:
            print("- 暂无可归纳的共同失败模式")
        print("下一轮建议：")
        print("- 降低目标交易数到 25~80")
        print("- 不要继续宽松高频入场")
        print("- 优先控制止损损失")
        if reset_best_used:
            print("已初始化历史最佳策略，但本轮没有产生有效新 best。")
            print("正式 best 暂为空，nearest_candidate 已保存。")
    print("\n========== 结束状态 ==========")
    if current_best_saved:
        print(f"当前正式 best：来源=本轮新 best；策略名={current_best_saved.get('strategy_class')}；final_score={_safe_float(current_best_saved.get('final_score')):.4f}")
    elif historical_best and historical_best.get("strategy_class") and historical_best.get("source") != "reset":
        print(f"当前正式 best：来源=历史 best；策略名={historical_best.get('strategy_class')}；final_score={_safe_float(historical_best.get('final_score')):.4f}")
    else:
        print("当前正式 best：来源=无")
    if session_best:
        reason = session_best.get("invalid_reason") or ("通过" if session_best.get("is_valid") else "未通过有效性约束")
        print(f"本轮 session best：策略名={session_best.get('class_name')}；final_score={_safe_float(session_best.get('final_score')):.4f}；未成为正式 best 原因：{reason}")
    if closest_failed:
        print("========== 本轮最接近目标的失败策略 ==========")
        print(f"版本：{closest_failed.get('version')}")
        print(f"策略：{closest_failed.get('strategy_class')}")
        print(f"final_score：{_safe_float(closest_failed.get('final_score')):.4f}")
        print(f"失败原因：{closest_failed.get('failure_reason') or closest_failed.get('invalid_reason') or '综合评分不达标'}")
        print("为什么仍未成为 best：未通过有效性约束或综合得分不足。")
    if not best:
        print("========== 冠军-挑战者状态 ==========")
        print(f"当前最佳策略 champion：{champion.get('meta', {}).get('strategy_class', 'baseline')}")
        if nearest_candidate:
            print(f"本轮最接近目标 challenger：{nearest_candidate.get('strategy_class')}")
            print(f"比 baseline 差异：{nearest_candidate.get('improvement_vs_baseline')}")
            print("下一轮应该怎么改：优先选择未失败过的 mutation_type，继续单点小步调整。")
        else:
            print("本轮没有可参考 challenger。")
        print("本轮失败模式总结：高频风险 / 止损吞噬利润 / trailing 结构失败 / 相似失败策略。")

    print("========== 记忆写入状态 ==========")
    print(f"nearest_candidate.json：{'已写入' if nearest_candidate else '未写入'}")
    print(f"last_run_summary.json：{'已写入' if LAST_RUN_SUMMARY_FILE.exists() else '未写入'}")
    print(f"best_strategy.json：{'已更新' if best else '未更新'}")
    print("strategy_memory.json：已追加")
    print("strategy_lessons.json：已更新")
    print("strategy_blacklist.json：已更新")
    print(f"下一轮 advisor prompt 将加载 last_run_summary：{'是' if LAST_RUN_SUMMARY_FILE.exists() else '否'}")
    print(f"下一轮 advisor prompt 将加载 nearest_candidate：{'是' if NEAREST_CANDIDATE_FILE.exists() else '否'}")
    print(f"下一轮 advisor prompt 将加载 historical_best：{'是' if BEST_STRATEGY_FILE.exists() else '否'}")
    flush_iteration_stats()
    print("\n========== 本次迭代统计 ==========")
    print(f"计划迭代轮数：{iteration_stats.get('planned_iterations')}")
    print(f"策略顾问成功次数：{iteration_stats.get('advisor_success_count')}")
    print(f"代码生成成功次数：{iteration_stats.get('codegen_success_count')}")
    print(f"实际生成策略版本数：{iteration_stats.get('generated_versions_count')}")
    print(f"训练回测版本数：{iteration_stats.get('train_backtest_count')}")
    print(f"验证回测版本数：{iteration_stats.get('validation_backtest_count')}")
    print(f"验证区间回测总次数：{iteration_stats.get('validation_backtest_total_count')}")
    print(f"跳过验证版本数：{iteration_stats.get('skipped_validation_count')}")
    print(f"有效策略数：{iteration_stats.get('valid_strategy_count')}")
    print(f"无效策略数：{iteration_stats.get('invalid_strategy_count')}")
    print(f"新 best 更新次数：{iteration_stats.get('new_best_update_count')}")
    print(f"当前策略迭代版本：{iteration_stats.get('current_iteration_version')}")
    print(f"累计历史策略版本数：{iteration_stats.get('history_strategy_total_count')}")
    print(f"统计文件：{iteration_stats_path}")
    print("详细状态：")
    for row in version_statuses:
        print(
            f"{row.get('version')}：顾问={row.get('advisor_status')} / 代码={row.get('codegen_status')} / 语法={row.get('syntax_check_status')} / 静态={row.get('static_check_status')} / 训练={row.get('train_backtest_status')} / 验证={row.get('validation_backtest_status')} / 有效={'是' if row.get('is_valid') else '否'} / 新best={'是' if row.get('is_best') else '否'} / 原因={row.get('invalid_reason') or '通过'}"
        )
    print("\n========== Prompt 审计文件 ==========")
    if getattr(args, "print_prompt_files", False):
        for p in sorted(run_dir.glob('v*/advisor_prompt.txt')):
            print(p)
        for p in sorted(run_dir.glob('v*/codegen_prompt.txt')):
            print(p)
    else:
        print(f'Prompt 审计文件：已保存到各版本目录，如需查看请执行：find {run_dir} -maxdepth 3 -type f \\( -name "*prompt*" \\) -print')

    if args.retest_current_best_at_end:
        print("\n========== 结束前复测 current best / baseline ==========")
        train_timerange = train.timerange
        candidate = current_best_saved or historical_best
        if candidate and candidate.get("strategy_class") and candidate.get("strategy_class") != "baseline":
            cls = str(candidate.get("strategy_class"))
            print(f"复测 current best: {cls}")
            try:
                metrics = run_single_backtest_metrics(config, cls, timeframe, train_timerange)
                _print_round_table("retest", train_timerange, metrics)
            except RuntimeError as exc:
                print(f"current best 复测失败：{exc}")
        else:
            print("current best 不可复测（仅 baseline 或缺少策略类名）。")
        baseline_cfg = runtime_goal.get("baseline", {}) or {}
        print("baseline 完整指标：")
        _print_round_table("baseline", train_timerange, baseline_cfg)
    if args.print_current_best:
        print_current_best_summary(runtime_goal, historical_best, current_best_saved if best else None)

    if hasattr(args, "_run_start_ts"):
        elapsed = time.time() - float(getattr(args, "_run_start_ts"))
        print("\n========== 本次运行总耗时 ==========")
        print(f"总耗时：{_format_elapsed_seconds(elapsed)}（{elapsed:.1f} 秒）")
    print_log_saved_summary(args)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="ai_tools/optimization_goal.json")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ai-timeout", type=int, default=int(os.getenv("AI_MODEL_TIMEOUT_SECONDS", "180")), help="AI API 调用超时时间（秒）")
    parser.add_argument("--ai-max-retries", type=int, default=int(os.getenv("AI_MAX_RETRIES", "5")), help="兼容旧参数；模型池轮换使用 --ai-model-max-attempts-per-call")
    parser.add_argument("--ai-model-max-attempts-per-call", type=int, default=int(os.getenv("AI_MODEL_MAX_ATTEMPTS_PER_CALL", "5")), help="单次 AI 调用最大模型池尝试次数")
    parser.add_argument("--ai-model-switch-on-error", dest="ai_model_switch_on_error", action="store_true", default=parse_yes_no(os.getenv("AI_MODEL_SWITCH_ON_ERROR", "true")) is not False, help="AI 调用失败时自动切换同角色模型池中的下一个模型")
    parser.add_argument("--no-ai-model-switch-on-error", dest="ai_model_switch_on_error", action="store_false", help="关闭 AI 模型池失败自动轮换")
    parser.add_argument("--ai-call-cooldown-seconds", type=float, default=float(os.getenv("AI_CALL_COOLDOWN_SECONDS", "10")), help="每次 AI 调用后到下一次调用前最小间隔秒数")
    parser.add_argument("--advisor-to-codegen-delay-seconds", type=float, default=float(os.getenv("ADVISOR_TO_CODEGEN_DELAY_SECONDS", "5")), help="策略顾问成功后到代码生成前额外等待秒数")
    parser.add_argument("--no-wizard", action="store_true")
    parser.add_argument("--save-goal", action="store_true")
    parser.add_argument("--config", default="user_data/config.5coins.json")
    parser.add_argument("--base-strategy", default="MultiCoin_AI_Strategy")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timerange", default=None, help="训练回测区间，例如 20260501-20260525")
    parser.add_argument("--print-current-best", dest="print_current_best", action="store_true", default=True)
    parser.add_argument("--no-print-current-best", dest="print_current_best", action="store_false")
    parser.add_argument("--retest-current-best-at-end", action="store_true", default=False)
    parser.add_argument("--reset-best", action="store_true", default=False)
    parser.add_argument("--force-session-best", action="store_true", default=False)
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--auto-continue-if-similar-to-parent", action="store_true", default=True)
    parser.add_argument("--auto-reject-failed-similarity", action="store_true", default=False)
    parser.add_argument("--allow-near-min-trades-best", action="store_true", default=False, help="允许训练交易数处于 min_trades 宽限范围且验证强劲的策略成为正式 best")
    parser.add_argument("--print-prompt-files", action="store_true", default=False, help="运行结束时打印完整 Prompt 审计文件列表")
    parser.add_argument("--no-log-file", action="store_true", default=False, help="不保存本次终端完整日志，仅输出到终端")
    parser.add_argument("--log-dir", default="user_data/logs", help="全局终端日志目录，默认 user_data/logs")
    args = parser.parse_args()

    run_dir = RESULT_ROOT / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args._run_start_ts = time.time()
    log_ctx = setup_terminal_logging(run_dir, args)
    args._log_context = log_ctx
    try:
        print_log_start_banner(log_ctx)
        goal_path = ROOT_DIR / args.goal
        ensure_goal_file(goal_path)
        goal = read_json(goal_path)
        goal.setdefault("language", "zh-CN")

        runtime_goal = goal if args.no_wizard else run_wizard(goal, args)
        if runtime_goal.get("runtime_auto_approve", False):
            args.auto_approve = True
        if runtime_goal.get("runtime_force_download", False):
            args.force_download = True
        if runtime_goal.get("runtime_reset_best", False):
            args.reset_best = True

        maybe_reset_best_strategy(args.reset_best)

        effective_iterations = args.iterations if args.iterations is not None else int(runtime_goal.get("max_iterations", 5))
        runtime_goal["max_iterations"] = int(effective_iterations)
        print(f"本次实际迭代轮数：{int(effective_iterations)}")

        if args.save_goal:
            write_json(goal_path, runtime_goal)
            print(f"已保存修改后的目标配置到：{goal_path}")

        write_json(run_dir / "goal.runtime.json", runtime_goal)
        write_json(run_dir / "goal.json", goal)
        print(f"运行时配置已保存：{run_dir / 'goal.runtime.json'}")

        run_auto_optimization(runtime_goal, args, run_dir)
    except BaseException:
        if log_ctx is not None:
            print("\n========== 异常 traceback ==========")
        traceback.print_exc()
        raise
    finally:
        if log_ctx is not None:
            try:
                _update_latest_log(log_ctx.latest_log_path, log_ctx.global_log_path)
            except OSError as exc:
                print(f"更新 latest.log 失败：{exc}", file=sys.stderr)
        restore_terminal_logging(log_ctx)


if __name__ == "__main__":
    main()
