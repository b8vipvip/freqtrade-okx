# -*- coding: utf-8 -*-
"""自动化 Freqtrade 策略优化脚本（中文交互向导 + 防过拟合 + 多区间验证）。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT_DIR / "user_data" / "backtest_results" / "ai_optimization_runs"
GENERATED_DIR = ROOT_DIR / "user_data" / "strategies" / "generated"
STRATEGY_DIR = ROOT_DIR / "user_data" / "strategies"
TIMERANGE_RE = re.compile(r"^\d{8}-\d{8}$")


@dataclass
class PeriodDef:
    name: str
    timerange: str
    weight: float
    kind: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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

    default_iter = args.iterations if args.iterations else int(runtime.get("max_iterations", 5))
    runtime["max_iterations"] = ask_int("最大迭代轮数", int(default_iter))

    auto_default = bool(args.auto_approve)
    runtime["runtime_auto_approve"] = ask_bool("是否全自动不中途确认", auto_default)

    runtime.setdefault("target", {})
    runtime["target"]["min_profit_total_pct"] = ask_float("目标最低收益率", float(runtime["target"].get("min_profit_total_pct", 0)))
    runtime["target"]["max_drawdown_pct"] = ask_float("最大允许回撤(%)", float(runtime["target"].get("max_drawdown_pct", 3)))
    runtime["target"]["min_profit_factor"] = ask_float("最低 Profit factor", float(runtime["target"].get("min_profit_factor", 1.0)))
    runtime["target"]["min_trades"] = ask_int("目标最小交易数", int(runtime["target"].get("min_trades", 80)))
    runtime["target"]["max_trades"] = ask_int("目标最大交易数", int(runtime["target"].get("max_trades", 200)))

    runtime.setdefault("overfit_guard", {})
    runtime["overfit_guard"]["enabled"] = ask_bool("是否启用防过拟合", bool(runtime["overfit_guard"].get("enabled", True)))

    print("提示：当前项目历史回测中 exit_signal 曾造成大量亏损，建议保持关闭。")
    runtime["target"]["prefer_exit_signal"] = ask_bool("是否允许 exit_signal", bool(runtime["target"].get("prefer_exit_signal", False)))

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


def latest_backtest_zip(results_dir: Path) -> Path:
    zips = sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError("未找到 backtest-result-*.zip")
    return zips[0]


def parse_backtest_from_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith('.json') and not n.endswith('.meta.json') and not n.endswith('_config.json')]
        with zf.open(names[0]) as fp:
            return json.load(fp)


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


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
    roi_profit = _safe_float(metrics.get("roi_profit_total"))
    stop_loss_abs = _safe_float(metrics.get("stop_loss_abs"))
    print("版本 | 区间 | 交易数 | 收益率 | 收益USDT | 胜率 | PF | 最大回撤 | ROI收益 | 止损亏损")
    print(
        f"{version} | {interval} | {trades} | {_format_pct(profit_pct)} | {profit_abs:.4f} | "
        f"{_format_pct(winrate)} | {pf:.4f} | {_format_pct(max_dd)} | {_format_pct(roi_profit)} | {stop_loss_abs:.4f}"
    )


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


def ask_ai(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    res = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return (res.choices[0].message.content or "").strip()


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


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    s = data.get("strategy", {})
    first = next(iter(s.values()), {}) if isinstance(s, dict) else {}
    t = first.get("total", {}) if isinstance(first, dict) else {}
    results_per_pair = first.get("results_per_pair", [])
    return {
        "total_trades": int(t.get("total_trades", 0) or 0),
        "profit_total_abs": float(t.get("profit_total_abs", 0) or 0),
        "profit_total_pct": float(t.get("profit_total_pct", 0) or 0),
        "profit_factor": float(t.get("profit_factor", 0) or 0),
        "max_drawdown": float(t.get("max_drawdown_account", t.get("max_drawdown", 0)) or 0),
        "winrate": float(t.get("winrate", 0) or 0),
        "roi_profit_total": float(t.get("profit_total_pct", 0) or 0),
        "stop_loss_abs": float(t.get("stop_loss_abs", 0) or 0),
        "pairs": results_per_pair,
    }


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

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY，请检查 .env。")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=api_key, base_url=base_url)

    best: dict[str, Any] | None = None
    leaderboard: list[dict[str, Any]] = []
    best_summary_path: Path | None = None
    for i in range(1, iterations + 1):
        ver = f"v{i:03d}"
        class_name = f"{strategy_family}_{ver}"
        strategy_file = STRATEGY_DIR / f"{class_name}.py"
        version_dir = run_dir / ver
        version_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在生成第 {i} 版策略……")
        prompt = (
            f"请生成完整 freqtrade 策略代码，只输出 Python 代码。类名必须为 {class_name}，继承 IStrategy，"
            f"timeframe='{timeframe}'，并实现 populate_indicators/populate_entry_trend/populate_exit_trend。"
        )
        code = extract_python_code(ask_ai(client, model, [{"role": "user", "content": prompt}]))
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        strategy_file.write_text(code, encoding="utf-8")
        shutil.copy2(strategy_file, version_dir / "strategy.py")
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(strategy_file, GENERATED_DIR / strategy_file.name)

        print("正在检查 Python 语法……")
        pyc = run_cmd([sys.executable, "-m", "py_compile", str(strategy_file)], ROOT_DIR)
        if pyc.returncode != 0:
            print(pyc.stderr)
            print(f"第 {i} 轮语法检查失败，跳过。")
            continue
        validate_strategy_class_name(strategy_file, class_name)

        print(f"正在回测训练区间：{train.timerange}")
        train_cmd = [
            "docker", "compose", "run", "--rm", "freqtrade", "backtesting",
            "--config", config, "--strategy", class_name, "--timeframe", timeframe,
            "--timerange", train.timerange, "--export", "trades", "--cache", "none",
        ]
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
        train_metrics = _extract_metrics(parse_backtest_from_zip(latest_backtest_zip(ROOT_DIR / "user_data" / "backtest_results")))
        write_json(version_dir / "train_metrics.json", train_metrics)
        train_score = _score(train_metrics, train)
        _print_round_table(ver, train.timerange, train_metrics)

        val_scores = []
        validation_metrics: list[dict[str, Any]] = []
        for p in validations:
            print(f"正在回测验证区间：{p.timerange}")
            vcmd = train_cmd.copy()
            vcmd[vcmd.index("--timerange") + 1] = p.timerange
            val_cp = run_cmd(vcmd, ROOT_DIR)
            with (version_dir / "backtest_logs.txt").open("a", encoding="utf-8") as logf:
                logf.write(f"\n[Validation {p.name} {p.timerange}]\nSTDOUT:\n{val_cp.stdout}\n\nSTDERR:\n{val_cp.stderr}\n")
            if val_cp.returncode != 0:
                print(val_cp.stderr)
                continue
            print("正在解析回测结果……")
            vm = _extract_metrics(parse_backtest_from_zip(latest_backtest_zip(ROOT_DIR / "user_data" / "backtest_results")))
            validation_metrics.append({"period": p.name, "timerange": p.timerange, "metrics": vm})
            _print_round_table(ver, p.timerange, vm)
            val_scores.append(_score(vm, p))
        validation_score = sum(val_scores) / len(val_scores) if val_scores else 0.0
        write_json(
            version_dir / "validation_metrics.json",
            {
                "periods": validation_metrics,
                "average_score": validation_score,
            },
        )
        overfit_penalty = max(0.0, train_score - validation_score) * 0.3
        final_score = train_score * 0.6 + validation_score * 0.4 - overfit_penalty
        is_overfit = train_score > validation_score * 1.3 if validation_score else True
        zero_reason = _score_zero_reason(final_score, train_metrics, validation_metrics, validation_score)
        if zero_reason:
            print(f"第 {i} 轮 final_score 为 0，原因：{zero_reason}")

        round_data = {
            "iteration": i, "class_name": class_name, "strategy_file": str(strategy_file),
            "train_metrics": train_metrics, "train_score": train_score, "validation_score": validation_score,
            "overfit_penalty": overfit_penalty, "final_score": final_score, "is_overfit": is_overfit,
        }
        write_json(run_dir / f"round_{i:03d}.json", round_data)
        is_best = best is None or final_score > float(best["final_score"])
        score_breakdown = {
            "train_score": train_score,
            "validation_score": validation_score,
            "overfit_penalty": overfit_penalty,
            "formula": "final_score = train_score*0.6 + validation_score*0.4 - overfit_penalty",
            "zero_score_reason": zero_reason,
        }
        summary = {
            "strategy_class": class_name,
            "strategy_file": str(strategy_file),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "score_breakdown": score_breakdown,
            "overfit_result": {
                "is_overfit": is_overfit,
                "train_score": train_score,
                "validation_score": validation_score,
            },
            "final_score": final_score,
            "is_best": is_best,
        }
        summary_path = version_dir / "summary.json"
        write_json(summary_path, summary)
        avg_validation_profit_pct = (
            sum(_safe_float(item["metrics"].get("profit_total_pct")) for item in validation_metrics) / len(validation_metrics)
            if validation_metrics else 0.0
        )
        leaderboard_entry = {
            "version": ver,
            "strategy_class": class_name,
            "final_score": final_score,
            "train_profit_pct": _safe_float(train_metrics.get("profit_total_pct")),
            "avg_validation_profit_pct": avg_validation_profit_pct,
            "profit_factor": _safe_float(train_metrics.get("profit_factor")),
            "max_drawdown_pct": _safe_float(train_metrics.get("max_drawdown")) * 100.0,
            "total_trades": _safe_int(train_metrics.get("total_trades")),
            "is_overfit": is_overfit,
            "is_best": False,
        }
        leaderboard.append(leaderboard_entry)
        if is_best:
            best = round_data
            best_summary_path = summary_path
            write_json(run_dir / "best_strategy.json", best)
            shutil.copy2(strategy_file, GENERATED_DIR / f"BEST_{strategy_family}.py")
        print(f"第 {i} 轮完成")
        print(f"是否成为新最佳：{'是' if is_best else '否'}")

    leaderboard_sorted = sorted(leaderboard, key=lambda x: float(x["final_score"]), reverse=True)
    best_version = None
    if best:
        best_version = f"v{int(best['iteration']):03d}"
    for row in leaderboard_sorted:
        row["is_best"] = row["version"] == best_version
    write_json(run_dir / "leaderboard.json", {"items": leaderboard_sorted})

    if best:
        best_strategy_file = run_dir / best_version / "strategy.py" if best_version else Path(best["strategy_file"])
        print("自动优化完成")
        print(f"- 最佳策略: {best['class_name']}")
        print(f"- 最佳得分: {best['final_score']:.4f}")
        print(f"- 最佳策略文件路径: {best_strategy_file}")
        print(f"- leaderboard.json 路径: {run_dir / 'leaderboard.json'}")
        if best_summary_path:
            print(f"- summary.json 路径: {best_summary_path}")
    else:
        print("自动优化结束：没有可用策略通过流程。")


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="ai_tools/optimization_goal.json")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-wizard", action="store_true")
    parser.add_argument("--save-goal", action="store_true")
    parser.add_argument("--config", default="user_data/config.5coins.json")
    parser.add_argument("--base-strategy", default="MultiCoin_AI_Strategy")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--timerange", default=None, help="训练回测区间，例如 20260501-20260525")
    args = parser.parse_args()

    goal_path = ROOT_DIR / args.goal
    ensure_goal_file(goal_path)
    goal = read_json(goal_path)
    goal.setdefault("language", "zh-CN")

    runtime_goal = goal if args.no_wizard else run_wizard(goal, args)
    if runtime_goal.get("runtime_auto_approve", False):
        args.auto_approve = True
    if runtime_goal.get("runtime_force_download", False):
        args.force_download = True

    if args.save_goal:
        write_json(goal_path, runtime_goal)
        print(f"已保存修改后的目标配置到：{goal_path}")

    run_dir = RESULT_ROOT / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "goal.runtime.json", runtime_goal)
    write_json(run_dir / "goal.json", goal)
    print(f"运行时配置已保存：{run_dir / 'goal.runtime.json'}")

    run_auto_optimization(runtime_goal, args, run_dir)


if __name__ == "__main__":
    main()
