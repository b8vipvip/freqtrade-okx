# -*- coding: utf-8 -*-
"""自动化 Freqtrade 策略优化脚本（中文交互向导 + 防过拟合 + 多区间验证）。"""

from __future__ import annotations

import argparse
import ast
import difflib
import fcntl
import hashlib
import json
import os
import random
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
RECOMMENDED_PAIRS_FILE = ROOT_DIR / "user_data" / "ai_memory" / "recommended_pairs.json"
PAIR_LEADERBOARD_FILE = ROOT_DIR / "user_data" / "ai_memory" / "pair_leaderboard.json"
ITERATION_STATS_FILE_NAME = "iteration_stats.json"
MEMORY_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_memory.example.json"
BLACKLIST_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_blacklist.example.json"
LESSONS_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "strategy_lessons.example.json"
MODEL_CONFIG_FILE = ROOT_DIR / "ai_tools" / "model_config.json"
MODEL_CONFIG_EXAMPLE_FILE = ROOT_DIR / "ai_tools" / "model_config.example.json"
TIMERANGE_RE = re.compile(r"^\d{8}-\d{8}$")
AUTO_OPTIMIZE_LOCK_FILE = Path("/tmp/freqtrade_ai_auto_optimize.lock")


def load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


load_dotenv_file(ROOT_DIR / ".env")


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


@dataclass
class LogRepoPushResult:
    enabled: bool
    repo_path: Path | None = None
    branch: str = "main"
    rel_log_dir: Path | None = None
    status: str = "未启用"
    github_path: str = ""
    error: str = ""


SENSITIVE_ENV_ASSIGN_RE = re.compile(
    r"(?im)\b(OPENAI_API_KEY|CLAUDE_API_KEY|OKX_API_KEY|OKX_API_SECRET|OKX_API_PASSPHRASE)\s*=\s*(?:[^\s'\"]+|'[^']*'|\"[^\"]*\")"
)
SENSITIVE_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b")
SENSITIVE_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]+")
SENSITIVE_FIELD_RE = re.compile(
    r"(?i)([\"']?(?:password|token|api_key|api_secret)[\"']?\s*[:=]\s*)([\"']?)([^\"'\s,}]+)([\"']?)"
)


def _env_bool(name: str, default: bool) -> bool:
    parsed = parse_yes_no(os.getenv(name, ""))
    return default if parsed is None else parsed


def _sanitize_log_text(text: str) -> str:
    text = SENSITIVE_ENV_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    text = SENSITIVE_SK_RE.sub("[REDACTED]", text)
    text = SENSITIVE_BEARER_RE.sub("Bearer [REDACTED]", text)
    text = SENSITIVE_FIELD_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(4)}", text)
    return text


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _run_git_for_logs(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def configure_log_repo_args(args: argparse.Namespace) -> None:
    if args.push_logs_to_git is None:
        args.push_logs_to_git = _env_bool("AUTO_PUSH_LOGS_TO_GIT", False)
    args.log_repo_path = str(args.log_repo_path or os.getenv("LOG_REPO_PATH", "")).strip()
    args.log_repo_remote = str(os.getenv("LOG_REPO_REMOTE", "origin")).strip() or "origin"
    args.log_repo_branch = str(os.getenv("LOG_REPO_BRANCH", "main")).strip() or "main"
    args.log_repo_include_summary = _env_bool("LOG_REPO_INCLUDE_SUMMARY", True)
    args.log_repo_include_prompts = _env_bool("LOG_REPO_INCLUDE_PROMPTS", False)
    args.log_repo_include_strategy = _env_bool("LOG_REPO_INCLUDE_STRATEGY", False)


def check_log_repo_startup(args: argparse.Namespace) -> None:
    print("\n========== 日志仓库启动检查 ==========")
    repo_path_raw = getattr(args, "log_repo_path", "")
    if not repo_path_raw:
        print("LOG_REPO_PATH 未配置，跳过日志仓库启动拉取。")
        return
    repo_path = Path(repo_path_raw).expanduser()
    if not repo_path.exists():
        print(f"警告：LOG_REPO_PATH 不存在：{repo_path}")
        return
    if not (repo_path / ".git").exists():
        print(f"警告：LOG_REPO_PATH 不是 Git 仓库：{repo_path}")
        return
    remote = getattr(args, "log_repo_remote", "origin")
    branch = getattr(args, "log_repo_branch", "main")
    remote_cp = _run_git_for_logs(["git", "remote", "get-url", remote], repo_path)
    if remote_cp.returncode != 0:
        print(f"警告：日志仓库 remote 不存在：{remote}")
    branch_cp = _run_git_for_logs(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    current_branch = branch_cp.stdout.strip() if branch_cp.returncode == 0 else ""
    if current_branch != branch:
        print(f"警告：日志仓库当前分支为 {current_branch or '未知'}，目标分支为 {branch}。结束推送时会尝试 git checkout {branch}。")
    pull_cp = _run_git_for_logs(["git", "pull", "--ff-only", remote, branch], repo_path)
    if pull_cp.stdout:
        print(pull_cp.stdout.rstrip())
    if pull_cp.stderr:
        print(pull_cp.stderr.rstrip())
    if pull_cp.returncode != 0:
        print(f"警告：日志仓库启动拉取失败（不阻断运行）：git -C {repo_path} pull --ff-only {remote} {branch}")
    else:
        print(f"日志仓库启动拉取完成：git -C {repo_path} pull --ff-only {remote} {branch}")


def _copy_log_repo_summary_files(run_dir: Path, dest_dir: Path) -> None:
    run_log = run_dir / "run.log"
    if run_log.exists() and run_log.is_file():
        (dest_dir / "run.log").write_text(_sanitize_log_text(run_log.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
    for name in [
        "pre_run_ai_review.json",
        "pre_run_ai_review_prompt.txt",
        "pre_run_ai_review.raw.txt",
        ITERATION_STATS_FILE_NAME,
        "leaderboard.json",
        "pair_leaderboard.json",
        "recommended_pairs.json",
        "last_run_summary.json",
        "nearest_candidate_snapshot.json",
        "best_strategy_snapshot.json",
        "goal.runtime.json",
    ]:
        _copy_if_exists(run_dir / name, dest_dir / name)
    snapshots = [
        (BEST_STRATEGY_FILE, "best_strategy_snapshot.json"),
        (NEAREST_CANDIDATE_FILE, "nearest_candidate_snapshot.json"),
        (LAST_RUN_SUMMARY_FILE, "last_run_summary.json"),
        (PAIR_LEADERBOARD_FILE, "pair_leaderboard.json"),
        (RECOMMENDED_PAIRS_FILE, "recommended_pairs.json"),
    ]
    for src, dest_name in snapshots:
        _copy_if_exists(src, dest_dir / dest_name)


def _copy_log_repo_optional_prompts(run_dir: Path, dest_dir: Path) -> None:
    for prompt in sorted(run_dir.glob("v*/advisor_prompt.txt")) + sorted(run_dir.glob("v*/codegen_prompt.txt")):
        try:
            rel = prompt.relative_to(run_dir)
        except ValueError:
            continue
        _copy_if_exists(prompt, dest_dir / rel)


def _copy_log_repo_optional_strategy(run_dir: Path, dest_dir: Path) -> None:
    candidates: list[Path] = []
    if BEST_STRATEGY_FILE.exists():
        try:
            best_data = read_json(BEST_STRATEGY_FILE)
            if str(best_data.get("source_run_id", "")) == run_dir.name.replace("run_", ""):
                raw = str(best_data.get("strategy_file", "") or "")
                if raw:
                    candidates.append(Path(raw))
        except Exception:
            pass
    candidates.extend(sorted(run_dir.glob("v*/strategy.py"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    for src in candidates:
        if src.exists() and src.is_file():
            _copy_if_exists(src, dest_dir / "strategy.py")
            return


def push_run_logs_to_log_repo(run_dir: Path, args: argparse.Namespace) -> LogRepoPushResult:
    enabled = bool(getattr(args, "push_logs_to_git", False))
    repo_path_raw = str(getattr(args, "log_repo_path", "") or "").strip()
    branch = str(getattr(args, "log_repo_branch", "main") or "main")
    result = LogRepoPushResult(enabled=enabled, repo_path=Path(repo_path_raw).expanduser() if repo_path_raw else None, branch=branch)
    if not enabled:
        return result
    if not repo_path_raw:
        result.status = "失败"
        result.error = "LOG_REPO_PATH 未配置"
        print("警告：日志推送失败，但自动优化主流程已完成。")
        print(f"错误信息：{result.error}")
        return result
    repo_path = Path(repo_path_raw).expanduser()
    result.repo_path = repo_path
    remote = str(getattr(args, "log_repo_remote", "origin") or "origin")
    rel_log_dir = Path("logs") / datetime.utcnow().strftime("%Y-%m-%d") / run_dir.name
    result.rel_log_dir = rel_log_dir
    result.github_path = str(rel_log_dir / "run.log")

    try:
        if not repo_path.exists() or not (repo_path / ".git").exists():
            raise RuntimeError(f"日志仓库不存在或不是 Git 仓库：{repo_path}")
        run_log = run_dir / "run.log"
        if not run_log.exists():
            raise RuntimeError(f"本次 run.log 不存在：{run_log}")

        commands = [
            ["git", "checkout", branch],
            ["git", "pull", "--rebase", remote, branch],
        ]
        for cmd in commands:
            cp = _run_git_for_logs(cmd, repo_path)
            if cp.stdout:
                print(cp.stdout.rstrip())
            if cp.stderr:
                print(cp.stderr.rstrip())
            if cp.returncode != 0:
                raise RuntimeError(f"Git 命令失败：{' '.join(cmd)}")

        dest_dir = repo_path / rel_log_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        sanitized = _sanitize_log_text(run_log.read_text(encoding="utf-8", errors="ignore"))
        (dest_dir / "run.log").write_text(sanitized, encoding="utf-8")
        _copy_log_repo_summary_files(run_dir, dest_dir)
        if getattr(args, "log_repo_include_prompts", False):
            _copy_log_repo_optional_prompts(run_dir, dest_dir)
        if getattr(args, "log_repo_include_strategy", False):
            _copy_log_repo_optional_strategy(run_dir, dest_dir)

        commands = [
            ["git", "add", str(rel_log_dir)],
        ]
        for cmd in commands:
            cp = _run_git_for_logs(cmd, repo_path)
            if cp.stdout:
                print(cp.stdout.rstrip())
            if cp.stderr:
                print(cp.stderr.rstrip())
            if cp.returncode != 0:
                raise RuntimeError(f"Git 命令失败：{' '.join(cmd)}")
        diff_cp = _run_git_for_logs(["git", "diff", "--cached", "--quiet", "--", str(rel_log_dir)], repo_path)
        if diff_cp.returncode == 0:
            print("日志仓库没有新变更，跳过 commit。")
            result.status = "成功"
            return result
        commit_cp = _run_git_for_logs(["git", "commit", "-m", f"Add auto optimize log {run_dir.name}", "--", str(rel_log_dir)], repo_path)
        if commit_cp.stdout:
            print(commit_cp.stdout.rstrip())
        if commit_cp.stderr:
            print(commit_cp.stderr.rstrip())
        if commit_cp.returncode != 0:
            raise RuntimeError("git commit 失败。")
        push_cp = _run_git_for_logs(["git", "push", remote, branch], repo_path)
        if push_cp.stdout:
            print(push_cp.stdout.rstrip())
        if push_cp.stderr:
            print(push_cp.stderr.rstrip())
        if push_cp.returncode != 0:
            raise RuntimeError("git push 失败。")
        result.status = "成功"
        return result
    except Exception as exc:
        result.status = "失败"
        result.error = str(exc)
        print("警告：日志推送失败，但自动优化主流程已完成。")
        print(f"错误信息：{exc}")
        return result


def print_log_repo_push_summary(result: LogRepoPushResult) -> None:
    print("\n========== 日志仓库推送 ==========")
    print(f"日志仓库：{result.repo_path or '-'}")
    print(f"目标分支：{result.branch}")
    print(f"日志目录：{str(result.rel_log_dir) + '/' if result.rel_log_dir else '-'}")
    print(f"推送状态：{result.status}")
    print(f"GitHub 路径：{result.github_path or '-'}")
    print(f"错误信息：{result.error or '-'}")


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
    provider_pool: list[dict[str, Any]] = field(default_factory=list)
    used_provider: str = ""
    forced_provider_offset: int = 0

    @property
    def display_name(self) -> str:
        return ROLE_DISPLAY_NAMES.get(self.role, self.role)

    def begin_call(self) -> None:
        self.attempts = []
        self.used_model = ""
        self.used_provider = ""

    def usage_snapshot(self) -> dict[str, Any]:
        provider_pool_summary = [
            {"name": item.get("name", ""), "model": item.get("model", ""), "base_url": item.get("base_url", "")}
            for item in self.provider_pool
        ]
        return {
            "model_pool": list(self.model_pool),
            "provider_pool": provider_pool_summary,
            "used_model": self.used_model,
            "used_provider": self.used_provider,
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
        "provider_pool_env": "STRATEGY_ADVISOR_PROVIDER_POOL",
        "default_model": "claude-opus-4-7",
    },
    "code_generator": {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model_pool_env": "OPENAI_MODEL_POOL",
        "provider_pool_env": "STRATEGY_CODEGEN_PROVIDER_POOL",
        "default_model": "gpt-5.5",
    },
    "code_repair": {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model_pool_env": "OPENAI_MODEL_POOL",
        "provider_pool_env": "STRATEGY_CODEGEN_PROVIDER_POOL",
        "default_model": "gpt-5.5",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_recommended_pairs_data(pairs_file: str | Path | None) -> dict[str, Any]:
    if not pairs_file:
        return {}
    path = _resolve_repo_path(str(pairs_file))
    if not path.exists():
        raise FileNotFoundError(f"推荐币种文件不存在：{path}")
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _pair_name_from_recommended_item(item: Any) -> str:
    return str(item.get("pair") if isinstance(item, dict) else item).strip()


def _extract_active_pair_names(recommended_data: dict[str, Any]) -> list[str]:
    active = recommended_data.get("active_pairs", []) if isinstance(recommended_data, dict) else []
    pairs: list[str] = []
    if not isinstance(active, list):
        return pairs
    for item in active:
        pair = _pair_name_from_recommended_item(item)
        if pair and pair not in pairs:
            pairs.append(pair)
    return pairs


def _load_pairs_from_recommended_file(pairs_file: str | Path | None) -> list[str]:
    return _extract_active_pair_names(_load_recommended_pairs_data(pairs_file))


def _read_config_pair_whitelist(config_path: str | Path) -> list[str]:
    path = _resolve_repo_path(str(config_path))
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    data = read_json(path)
    exchange = data.get("exchange", {}) if isinstance(data, dict) else {}
    pair_whitelist = exchange.get("pair_whitelist", []) if isinstance(exchange, dict) else []
    if not isinstance(pair_whitelist, list):
        return []
    pairs: list[str] = []
    for item in pair_whitelist:
        pair = str(item).strip()
        if pair and pair not in pairs:
            pairs.append(pair)
    return pairs


def _write_temp_config_with_pairs(base_config: str, pairs: list[str], run_dir: Path, label: str) -> str:
    if not pairs:
        return base_config
    base_path = _resolve_repo_path(base_config)
    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{base_path}")
    config_data = read_json(base_path)
    exchange = config_data.setdefault("exchange", {})
    if not isinstance(exchange, dict):
        raise RuntimeError(f"配置文件 exchange 字段不是 object：{base_path}")
    exchange["pair_whitelist"] = list(pairs)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "pairs"
    dest = run_dir / f"config.{safe_label}.json"
    write_json(dest, config_data)
    try:
        return str(dest.relative_to(ROOT_DIR))
    except ValueError:
        return str(dest)


def _format_pair_source_label(source: str) -> str:
    labels = {
        "pairs_file": "--pairs-file",
        "refresh_pairs": "refresh-pairs",
        "default_auto": "recommended_pairs.json",
        "ignored": "原始 config",
        "missing": "原始 config",
        "fallback_original": "原始 config",
    }
    return labels.get(source, source)


def _format_legacy_pair_source_label(source: str) -> str:
    labels = {
        "pairs_file": "--pairs-file",
        "refresh_pairs": "--refresh-pairs",
        "default_auto": "默认自动读取",
        "ignored": "忽略",
        "missing": "不存在",
    }
    return labels.get(source, source)


def print_pair_selection_status(
    *,
    default_file: Path,
    default_exists: bool,
    used_recommended: bool,
    source: str,
    active_pairs: list[str],
    temp_config: str | None,
) -> None:
    print("\n========== 交易对选择状态 ==========")
    print(f"recommended_pairs 默认文件：{default_file}")
    print(f"是否存在：{'是' if default_exists else '否'}")
    print(f"本次是否使用 recommended_pairs：{'是' if used_recommended else '否'}")
    print(f"使用来源：{_format_legacy_pair_source_label(source)}")
    print(f"active_pairs 数量：{len(active_pairs)}")
    print("active_pairs：" + (", ".join(active_pairs) if active_pairs else "无"))
    print(f"临时 config 路径：{temp_config or '未生成（使用原始 config）'}")


def _print_numbered_pairs(pairs: list[str], empty_text: str = "无") -> None:
    if not pairs:
        print(empty_text)
        return
    for idx, pair in enumerate(pairs, start=1):
        print(f"{idx}. {pair}")


def _print_recommended_item_list(items: Any, *, detail_active: bool = False) -> None:
    if not isinstance(items, list) or not items:
        print("无")
        return
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            pair = _pair_name_from_recommended_item(item) or "-"
            if detail_active:
                score = item.get("score", "")
                reason = item.get("reason", "")
                score_text = score if score != "" else "-"
                reason_text = reason if reason != "" else "-"
                print(f"{idx}. {pair} | score={score_text} | reason={reason_text}")
            else:
                score = item.get("score", "")
                reason = item.get("reason", "")
                suffix_parts = []
                if score != "":
                    suffix_parts.append(f"score={score}")
                if reason:
                    suffix_parts.append(f"reason={reason}")
                suffix = " | " + " | ".join(suffix_parts) if suffix_parts else ""
                print(f"{idx}. {pair}{suffix}")
        else:
            print(f"{idx}. {item}")


def _store_pair_selection_log(runtime_goal: dict[str, Any], data: dict[str, Any]) -> None:
    runtime_goal["runtime_pair_selection_log"] = data


def print_effective_pair_selection_log(runtime_goal: dict[str, Any]) -> None:
    info = runtime_goal.get("runtime_pair_selection_log", {})
    if not isinstance(info, dict) or not info:
        return

    source = str(info.get("source", "missing"))
    fallback_reason = str(info.get("fallback_reason") or "")
    display_source = "fallback_original" if fallback_reason == "empty_active_pairs" else source
    source_label = _format_pair_source_label(display_source)
    effective_pairs = [str(pair) for pair in info.get("effective_pairs", []) if str(pair).strip()]
    recommended_data = info.get("recommended_data", {}) if isinstance(info.get("recommended_data"), dict) else {}
    recommended_path = str(info.get("recommended_file") or "")
    pairs_file_path = str(info.get("pairs_file") or "")
    temp_config = str(info.get("temp_config") or "")

    print("\n========== 本次实际测试交易对 ==========")
    print(f"交易对来源：{source_label}")
    if source == "pairs_file" and pairs_file_path:
        print(f"文件路径：{pairs_file_path}")
    if source == "missing":
        print("说明：未找到 recommended_pairs.json，本次未启用优质币筛选。")
    elif source == "ignored":
        print("说明：已通过 --ignore-recommended-pairs 忽略 recommended_pairs.json。")
    if fallback_reason == "empty_active_pairs":
        print("⚠️ recommended_pairs.active_pairs 为空，已回退使用原始 config pair_whitelist。")
    print(f"交易对数量：{len(effective_pairs)}")
    print("交易对列表：")
    _print_numbered_pairs(effective_pairs)
    if temp_config:
        print("\n本次临时 config：")
        print(temp_config)

    if fallback_reason == "empty_active_pairs" or source not in {"default_auto", "refresh_pairs"}:
        return

    active_items = recommended_data.get("active_pairs", []) if isinstance(recommended_data, dict) else []
    watch_items = recommended_data.get("watch_pairs", []) if isinstance(recommended_data, dict) else []
    cooldown_items = recommended_data.get("cooldown_pairs", []) if isinstance(recommended_data, dict) else []
    scan_periods = recommended_data.get("scan_periods", "") if isinstance(recommended_data, dict) else ""

    print("\n========== 优质交易对筛选来源 ==========")
    print(f"recommended_pairs 文件：{recommended_path}")
    print(f"source_strategy：{recommended_data.get('source_strategy', '')}")
    print(f"created_at：{recommended_data.get('created_at', '')}")
    print(f"scan_periods：{scan_periods}")
    print(f"active_pairs 数量：{len(active_items) if isinstance(active_items, list) else 0}")
    print(f"watch_pairs 数量：{len(watch_items) if isinstance(watch_items, list) else 0}")
    print(f"cooldown_pairs 数量：{len(cooldown_items) if isinstance(cooldown_items, list) else 0}")

    print("\n========== 优质交易对 active_pairs ==========")
    _print_recommended_item_list(active_items, detail_active=True)

    print("\n========== 观察交易对 watch_pairs ==========")
    _print_recommended_item_list(watch_items)

    print("\n========== 暂时冷却交易对 cooldown_pairs ==========")
    _print_recommended_item_list(cooldown_items)
    print("说明：cooldown_pairs 只是当前区间暂时不参与优化，不是永久剔除。")


def apply_recommended_pairs_override(runtime_goal: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    base_config = str(runtime_goal.get("config", args.config))
    default_file = RECOMMENDED_PAIRS_FILE
    default_exists = default_file.exists()
    pairs_path: str | Path | None = None
    source = "missing"
    recommended_data: dict[str, Any] = {}
    active_pairs: list[str] = []
    temp_config: str | None = None
    fallback_reason = ""

    original_pairs = _read_config_pair_whitelist(base_config)

    if getattr(args, "pairs_file", None):
        pairs_path = getattr(args, "pairs_file")
        source = "pairs_file"
        recommended_data = _load_recommended_pairs_data(pairs_path)
        active_pairs = _extract_active_pair_names(recommended_data)
    elif getattr(args, "ignore_recommended_pairs", False):
        source = "ignored"
    elif getattr(args, "refresh_pairs", False):
        source = "refresh_pairs"
        print("\n========== 刷新 recommended_pairs ==========")
        print("已启用 --refresh-pairs：optimize 开始前先执行一次 pair-scan。")
        scan_goal = dict(runtime_goal)
        scan_goal["config"] = base_config
        run_pair_scan(scan_goal, args, run_dir)
        default_exists = default_file.exists()
        pairs_path = default_file
        recommended_data = _load_recommended_pairs_data(pairs_path) if default_exists else {}
        active_pairs = _extract_active_pair_names(recommended_data)
    elif default_exists:
        pairs_path = default_file
        source = "default_auto"
        recommended_data = _load_recommended_pairs_data(pairs_path)
        active_pairs = _extract_active_pair_names(recommended_data)

    if pairs_path and not active_pairs:
        fallback_reason = "empty_active_pairs"
        print("警告：recommended_pairs.active_pairs 为空，本次 fallback 使用原始 config 的 pair_whitelist。")

    if active_pairs:
        effective_pairs = active_pairs
        temp_config = _write_temp_config_with_pairs(base_config, active_pairs, run_dir, "active_pairs")
        runtime_goal["config"] = temp_config
        runtime_goal["pairs"] = active_pairs
        runtime_goal["runtime_active_pairs"] = active_pairs
        args.config = temp_config
    else:
        effective_pairs = original_pairs
        runtime_goal["config"] = base_config
        runtime_goal["pairs"] = original_pairs
        runtime_goal.pop("runtime_active_pairs", None)
        args.config = base_config

    print_pair_selection_status(
        default_file=default_file,
        default_exists=default_exists,
        used_recommended=bool(active_pairs),
        source=source,
        active_pairs=active_pairs,
        temp_config=temp_config,
    )

    _store_pair_selection_log(
        runtime_goal,
        {
            "source": source,
            "source_label": _format_pair_source_label(source),
            "default_recommended_file": str(default_file),
            "default_recommended_exists": default_exists,
            "recommended_file": str(_resolve_repo_path(str(pairs_path))) if pairs_path else "",
            "pairs_file": str(_resolve_repo_path(str(pairs_path))) if source == "pairs_file" and pairs_path else "",
            "base_config": base_config,
            "temp_config": temp_config or "",
            "original_config_pairs": original_pairs,
            "active_pairs": active_pairs,
            "effective_pairs": effective_pairs,
            "recommended_data": recommended_data,
            "fallback_reason": fallback_reason,
        },
    )
    return active_pairs


def apply_pairs_file_override(runtime_goal: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> list[str]:
    return apply_recommended_pairs_override(runtime_goal, args, run_dir)


def _auto_trade_count_target_cfg(runtime_goal: dict[str, Any]) -> dict[str, Any]:
    raw = runtime_goal.get("auto_trade_count_target", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "min_trades_per_pair": int(raw.get("min_trades_per_pair", 4) or 4),
        "ideal_min_trades_per_pair": int(raw.get("ideal_min_trades_per_pair", 5) or 5),
        "ideal_max_trades_per_pair": int(raw.get("ideal_max_trades_per_pair", 8) or 8),
        "max_trades_per_pair": int(raw.get("max_trades_per_pair", 10) or 10),
        "min_total_trades_floor": int(raw.get("min_total_trades_floor", 20) or 20),
        "ideal_max_trades_cap": int(raw.get("ideal_max_trades_cap", 75) or 75),
        "max_trades_cap": int(raw.get("max_trades_cap", 90) or 90),
    }


def _auto_trade_count_pair_source(runtime_goal: dict[str, Any]) -> tuple[int | None, str, list[str]]:
    """Return the actual pairs used by this optimize run and their source."""
    info = runtime_goal.get("runtime_pair_selection_log", {})
    if not isinstance(info, dict):
        info = {}

    active_pairs = [str(pair) for pair in info.get("active_pairs", []) if str(pair).strip()]
    effective_pairs = [str(pair) for pair in info.get("effective_pairs", []) if str(pair).strip()]
    original_pairs = [str(pair) for pair in info.get("original_config_pairs", []) if str(pair).strip()]
    source = str(info.get("source") or "")

    if active_pairs:
        if source == "pairs_file":
            return len(active_pairs), "--pairs-file", active_pairs
        return len(active_pairs), "recommended_pairs", active_pairs
    if effective_pairs:
        return len(effective_pairs), "原始 config", effective_pairs
    if original_pairs:
        return len(original_pairs), "原始 config", original_pairs

    runtime_active_pairs = [str(pair) for pair in runtime_goal.get("runtime_active_pairs", []) if str(pair).strip()]
    if runtime_active_pairs:
        return len(runtime_active_pairs), "recommended_pairs", runtime_active_pairs
    runtime_pairs = [str(pair) for pair in runtime_goal.get("pairs", []) if str(pair).strip()]
    if runtime_pairs:
        return len(runtime_pairs), "原始 config", runtime_pairs
    return None, "取不到", []


def _runtime_trade_target_values(runtime_goal: dict[str, Any]) -> dict[str, int]:
    target = runtime_goal.get("target", {}) or {}
    return {
        "min_trades": _safe_int(target.get("min_trades", 25)),
        "ideal_min_trades": _safe_int(target.get("ideal_min_trades", target.get("min_trades", 25))),
        "ideal_max_trades": _safe_int(target.get("ideal_max_trades", target.get("max_trades", 80))),
        "max_trades": _safe_int(target.get("max_trades", 80)),
    }


def _runtime_trade_target_text(runtime_goal: dict[str, Any]) -> str:
    values = _runtime_trade_target_values(runtime_goal)
    return (
        f"保持目标交易数 {values['min_trades']}~{values['max_trades']}，"
        f"理想区间 {values['ideal_min_trades']}~{values['ideal_max_trades']}。"
        f"不要低于 {values['min_trades']}，也不要高于 {values['max_trades']}。"
    )


def _apply_runtime_trade_target_prompt_guidance(runtime_goal: dict[str, Any]) -> None:
    """Make runtime prompt guidance agree with final trade-count targets without editing the goal file."""
    guidance = runtime_goal.get("prompt_guidance")
    if not isinstance(guidance, dict):
        return
    replacement = _runtime_trade_target_text(runtime_goal)
    for key in ("advisor_rules", "codegen_rules"):
        values = guidance.get(key)
        if not isinstance(values, list):
            continue
        cleaned = [item for item in values if "交易数" not in str(item)]
        cleaned.append(replacement)
        guidance[key] = cleaned


def apply_auto_trade_count_target(runtime_goal: dict[str, Any]) -> None:
    cfg = _auto_trade_count_target_cfg(runtime_goal)
    target = runtime_goal.setdefault("target", {})
    if not isinstance(target, dict):
        target = {}
        runtime_goal["target"] = target

    pair_count, pair_source, _pairs = _auto_trade_count_pair_source(runtime_goal)
    enabled = bool(cfg.get("enabled", False))
    computed: dict[str, int] | None = None
    if enabled and pair_count and pair_count > 0:
        min_trades = max(pair_count * int(cfg["min_trades_per_pair"]), int(cfg["min_total_trades_floor"]))
        ideal_min_trades = max(pair_count * int(cfg["ideal_min_trades_per_pair"]), min_trades)
        ideal_max_trades = min(pair_count * int(cfg["ideal_max_trades_per_pair"]), int(cfg["ideal_max_trades_cap"]))
        max_trades = min(pair_count * int(cfg["max_trades_per_pair"]), int(cfg["max_trades_cap"]))
        if ideal_max_trades < ideal_min_trades:
            ideal_max_trades = ideal_min_trades
        if max_trades < ideal_max_trades:
            max_trades = ideal_max_trades
        computed = {
            "min_trades": min_trades,
            "ideal_min_trades": ideal_min_trades,
            "ideal_max_trades": ideal_max_trades,
            "max_trades": max_trades,
        }
        target.update(computed)
        runtime_goal["auto_trade_count_target_runtime"] = {
            "enabled": True,
            "pair_count": pair_count,
            "source": pair_source,
            "computed_min_trades": min_trades,
            "computed_ideal_min_trades": ideal_min_trades,
            "computed_ideal_max_trades": ideal_max_trades,
            "computed_max_trades": max_trades,
        }
    else:
        runtime_goal["auto_trade_count_target_runtime"] = {
            "enabled": enabled,
            "pair_count": pair_count,
            "source": pair_source,
        }

    _apply_runtime_trade_target_prompt_guidance(runtime_goal)
    values = _runtime_trade_target_values(runtime_goal)
    runtime_goal.update(values)

    print("========== 自动交易数目标 ==========")
    print(f"是否启用：{'是' if enabled else '否'}")
    print(f"交易对数量 pair_count：{pair_count if pair_count is not None else '取不到'}")
    print(f"交易对来源：{pair_source}")
    print(f"min_trades_per_pair：{cfg['min_trades_per_pair']}")
    print(f"ideal_min_trades_per_pair：{cfg['ideal_min_trades_per_pair']}")
    print(f"ideal_max_trades_per_pair：{cfg['ideal_max_trades_per_pair']}")
    print(f"max_trades_per_pair：{cfg['max_trades_per_pair']}")
    print("自动计算结果：")
    print(f"min_trades: {values['min_trades']}")
    print(f"ideal_min_trades: {values['ideal_min_trades']}")
    print(f"ideal_max_trades: {values['ideal_max_trades']}")
    print(f"max_trades: {values['max_trades']}")
    print("上限：")
    print(f"ideal_max_trades_cap: {cfg['ideal_max_trades_cap']}")
    print(f"max_trades_cap: {cfg['max_trades_cap']}")
    if enabled and computed is None:
        print("警告：auto_trade_count_target 已启用，但无法取得本次实际测试交易对数量；保留 optimization_goal.json 中原来的固定交易数目标。")


def _auto_trade_count_target_prompt_note(runtime_goal: dict[str, Any]) -> str:
    runtime = runtime_goal.get("auto_trade_count_target_runtime", {}) or {}
    if not isinstance(runtime, dict) or not runtime.get("enabled"):
        return ""
    values = _runtime_trade_target_values(runtime_goal)
    pair_count = runtime.get("pair_count")
    if not pair_count:
        return ""
    return (
        f"本次实际使用 {pair_count} 个交易对。\n"
        "自动交易数目标：\n"
        f"- 最低交易数：{values['min_trades']}\n"
        f"- 理想交易数区间：{values['ideal_min_trades']}~{values['ideal_max_trades']}\n"
        f"- 最大交易数：{values['max_trades']}\n"
        "以上交易数目标来自 runtime_goal 的最终值，必须覆盖历史经验或旧 prompt 中的固定交易数。\n"
    )

def compute_global_strategy_stats() -> dict[str, int]:
    """Scan all ai optimization run/version directories and return project-wide counters."""
    run_dirs = sorted(p for p in RESULT_ROOT.glob("run_*") if p.is_dir())
    stats = {
        "run_count": len(run_dirs),
        "nonempty_run_count": 0,
        "version_dir_count": 0,
        "strategy_file_count": 0,
        "mutation_spec_count": 0,
        "train_backtested_count": 0,
        "validation_backtested_count": 0,
        "summary_count": 0,
        "valid_strategy_count": 0,
        "new_best_count": 0,
    }
    for run_dir in run_dirs:
        version_dirs = sorted(p for p in run_dir.glob("v*") if p.is_dir())
        if version_dirs:
            stats["nonempty_run_count"] += 1
        stats["version_dir_count"] += len(version_dirs)
        for version_dir in version_dirs:
            if (version_dir / "strategy.py").exists():
                stats["strategy_file_count"] += 1
            if (version_dir / "mutation_spec.json").exists():
                stats["mutation_spec_count"] += 1
            if any((version_dir / "backtests").glob("train_*.zip")):
                stats["train_backtested_count"] += 1
            if any((version_dir / "backtests").glob("validation_*.zip")):
                stats["validation_backtested_count"] += 1
            summary_file = version_dir / "summary.json"
            if not summary_file.exists():
                continue
            stats["summary_count"] += 1
            try:
                summary = read_json(summary_file)
            except Exception:  # noqa: BLE001
                continue
            if summary.get("is_valid") is True:
                stats["valid_strategy_count"] += 1
            if summary.get("is_best") is True:
                stats["new_best_count"] += 1
    return stats


def update_iteration_global_stats(iteration_stats: dict[str, Any]) -> None:
    """Attach project-wide counters and the current retained memory size to iteration stats."""
    strategy_memory_retained_count = len(_read_json_list_file(MEMORY_FILE)) if MEMORY_FILE.exists() else 0
    global_stats = compute_global_strategy_stats()
    global_stats["strategy_memory_retained_count"] = strategy_memory_retained_count
    iteration_stats["strategy_memory_retained_count"] = strategy_memory_retained_count
    iteration_stats["global_stats"] = global_stats


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


def parse_cli_y_or_n(value: str, option_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"y", "n"}:
        raise argparse.ArgumentTypeError(f"{option_name} 只允许 y 或 n")
    return normalized


def ask_confirm(question: str, default: bool = False, args: argparse.Namespace | None = None) -> bool:
    if args is not None and getattr(args, "confirm", "n") == "n":
        print(f"{question}：中途人工确认已关闭，自动选择不确认/跳过（--confirm n）。")
        return False

    d = "是" if default else "否"
    while True:
        v = input(f"{question}（默认：{d}，输入 y/n）：").strip()
        if not v:
            return default
        p = parse_yes_no(v)
        if p is not None:
            return p
        print("输入无效，请输入 y 或 n。")


def print_interaction_config(args: argparse.Namespace) -> None:
    setup_label = "进入" if getattr(args, "setup", "n") == "y" and not getattr(args, "no_wizard", False) else "跳过"
    setup_reason = "--no-wizard" if getattr(args, "no_wizard", False) else f"--setup {getattr(args, 'setup', 'n')}"
    confirm_label = "开启" if getattr(args, "confirm", "n") == "y" else "关闭"
    print("\n========== 启动交互配置 ==========")
    print(f"交互式设置：{setup_label}（{setup_reason}）")
    print(f"中途人工确认：{confirm_label}（--confirm {getattr(args, 'confirm', 'n')}）")


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
    if getattr(args, "setup", "n") != "y":
        print("交互式设置已跳过（--setup n）。")
        return goal
    print("交互式设置已进入（--setup y）。")

    runtime = json.loads(json.dumps(goal))
    runtime.setdefault("language", "zh-CN")
    runtime["strategy_family"] = ask_text("策略家族名", str(runtime.get("strategy_family", args.base_strategy)))
    runtime["config"] = ask_text("配置文件路径", str(runtime.get("config", args.config)))
    cfg_path = ROOT_DIR / runtime["config"]
    if not cfg_path.exists():
        cont = ask_confirm(f"配置文件不存在：{runtime['config']}，是否继续", False, args)
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

    if not ask_confirm("是否开始自动优化", True, args):
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


def _backtest_zip_candidates(results_dir: Path, started_at: float, existing_zips: set[Path]) -> list[Path]:
    """Return only zips that could have been produced by the current backtest."""
    candidates: list[Path] = []
    for zip_path in _list_backtest_zips(results_dir):
        try:
            is_new = zip_path not in existing_zips
            is_recent = zip_path.stat().st_mtime >= started_at
        except OSError:
            continue
        if is_new or is_recent:
            candidates.append(zip_path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_backtest_zip_payload(zip_path: Path) -> tuple[dict[str, Any], str]:
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
        with zf.open(json_name) as fp:
            data = json.load(fp)
    if not isinstance(data, dict):
        raise RuntimeError(f"回测结果 JSON 顶层不是 object: {zip_path}")
    return data, json_name


def _read_backtest_zip_strategy_keys(zip_path: Path) -> list[str]:
    data, _ = _read_backtest_zip_payload(zip_path)
    strategy_data = data.get("strategy")
    if not isinstance(strategy_data, dict):
        return []
    return sorted(str(k) for k in strategy_data.keys())


def find_backtest_zip_for_strategy(
    backtest_dir: Path,
    strategy_class: str,
    started_at: float,
    existing_zips: set[Path],
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Find a current-run backtest zip whose strategy keys include strategy_class."""
    details: list[dict[str, Any]] = []
    for zip_path in _backtest_zip_candidates(backtest_dir, started_at, existing_zips):
        try:
            strategy_keys = _read_backtest_zip_strategy_keys(zip_path)
            error = ""
        except Exception as exc:  # noqa: BLE001 - bad zips must not stop optimization.
            strategy_keys = []
            error = f"{type(exc).__name__}: {exc}"
        details.append({"zip": str(zip_path), "actual_strategies": strategy_keys, "error": error})
        if strategy_class in strategy_keys:
            return zip_path, details
    return None, details


def _log_backtest_zip_filter_failure(strategy_class: str, candidates: list[dict[str, Any]]) -> None:
    print("回测结果 zip 筛选失败：")
    print(f"期望策略：{strategy_class}")
    print("候选 zip：")
    if not candidates:
        print("- 无")
    for item in candidates:
        actual = ", ".join(item.get("actual_strategies") or []) or "无"
        suffix = f"，读取错误：{item.get('error')}" if item.get("error") else ""
        print(f"- {item.get('zip')}，实际策略：{actual}{suffix}")
    print("处理结果：当前区间回测结果无效，当前轮标记无效，继续下一轮。")


def _copy_backtest_zip_to_version(zip_path: Path, version_dir: Path, stage: str, timerange: str, label: str | None = None) -> Path:
    safe_label = f"_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}" if label else ""
    dest_name = f"{stage}{safe_label}_{timerange}.zip"
    dest = version_dir / "backtests" / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, dest)
    return dest


def _write_backtest_process_log(version_dir: Path, stage: str, timerange: str, cp: subprocess.CompletedProcess[str]) -> Path:
    log_dir = version_dir / "backtest_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
    safe_timerange = re.sub(r"[^A-Za-z0-9_.-]+", "_", timerange)
    path = log_dir / f"{safe_stage}_{safe_timerange}.log"
    path.write_text(
        f"[{stage} {timerange}]\nRETURNCODE: {cp.returncode}\nSTDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}\n",
        encoding="utf-8",
    )
    return path


def _record_backtest_error(
    errors: list[dict[str, Any]],
    *,
    stage: str,
    timerange: str,
    expected_strategy: str,
    error: str,
    wrong_zip: str = "",
    actual_strategies: list[str] | None = None,
) -> None:
    errors.append({
        "stage": stage,
        "timerange": timerange,
        "expected_strategy": expected_strategy,
        "wrong_zip": wrong_zip,
        "actual_strategies": actual_strategies or [],
        "error": error,
    })


def _print_backtest_mismatch_summary(errors: list[dict[str, Any]]) -> None:
    mismatch_errors = [e for e in errors if e.get("error") in {"wrong_strategy_zip_detected", "backtest_parse_failed", "zip_missing"}]
    if not mismatch_errors:
        return
    print("\n检测到回测 zip 错配：")
    for item in mismatch_errors:
        actual = ", ".join(item.get("actual_strategies") or []) or "未知"
        print(f"当前策略：{item.get('expected_strategy')}")
        print(f"错误 zip 实际策略：{actual}")
        if item.get("wrong_zip"):
            print(f"错误 zip：{item.get('wrong_zip')}")
    print("原因：可能存在并发回测或全局 latest zip 误判。")
    print("处理：当前轮已标记无效，程序继续后续轮次。")


def parse_backtest_from_zip(zip_path: Path, strategy_class: str, strict: bool = True) -> tuple[dict[str, Any] | None, list[str]]:
    print(f"正在解析 zip: {zip_path}")
    try:
        data, json_name = _read_backtest_zip_payload(zip_path)
    except Exception:
        if strict:
            raise
        return None, []
    print(f"正在读取 json: {json_name}")

    strategy_data = data.get("strategy")
    if not isinstance(strategy_data, dict):
        if strict:
            raise RuntimeError("回测结果缺少 strategy 字段")
        return None, []
    actual_strategy_keys = sorted(str(k) for k in strategy_data.keys())
    if strategy_class not in strategy_data:
        if strict:
            raise RuntimeError(f"未在回测结果中找到当前策略 {strategy_class}，实际策略：{actual_strategy_keys}")
        return None, actual_strategy_keys
    result = strategy_data[strategy_class]
    print(f"找到策略: {strategy_class}")
    return result, actual_strategy_keys


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
    result_zip, candidates = find_backtest_zip_for_strategy(results_dir, class_name, start_ts, before_zips)
    if result_zip is None:
        _log_backtest_zip_filter_failure(class_name, candidates)
        raise RuntimeError(f"未找到包含当前策略 {class_name} 的本次回测 zip")
    result, actual_keys = parse_backtest_from_zip(result_zip, class_name, strict=False)
    if result is None:
        raise RuntimeError(f"未在回测结果中找到当前策略 {class_name}，实际策略：{actual_keys}")
    return _extract_metrics(result)


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


RANDOM_SAMPLE_USAGE = {
    "used_for_final_score": False,
    "used_for_best_selection": False,
    "used_for_ai_prompt": False,
    "manual_observation_only": True,
}


def _strip_random_samples_for_ai_prompt(data: Any) -> Any:
    """Return a copy without random-sample details so advisor prompts ignore observation-only data."""
    if isinstance(data, dict):
        return {
            k: _strip_random_samples_for_ai_prompt(v)
            for k, v in data.items()
            if not str(k).startswith("random_sample")
        }
    if isinstance(data, list):
        return [_strip_random_samples_for_ai_prompt(v) for v in data]
    return data


def _parse_yyyymmdd(value: str, arg_name: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} 必须是 YYYYMMDD 格式，例如 20260101") from exc


def _timerange_days(timerange: str) -> int:
    if not TIMERANGE_RE.match(str(timerange or "")):
        return 0
    start_raw, end_raw = str(timerange).split("-", 1)
    return max(0, (_parse_yyyymmdd(end_raw, "timerange.end") - _parse_yyyymmdd(start_raw, "timerange.start")).days)


def _timerange_overlap_days(a: str, b: str) -> int:
    if not TIMERANGE_RE.match(str(a or "")) or not TIMERANGE_RE.match(str(b or "")):
        return 0
    a_start_raw, a_end_raw = str(a).split("-", 1)
    b_start_raw, b_end_raw = str(b).split("-", 1)
    a_start = _parse_yyyymmdd(a_start_raw, "timerange.start")
    a_end = _parse_yyyymmdd(a_end_raw, "timerange.end")
    b_start = _parse_yyyymmdd(b_start_raw, "timerange.start")
    b_end = _parse_yyyymmdd(b_end_raw, "timerange.end")
    return max(0, (min(a_end, b_end) - max(a_start, b_start)).days)


def _random_sample_enabled(args: argparse.Namespace) -> bool:
    return _safe_int(getattr(args, "random_sample_windows", 0)) > 0


def build_random_sample_plan(args: argparse.Namespace, runtime_goal: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Generate and persist the run-level random-sample window plan."""
    requested = max(0, _safe_int(getattr(args, "random_sample_windows", 0)))
    enabled = requested > 0
    plan: dict[str, Any] = {
        "enabled": enabled,
        "count_requested": requested,
        "min_days": _safe_int(getattr(args, "random_sample_min_days", 25)),
        "max_days": _safe_int(getattr(args, "random_sample_max_days", 35)),
        "data_start": getattr(args, "random_sample_data_start", None),
        "data_end": getattr(args, "random_sample_data_end", None),
        "seed": getattr(args, "random_sample_seed", None),
        "max_overlap_days": 7,
        "usage": dict(RANDOM_SAMPLE_USAGE),
        "windows": [],
    }
    if not enabled:
        return plan

    start = _parse_yyyymmdd(str(plan["data_start"]), "--random-sample-data-start")
    end = _parse_yyyymmdd(str(plan["data_end"]), "--random-sample-data-end")
    min_days = int(plan["min_days"])
    max_days = int(plan["max_days"])
    if min_days <= 0 or max_days < min_days:
        raise ValueError("--random-sample-min-days 必须大于 0，且不能大于 --random-sample-max-days")
    total_days = (end - start).days
    if total_days < min_days:
        raise ValueError("随机采样数据范围短于最小窗口天数，无法生成随机采样窗口")

    holdout_timeranges = {
        str(h.get("timerange"))
        for h in (runtime_goal.get("holdout_ranges", []) or [])
        if isinstance(h, dict) and TIMERANGE_RE.match(str(h.get("timerange") or ""))
    }
    rng = random.Random(plan["seed"])
    windows: list[dict[str, Any]] = []
    attempts = max(500, requested * 500)
    for _ in range(attempts):
        if len(windows) >= requested:
            break
        days = rng.randint(min_days, min(max_days, total_days))
        start_offset = rng.randint(0, total_days - days)
        w_start = start + timedelta(days=start_offset)
        w_end = w_start + timedelta(days=days)
        timerange = f"{w_start:%Y%m%d}-{w_end:%Y%m%d}"
        if timerange in holdout_timeranges:
            continue
        if any(w["timerange"] == timerange for w in windows):
            continue
        if any(_timerange_overlap_days(timerange, w["timerange"]) > int(plan["max_overlap_days"]) for w in windows):
            continue
        windows.append({"label": f"random_{len(windows) + 1:03d}", "timerange": timerange, "days": days})
    if len(windows) < requested:
        print(f"警告：随机采样窗口仅生成 {len(windows)}/{requested} 个；请扩大数据范围或降低窗口数量。")
    plan["windows"] = windows
    write_json(run_dir / "random_sample_plan.json", plan)
    return plan


def print_random_sample_config(plan: dict[str, Any]) -> None:
    if not plan.get("enabled"):
        print("额外随机采样：未启用")
        return
    print("\n========== 额外随机采样配置 ==========")
    print(f"随机采样窗口数：{plan.get('count_requested')}")
    print(f"随机采样数据范围：{plan.get('data_start')}-{plan.get('data_end')}")
    print(f"窗口天数：{plan.get('min_days')}~{plan.get('max_days')}")
    print("用途：仅人工观察，不参与评分，不提供给 AI")


def _flatten_random_sample_metric(window: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": window.get("label", ""),
        "timerange": window.get("timerange", ""),
        "days": _safe_int(window.get("days")) or _timerange_days(str(window.get("timerange") or "")),
        "total_trades": _safe_int(metrics.get("total_trades")),
        "profit_total_abs": _safe_float(metrics.get("profit_total_abs")),
        "profit_total_pct": _safe_float(metrics.get("profit_total_pct")),
        "profit_factor": _safe_float(metrics.get("profit_factor")),
        "max_drawdown_pct": _max_drawdown_pct(metrics),
        "roi_profit_abs": _safe_float(metrics.get("roi_profit_abs")),
        "stop_loss_profit_abs": _safe_float(metrics.get("stop_loss_profit_abs")),
        "trailing_stop_loss_profit_abs": _safe_float(metrics.get("trailing_stop_loss_profit_abs")),
        "force_exit_profit_abs": _safe_float(metrics.get("force_exit_profit_abs")),
    }


def run_random_sample_backtests(
    *,
    plan: dict[str, Any],
    train_cmd: list[str],
    results_dir: Path,
    version_dir: Path,
    class_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Run observation-only random samples after all official scoring decisions are complete."""
    windows = list(plan.get("windows", []) or [])
    if not plan.get("enabled") or not windows:
        return [], [], "skipped"
    metrics_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for window in windows:
        label = str(window.get("label") or f"random_{len(metrics_rows) + 1:03d}")
        timerange = str(window.get("timerange") or "")
        if not TIMERANGE_RE.match(timerange):
            errors.append({"label": label, "timerange": timerange, "error": "invalid_timerange"})
            continue
        print(f"正在回测额外随机采样窗口（仅人工观察）：{label} {timerange}")
        cmd = train_cmd.copy()
        cmd[cmd.index("--timerange") + 1] = timerange
        before = set(_list_backtest_zips(results_dir))
        started_at = time.time()
        cp = run_cmd(cmd, ROOT_DIR)
        _write_backtest_process_log(version_dir, f"random_sample_{label}", timerange, cp)
        if cp.returncode != 0:
            errors.append({"label": label, "timerange": timerange, "error": "backtest_failed", "stderr_tail": cp.stderr[-1000:]})
            continue
        result_zip, candidates = find_backtest_zip_for_strategy(results_dir, class_name, started_at, before)
        if result_zip is None:
            _log_backtest_zip_filter_failure(class_name, candidates)
            errors.append({"label": label, "timerange": timerange, "error": "zip_missing_or_wrong_strategy"})
            continue
        zip_local = _copy_backtest_zip_to_version(result_zip, version_dir, "random_sample", timerange, label)
        result, actual_keys = parse_backtest_from_zip(zip_local, class_name, strict=False)
        if result is None:
            errors.append({"label": label, "timerange": timerange, "error": "backtest_parse_failed", "actual_strategies": actual_keys})
            continue
        metrics_rows.append(_flatten_random_sample_metric(window, _extract_metrics(result)))
    status = "completed" if len(metrics_rows) == len(windows) and not errors else ("failed" if errors else "completed")
    return metrics_rows, errors, status


def summarize_random_sample_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "random_sample_observation_only": True,
            "count": 0,
            "average_profit_total_pct": 0.0,
            "average_profit_factor": 0.0,
            "worst_window": None,
            "obviously_unstable": False,
        }
    avg_profit = sum(_safe_float(r.get("profit_total_pct")) for r in rows) / len(rows)
    avg_pf = sum(_safe_float(r.get("profit_factor")) for r in rows) / len(rows)
    worst = min(rows, key=lambda r: _safe_float(r.get("profit_total_pct")))
    unstable = any(_safe_float(r.get("profit_total_pct")) < -0.5 or _safe_float(r.get("profit_factor")) < 0.8 for r in rows)
    return {
        "random_sample_observation_only": True,
        "count": len(rows),
        "average_profit_total_pct": avg_profit,
        "average_profit_factor": avg_pf,
        "worst_window": worst,
        "obviously_unstable": bool(unstable),
    }


def print_random_sample_observation(rows: list[dict[str, Any]]) -> None:
    print("\n========== 额外随机采样窗口观察 ==========")
    print("说明：以下结果仅供人工观察，不参与 final_score，不影响 best，不提供给下一轮 AI。")
    for row in rows:
        print(
            f"{row.get('label')} {row.get('timerange')}：交易数 {_safe_int(row.get('total_trades'))}，"
            f"收益 {_safe_float(row.get('profit_total_pct')):.2f}%，PF {_safe_float(row.get('profit_factor')):.2f}，"
            f"DD {_safe_float(row.get('max_drawdown_pct')):.2f}%"
        )
    observation = summarize_random_sample_observation(rows)
    worst = observation.get("worst_window") or {}
    print("\n人工观察结论：")
    print(f"- 随机窗口平均收益率：{_safe_float(observation.get('average_profit_total_pct')):.2f}%")
    print(f"- 随机窗口平均 PF：{_safe_float(observation.get('average_profit_factor')):.2f}")
    print(f"- 最差随机窗口：{worst.get('label', '无')} {worst.get('timerange', '')} 收益 {_safe_float(worst.get('profit_total_pct')):.2f}%")
    print(f"- 是否存在明显不稳：{'是' if observation.get('obviously_unstable') else '否'}")

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


def _strategy_label(data: dict[str, Any] | None, default: str = "无") -> str:
    if not isinstance(data, dict) or not data:
        return default
    return str(data.get("strategy_class") or data.get("class_name") or data.get("version") or default)


def _compact_prompt_json(data: Any, max_chars: int = 6000) -> str:
    if data is None:
        return "null"
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _prompt_guidance_rules(runtime_goal: dict[str, Any], key: str) -> list[str]:
    guidance = runtime_goal.get("prompt_guidance", {}) or {}
    raw_rules = guidance.get(key, []) if isinstance(guidance, dict) else []
    if not isinstance(raw_rules, list):
        return []
    return [str(rule).strip() for rule in raw_rules if str(rule).strip()]


def _format_prompt_guidance_section(title: str, rules: list[str]) -> str:
    if not rules:
        return ""
    lines = ["", title]
    lines.extend(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))
    return "\n".join(lines) + "\n"


def print_prompt_guidance_summary(runtime_goal: dict[str, Any]) -> None:
    advisor_rules = _prompt_guidance_rules(runtime_goal, "advisor_rules")
    codegen_rules = _prompt_guidance_rules(runtime_goal, "codegen_rules")
    print("\n========== Prompt 额外规则 ==========")
    print(f"策略顾问规则数量：{len(advisor_rules)}")
    print(f"代码生成规则数量：{len(codegen_rules)}")




PRE_RUN_AI_REVIEW_MEMORY_FILE = ROOT_DIR / "user_data" / "ai_memory" / "pre_run_ai_review.json"


def _pre_run_ai_review_cfg(runtime_goal: dict[str, Any]) -> dict[str, Any]:
    raw = runtime_goal.get("pre_run_ai_review", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "provider_role": str(raw.get("provider_role") or "strategy_advisor"),
        "save_file": str(raw.get("save_file") or "pre_run_ai_review.json"),
    }


def _latest_previous_run_dir(current_run_dir: Path) -> Path | None:
    run_dirs = [p for p in RESULT_ROOT.glob("run_*") if p.is_dir() and p.resolve() != current_run_dir.resolve()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return run_dirs[0] if run_dirs else None


def _log_repo_path_from_env() -> Path | None:
    raw = (os.getenv("LOG_REPO_PATH") or "/opt/freqtrade-ai-logs").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() and path.is_dir() else None


def _latest_previous_log_repo_run_dir(current_run_dir: Path) -> Path | None:
    repo_path = _log_repo_path_from_env()
    if repo_path is None:
        return None
    logs_root = repo_path / "logs"
    if not logs_root.exists():
        return None
    current_name = current_run_dir.name
    run_dirs = [p for p in logs_root.glob("*/*") if p.is_dir() and p.name != current_name]
    run_dirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return run_dirs[0] if run_dirs else None


def _summarize_pre_run_source_file(label: str, data: Any) -> Any:
    if label.endswith("iteration_stats.json") and isinstance(data, dict):
        return _strip_random_samples_for_ai_prompt({
            "planned_iterations": data.get("planned_iterations"),
            "generated_versions_count": data.get("generated_versions_count"),
            "valid_strategy_count": data.get("valid_strategy_count"),
            "invalid_strategy_count": data.get("invalid_strategy_count"),
            "new_best_update_count": data.get("new_best_update_count"),
            "early_stop_triggered": data.get("early_stop_triggered"),
            "early_stop_reason": data.get("early_stop_reason"),
            "early_stop_checked_counters": data.get("early_stop_checked_counters"),
            "version_statuses": data.get("version_statuses", [])[-8:] if isinstance(data.get("version_statuses"), list) else [],
            "global_stats": data.get("global_stats"),
        })
    if label.endswith("leaderboard.json"):
        rows = data.get("items", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        compact_rows = []
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            compact_rows.append({
                "version": row.get("version"),
                "strategy_class": row.get("strategy_class"),
                "is_valid": row.get("is_valid"),
                "is_best": row.get("is_best"),
                "invalid_reason": row.get("invalid_reason"),
                "not_best_reason": row.get("not_best_reason"),
                "final_score": row.get("final_score"),
                "train_profit_pct": row.get("train_profit_pct"),
                "profit_factor": row.get("profit_factor"),
                "avg_validation_profit_pct": row.get("avg_validation_profit_pct"),
                "avg_validation_profit_factor": row.get("avg_validation_profit_factor"),
                "max_validation_drawdown_pct": row.get("max_validation_drawdown_pct"),
                "total_trades": row.get("total_trades"),
                "failure_reason": row.get("failure_reason"),
                "official_best_hard_gate_reasons": row.get("official_best_hard_gate_reasons"),
            })
        return {"items": compact_rows}
    if label.endswith("last_run_summary.json") and isinstance(data, dict):
        return _strip_random_samples_for_ai_prompt({
            "run_id": data.get("run_id"),
            "best_updated": data.get("best_updated"),
            "official_best": data.get("official_best"),
            "nearest_candidate": data.get("nearest_candidate"),
            "session_best": data.get("session_best"),
            "common_failure_patterns": data.get("common_failure_patterns"),
            "recommended_next_mutation_types": data.get("recommended_next_mutation_types"),
            "forbidden_next_mutation_types": data.get("forbidden_next_mutation_types"),
            "round_history": data.get("round_history", [])[-5:] if isinstance(data.get("round_history"), list) else [],
        })
    if label.endswith("recommended_pairs.json") and isinstance(data, dict):
        return {
            "created_at": data.get("created_at"),
            "source_strategy": data.get("source_strategy"),
            "train_period": data.get("train_period"),
            "validation_periods": data.get("validation_periods"),
            "active_pairs": data.get("active_pairs", []),
            "watch_pairs": data.get("watch_pairs", []),
            "cooldown_pairs": data.get("cooldown_pairs", []),
        }
    if label.endswith("pair_leaderboard.json") and isinstance(data, dict):
        rows = data.get("items", []) if isinstance(data.get("items"), list) else []
        return {
            "created_at": data.get("created_at"),
            "source_strategy": data.get("source_strategy"),
            "train_period": data.get("train_period"),
            "validation_periods": data.get("validation_periods"),
            "top_pairs": rows[:10],
            "bottom_pairs": rows[-5:] if len(rows) > 5 else [],
        }
    return _strip_random_samples_for_ai_prompt(data)


def _load_pre_run_review_sources(current_run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    log_repo_run_dir = _latest_previous_log_repo_run_dir(current_run_dir)
    if log_repo_run_dir:
        source_paths: list[tuple[str, Path | None]] = [
            ("日志仓库上一轮 iteration_stats.json", log_repo_run_dir / ITERATION_STATS_FILE_NAME),
            ("日志仓库上一轮 leaderboard.json", log_repo_run_dir / "leaderboard.json"),
            ("日志仓库上一轮 pair_leaderboard.json", log_repo_run_dir / "pair_leaderboard.json"),
            ("日志仓库上一轮 recommended_pairs.json", log_repo_run_dir / "recommended_pairs.json"),
            ("日志仓库上一轮 last_run_summary.json", log_repo_run_dir / "last_run_summary.json"),
            ("日志仓库上一轮 best_strategy_snapshot.json", log_repo_run_dir / "best_strategy_snapshot.json"),
            ("日志仓库上一轮 nearest_candidate_snapshot.json", log_repo_run_dir / "nearest_candidate_snapshot.json"),
            ("日志仓库上一轮 pre_run_ai_review.json", log_repo_run_dir / "pre_run_ai_review.json"),
        ]
        loaded: dict[str, Any] = {
            "source_priority": "log_repo",
            "previous_log_repo_run_dir": str(log_repo_run_dir),
            "files": {},
        }
        missing: list[str] = []
        for label, path in source_paths:
            if path is None or not path.exists():
                missing.append(label)
                loaded["files"][label] = {"status": "missing", "path": str(path) if path else ""}
                continue
            try:
                data = _summarize_pre_run_source_file(path.name, read_json(path))
                loaded["files"][label] = {"status": "loaded", "path": str(path), "data": data}
            except Exception as exc:  # noqa: BLE001 - review must never block optimization.
                missing.append(f"{label}（读取失败：{exc}）")
                loaded["files"][label] = {"status": "error", "path": str(path), "error": str(exc)}
        return loaded, missing

    previous_run_dir = _latest_previous_run_dir(current_run_dir)
    source_paths: list[tuple[str, Path | None]] = [
        ("user_data/ai_memory/last_run_summary.json", LAST_RUN_SUMMARY_FILE),
        ("user_data/ai_memory/best_strategy.json", BEST_STRATEGY_FILE),
        ("user_data/ai_memory/nearest_candidate.json", NEAREST_CANDIDATE_FILE),
        ("user_data/ai_memory/recommended_pairs.json", RECOMMENDED_PAIRS_FILE),
        ("user_data/ai_memory/pair_leaderboard.json", PAIR_LEADERBOARD_FILE),
        ("user_data/ai_memory/strategy_lessons.json", LESSONS_FILE),
        ("user_data/ai_memory/strategy_blacklist.json", BLACKLIST_FILE),
        ("上一次 run 的 iteration_stats.json", (previous_run_dir / ITERATION_STATS_FILE_NAME) if previous_run_dir else None),
        ("上一次 run 的 leaderboard.json", (previous_run_dir / "leaderboard.json") if previous_run_dir else None),
    ]
    loaded: dict[str, Any] = {
        "previous_run_dir": str(previous_run_dir) if previous_run_dir else "",
        "files": {},
    }
    missing: list[str] = []
    for label, path in source_paths:
        if path is None or not path.exists():
            missing.append(label)
            loaded["files"][label] = {"status": "missing", "path": str(path) if path else ""}
            continue
        try:
            data = read_json(path)
            data = _summarize_pre_run_source_file(path.name, data)
            loaded["files"][label] = {"status": "loaded", "path": str(path), "data": data}
        except Exception as exc:  # noqa: BLE001 - review must never block optimization.
            missing.append(f"{label}（读取失败：{exc}）")
            loaded["files"][label] = {"status": "error", "path": str(path), "error": str(exc)}
    return loaded, missing


def _build_pre_run_ai_review_prompt(runtime_goal: dict[str, Any], sources: dict[str, Any], missing: list[str]) -> str:
    trade_target_text = _runtime_trade_target_text(runtime_goal)
    trade_target_values = _runtime_trade_target_values(runtime_goal)
    return (
        "你是 Freqtrade 策略优化的 strategy_advisor。现在不是生成 mutation_spec，也不要输出 Python 代码。\n"
        "任务：在本次 run 第 1 轮生成 mutation_spec 之前，复盘上一次 run 和长期记忆，总结可复用经验。\n"
        "重要限制：本复盘只能作为后续 advisor/codegen prompt 的参考，不得覆盖 official_best、historical_best、best_strategy.json、nearest_candidate.json；所有 best 判断仍必须基于真实回测指标。\n"
        "如果输入文件缺失，请在分析中体现缺失导致的不确定性，不要编造不存在的数据。\n"
        f"交易数建议必须使用：{trade_target_text}\n"
        "只输出一个 JSON object，不要 Markdown，不要解释，结构必须为：\n"
        "{\n"
        "  \"last_run_diagnosis\": {\n"
        "    \"overall_result\": \"上一轮是否有实质突破\",\n"
        "    \"best_candidate\": \"上一轮最值得参考的失败或成功版本\",\n"
        "    \"main_failure_patterns\": [],\n"
        "    \"dangerous_directions_to_avoid\": [],\n"
        "    \"promising_directions_to_continue\": []\n"
        "  },\n"
        "  \"next_run_guidance\": {\n"
        "    \"preferred_parent\": \"official_best 或 nearest_candidate 或 last near miss\",\n"
        "    \"preferred_mutation_types\": [],\n"
        "    \"forbidden_mutation_types\": [],\n"
        f"    \"trade_count_target\": \"{trade_target_values['min_trades']}~{trade_target_values['max_trades']}，理想 {trade_target_values['ideal_min_trades']}~{trade_target_values['ideal_max_trades']}\",\n"
        "    \"must_fix\": [],\n"
        "    \"do_not_do\": []\n"
        "  },\n"
        "  \"codegen_guidance\": {\n"
        "    \"must_implement\": [],\n"
        "    \"avoid_equivalent_code\": [],\n"
        "    \"required_detectable_changes\": []\n"
        "  }\n"
        "}\n\n"
        "========== 当前 optimization_goal 摘要 ==========\n"
        + _compact_prompt_json({
            "target": runtime_goal.get("target", {}),
            "train_period": runtime_goal.get("train_period", {}),
            "validation_periods": runtime_goal.get("validation_periods", []),
            "prompt_guidance": runtime_goal.get("prompt_guidance", {}),
            "pair_selection": runtime_goal.get("pair_selection", {}),
        }, 5000)
        + "\n\n========== 已读取的复盘输入数据 ==========\n"
        + _compact_prompt_json(sources, 24000)
        + "\n\n========== 缺失或不可用文件 ==========\n"
        + ("\n".join(f"- {item}" for item in missing) if missing else "无")
        + "\n"
    )


def _normalize_pre_run_ai_review(data: dict[str, Any], runtime_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    data.setdefault("last_run_diagnosis", {})
    data.setdefault("next_run_guidance", {})
    data.setdefault("codegen_guidance", {})
    diagnosis = data["last_run_diagnosis"] if isinstance(data.get("last_run_diagnosis"), dict) else {}
    next_guidance = data["next_run_guidance"] if isinstance(data.get("next_run_guidance"), dict) else {}
    codegen_guidance = data["codegen_guidance"] if isinstance(data.get("codegen_guidance"), dict) else {}
    for key in ["main_failure_patterns", "dangerous_directions_to_avoid", "promising_directions_to_continue"]:
        diagnosis.setdefault(key, [])
    for key in ["preferred_mutation_types", "forbidden_mutation_types", "must_fix", "do_not_do"]:
        next_guidance.setdefault(key, [])
    if runtime_goal is not None:
        values = _runtime_trade_target_values(runtime_goal)
        next_guidance.setdefault("trade_count_target", f"{values['min_trades']}~{values['max_trades']}，理想 {values['ideal_min_trades']}~{values['ideal_max_trades']}")
    else:
        next_guidance.setdefault("trade_count_target", "使用 runtime_goal.target 的最终交易数目标")
    for key in ["must_implement", "avoid_equivalent_code", "required_detectable_changes"]:
        codegen_guidance.setdefault(key, [])
    data["last_run_diagnosis"] = diagnosis
    data["next_run_guidance"] = next_guidance
    data["codegen_guidance"] = codegen_guidance
    return data


def _pre_run_review_bullets(review: dict[str, Any], section: str, keys: list[str], limit: int = 5) -> list[str]:
    root = review.get(section, {}) if isinstance(review, dict) else {}
    if not isinstance(root, dict):
        return []
    bullets: list[str] = []
    for key in keys:
        value = root.get(key)
        if isinstance(value, list):
            bullets.extend(str(item) for item in value if str(item).strip())
        elif value:
            bullets.append(f"{key}: {value}")
    return bullets[:limit]


def _format_pre_run_review_for_advisor(review: dict[str, Any] | None) -> str:
    if not review:
        return ""
    return (
        "\n========== 本次运行开始前 AI 复盘总结 ==========\n"
        "内容来自 pre_run_ai_review.json。该复盘只作为经验参考，不得覆盖 official_best/historical_best/best_strategy/nearest_candidate；best 判断必须基于真实回测指标。\n"
        + _compact_prompt_json(review, 8000)
        + "\n"
    )


def _format_pre_run_review_for_codegen(review: dict[str, Any] | None) -> str:
    if not review:
        return ""
    return (
        "\n========== 本次运行开始前 AI 复盘总结：codegen_guidance ==========\n"
        "内容来自 pre_run_ai_review.json。只作为代码生成参考，必须仍以 mutation_spec 为准，并且必须产生可检测的真实策略结构变化。\n"
        + _compact_prompt_json(review.get("codegen_guidance", {}) if isinstance(review, dict) else {}, 4000)
        + "\n"
    )


def run_pre_run_ai_review(
    runtime_goal: dict[str, Any],
    run_dir: Path,
    advisor_runtime: AIRoleRuntime,
    code_runtime: AIRoleRuntime,
    ai_runtime_state: dict[str, Any],
) -> dict[str, Any] | None:
    cfg = _pre_run_ai_review_cfg(runtime_goal)
    save_file = cfg["save_file"] or "pre_run_ai_review.json"
    review_path = run_dir / save_file
    prompt_path = run_dir / "pre_run_ai_review_prompt.txt"
    raw_path = run_dir / "pre_run_ai_review.raw.txt"
    enabled = bool(cfg.get("enabled"))
    provider_role = str(cfg.get("provider_role") or "strategy_advisor")
    runtime = advisor_runtime if provider_role == "strategy_advisor" else code_runtime if provider_role == "code_generator" else advisor_runtime

    print("\n========== 运行前 AI 复盘 ==========")
    print(f"是否启用：{'是' if enabled else '否'}")
    print(f"使用 provider：{runtime.used_provider or '待调用'}")
    print(f"使用模型：{runtime.used_model or '待调用'}")
    print(f"复盘文件：{review_path}")
    if not enabled:
        print("主要结论：")
        print("- 未启用 pre_run_ai_review。")
        print("下一轮建议：")
        print("- 正常按现有记忆、prompt_guidance 和真实回测指标优化。")
        return None

    sources, missing = _load_pre_run_review_sources(run_dir)
    prompt = _build_pre_run_ai_review_prompt(runtime_goal, sources, missing)
    prompt_path.write_text(prompt, encoding="utf-8")
    try:
        raw_text = safe_ask_ai(runtime, [{"role": "user", "content": prompt}], state=ai_runtime_state)
        raw_path.write_text(raw_text or "", encoding="utf-8")
        review = _normalize_pre_run_ai_review(extract_json_object(raw_text))
        write_json(review_path, review)
        write_json(PRE_RUN_AI_REVIEW_MEMORY_FILE, review)
        print(f"使用 provider：{runtime.used_provider or '未知'}")
        print(f"使用模型：{runtime.used_model or '未知'}")
        print("主要结论：")
        bullets = _pre_run_review_bullets(review, "last_run_diagnosis", ["overall_result", "best_candidate", "main_failure_patterns"])
        for item in bullets or ["AI 复盘未给出明确主要结论。"]:
            print(f"- {item}")
        print("下一轮建议：")
        bullets = _pre_run_review_bullets(review, "next_run_guidance", ["preferred_parent", "preferred_mutation_types", "must_fix", "do_not_do"])
        for item in bullets or ["继续按真实回测指标小步优化。"]:
            print(f"- {item}")
        return review
    except Exception as exc:  # noqa: BLE001 - pre-run review is advisory only.
        error_review = {
            "last_run_diagnosis": {
                "overall_result": "pre_run_ai_review 调用失败，无法生成复盘。",
                "best_candidate": "",
                "main_failure_patterns": [],
                "dangerous_directions_to_avoid": [],
                "promising_directions_to_continue": [],
            },
            "next_run_guidance": {
                "preferred_parent": "",
                "preferred_mutation_types": [],
                "forbidden_mutation_types": [],
                "trade_count_target": _runtime_trade_target_text(runtime_goal),
                "must_fix": ["pre_run_ai_review 失败，继续使用 last_run_summary、best_strategy、nearest_candidate 和 prompt_guidance。"],
                "do_not_do": [],
            },
            "codegen_guidance": {
                "must_implement": [],
                "avoid_equivalent_code": [],
                "required_detectable_changes": [],
            },
        }
        raw_path.write_text(str(exc), encoding="utf-8")
        write_json(review_path, error_review)
        write_json(PRE_RUN_AI_REVIEW_MEMORY_FILE, error_review)
        print(f"使用 provider：{runtime.used_provider or '未知'}")
        print(f"使用模型：{runtime.used_model or '未知'}")
        print("主要结论：")
        print(f"- pre_run_ai_review 调用失败：{exc}")
        print("下一轮建议：")
        print("- 已记录失败原因，本次 run 继续正常优化，不中断。")
        return error_review


def _round_validation_key(item: dict[str, Any], index: int) -> str:
    return str(item.get("timerange") or item.get("period") or item.get("period_name") or item.get("label") or index)


def _behavior_duplicate_report(
    *,
    train_metrics: dict[str, Any],
    validation_metrics: list[dict[str, Any]],
    official_best: dict[str, Any] | None,
) -> dict[str, Any]:
    duplicate_with = _strategy_label(official_best, "")
    base_report = {
        "is_duplicate": False,
        "duplicate_with": duplicate_with,
        "reason": "",
    }
    if not isinstance(official_best, dict) or not official_best or not duplicate_with or duplicate_with == "baseline":
        return base_report

    official_train = official_best.get("train_metrics", {}) or {}
    official_validations = official_best.get("validation_metrics", []) or []
    if not validation_metrics:
        return base_report
    if not isinstance(official_validations, list) or len(validation_metrics) != len(official_validations):
        return base_report

    train_checks = [
        _safe_int(train_metrics.get("total_trades")) == _safe_int(official_train.get("total_trades")),
        abs(_safe_float(train_metrics.get("profit_total_pct")) - _safe_float(official_train.get("profit_total_pct"))) < 0.005,
        abs(_safe_float(train_metrics.get("profit_factor")) - _safe_float(official_train.get("profit_factor"))) < 0.005,
        abs(_safe_float(train_metrics.get("max_drawdown_pct")) - _safe_float(official_train.get("max_drawdown_pct"))) < 0.02,
    ]
    if not all(train_checks):
        return base_report

    current_by_key = {_round_validation_key(item, idx): item for idx, item in enumerate(validation_metrics)}
    official_by_key = {_round_validation_key(item, idx): item for idx, item in enumerate(official_validations)}
    if set(current_by_key) != set(official_by_key):
        return base_report

    for key, current_item in current_by_key.items():
        current_metrics = current_item.get("metrics", {}) or {}
        official_metrics = (official_by_key[key].get("metrics", {}) or {})
        if abs(_safe_int(current_metrics.get("total_trades")) - _safe_int(official_metrics.get("total_trades"))) > 1:
            return base_report
        if abs(_safe_float(current_metrics.get("profit_total_pct")) - _safe_float(official_metrics.get("profit_total_pct"))) >= 0.005:
            return base_report
        if abs(_safe_float(current_metrics.get("profit_factor")) - _safe_float(official_metrics.get("profit_factor"))) >= 0.01:
            return base_report

    return {
        "is_duplicate": True,
        "duplicate_with": duplicate_with,
        "reason": "训练和验证行为指标几乎一致",
    }


def _round_summary_label(summary: dict[str, Any] | None) -> str:
    if not isinstance(summary, dict) or not summary:
        return "无"
    version = summary.get("version") or summary.get("iteration") or "未知轮次"
    valid_text = "有效" if summary.get("is_valid") else "无效"
    reason = summary.get("invalid_reason") or summary.get("failure_reason") or "通过"
    return f"{version}，{valid_text}，原因 {reason}"


def _append_unique(items: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


def _strategy_record_from_round(
    *,
    version: str,
    class_name: str,
    strategy_file: Path,
    train_metrics: dict[str, Any] | None,
    validation_metrics: list[dict[str, Any]] | None,
    final_score: float,
    is_valid: bool,
    invalid_reason: str,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train = train_metrics or {}
    return {
        "version": version,
        "strategy_class": class_name,
        "strategy_file": str(strategy_file),
        "train_metrics": train,
        "validation_metrics": validation_metrics or [],
        "final_score": final_score,
        "is_valid": is_valid,
        "invalid_reason": invalid_reason,
        "features": features or {},
    }


def _nearest_score_for_session(candidate: dict[str, Any] | None, target_cfg: dict[str, Any], baseline_cfg: dict[str, Any]) -> tuple[float, float, float, float, float]:
    if not isinstance(candidate, dict) or not candidate:
        return (1e18, 1e18, 1e18, 1e18, 1e18)
    train = candidate.get("train_metrics", {}) or {}
    total_trades = _safe_int(train.get("total_trades") if "total_trades" in train else candidate.get("total_trades"))
    if total_trades <= 0:
        return (1e18, 1e18, 1e18, 1e18, 1e18)
    min_trades = _safe_int(target_cfg.get("min_trades", 25))
    max_trades = _safe_int(target_cfg.get("max_trades", 80))
    target_dd = _safe_float(target_cfg.get("max_drawdown_pct", 3.0))
    baseline_profit = _safe_float(baseline_cfg.get("profit_total_pct"))
    profit_pct = _safe_float(train.get("profit_total_pct") if "profit_total_pct" in train else candidate.get("train_profit_pct"))
    profit_factor = _safe_float(train.get("profit_factor") if "profit_factor" in train else candidate.get("profit_factor"))
    drawdown_pct = _safe_float(train.get("max_drawdown_pct") if "max_drawdown_pct" in train else candidate.get("max_drawdown_pct"))
    trade_penalty = max(0, min_trades - total_trades) + max(0, total_trades - max_trades)
    return (
        max(0.0, abs(profit_pct - baseline_profit)),
        max(0.0, drawdown_pct - target_dd),
        float(trade_penalty),
        -profit_factor,
        -_safe_float(candidate.get("final_score")),
    )


def _is_better_session_nearest(candidate: dict[str, Any] | None, current: dict[str, Any] | None, target_cfg: dict[str, Any], baseline_cfg: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or not candidate:
        return False
    return _nearest_score_for_session(candidate, target_cfg, baseline_cfg) < _nearest_score_for_session(current, target_cfg, baseline_cfg)


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


def _worst_validation_profit_factor(validation_metrics: list[dict[str, Any]]) -> float:
    if not validation_metrics:
        return 0.0
    values = [_safe_float((item.get("metrics", {}) or {}).get("profit_factor")) for item in validation_metrics]
    return min(values) if values else 0.0


def _official_best_hard_gate_reasons(
    *,
    train_metrics: dict[str, Any],
    validation_metrics: list[dict[str, Any]],
    avg_validation_profit_pct: float,
    avg_validation_profit_factor: float,
    max_validation_drawdown_pct: float,
    current_official_best: dict[str, Any] | None,
) -> list[str]:
    """Return reasons that forbid replacing best_strategy.json."""
    reasons: list[str] = []
    train_profit_total_pct = _safe_float(train_metrics.get("profit_total_pct"))
    train_profit_factor = _safe_float(train_metrics.get("profit_factor"))
    worst_validation_profit_factor = _worst_validation_profit_factor(validation_metrics)
    official_avg = dict((current_official_best or {}).get("avg_validation_metrics", {}) or {})
    official_validations = (current_official_best or {}).get("validation_metrics", []) or []
    official_is_real = bool(current_official_best and _strategy_label(current_official_best, "") != "baseline")
    if official_validations:
        off_avg_profit, _off_avg_pf, off_max_dd = _aggregate_validation_metrics(official_validations)
        official_avg.setdefault("profit_total_pct", off_avg_profit)
        official_avg.setdefault("max_drawdown_pct", off_max_dd)
    official_avg_validation_profit_pct = _safe_float(official_avg.get("profit_total_pct"))
    official_max_validation_drawdown_pct = _safe_float(official_avg.get("max_drawdown_pct"))

    checks = [
        (train_profit_total_pct >= 0.0, f"official best 硬门槛未通过：train_profit_total_pct={train_profit_total_pct:.4f} < 0"),
        (train_profit_factor >= 1.0, f"official best 硬门槛未通过：train_profit_factor={train_profit_factor:.4f} < 1.0"),
        (avg_validation_profit_pct >= 0.0, f"official best 硬门槛未通过：avg_validation_profit_pct={avg_validation_profit_pct:.4f} < 0"),
        (avg_validation_profit_factor >= 1.0, f"official best 硬门槛未通过：avg_validation_profit_factor={avg_validation_profit_factor:.4f} < 1.0"),
        (worst_validation_profit_factor >= 0.7, f"official best 硬门槛未通过：worst_validation_profit_factor={worst_validation_profit_factor:.4f} < 0.7"),
    ]
    reasons.extend(reason for ok, reason in checks if not ok)
    if official_is_real:
        allowed_dd = official_max_validation_drawdown_pct + 0.2
        if max_validation_drawdown_pct > allowed_dd:
            reasons.append(
                f"official best 硬门槛未通过：max_validation_drawdown_pct={max_validation_drawdown_pct:.4f} > 当前 official_best.max_validation_drawdown_pct+0.2={allowed_dd:.4f}"
            )
    if official_is_real and avg_validation_profit_pct < official_avg_validation_profit_pct:
        reasons.append(
            f"official best 硬门槛未通过：avg_validation_profit_pct={avg_validation_profit_pct:.4f} 低于当前 official_best={official_avg_validation_profit_pct:.4f}"
        )
    if official_is_real and official_avg_validation_profit_pct >= 0.0 and avg_validation_profit_pct < 0.0:
        reasons.append("official best 硬门槛未通过：当前 official_best 验证平均收益非负，新策略验证平均收益为负")
    return reasons



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
        "strategy_fingerprint": None,
        "duplicate_report": None,
        "behavior_duplicate": {"is_duplicate": False, "duplicate_with": "", "reason": ""},
        "is_behavior_duplicate": False,
        "not_best_reason": "",
        "final_score": 0,
        "score_breakdown": {},
        "invalid_reason": "",
        "is_valid": False,
        "is_best": False,
        "trade_under_min": False,
        "cannot_be_official_best_unless_validation_strong": False,
        "validation_strong": False,
        "trade_count_warning": "",
        "backtest_errors": [],
        "random_sample_metrics": [],
        "random_sample_usage": dict(RANDOM_SAMPLE_USAGE),
        "random_sample_observation": summarize_random_sample_observation([]),
        "random_sample_errors": [],
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
        "strategy_fingerprint": state.get("strategy_fingerprint"),
        "duplicate_report": state.get("duplicate_report"),
        "behavior_duplicate": state.get("behavior_duplicate", {"is_duplicate": False, "duplicate_with": "", "reason": ""}),
        "is_behavior_duplicate": bool(state.get("is_behavior_duplicate", False)),
        "not_best_reason": state.get("not_best_reason", ""),
        "trade_under_min": bool(state.get("trade_under_min", False)),
        "cannot_be_official_best_unless_validation_strong": bool(state.get("cannot_be_official_best_unless_validation_strong", False)),
        "validation_strong": bool(state.get("validation_strong", False)),
        "trade_count_warning": state.get("trade_count_warning", ""),
        "backtest_errors": state.get("backtest_errors", []),
        "random_sample_metrics": state.get("random_sample_metrics", []),
        "random_sample_usage": state.get("random_sample_usage", dict(RANDOM_SAMPLE_USAGE)),
        "random_sample_observation": state.get("random_sample_observation", summarize_random_sample_observation([])),
        "random_sample_errors": state.get("random_sample_errors", []),
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

    def _extract_block(pattern: str) -> str | None:
        m = re.search(pattern, strategy_code, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        return "\n".join(line.strip() for line in m.group(1).splitlines() if line.strip())[:2000]

    def _norm_list(tokens: list[str]) -> list[str]:
        return sorted(set(t.strip().lower() for t in tokens if t and t.strip()))

    indicators = _norm_list(re.findall(r"\b(rsi|ema\d*|macd|bbands|bollinger|adx|atr|volume|sma\d*|stoch|mfi|cci|roc|willr|vwma|vwap)\b", lc))
    entry_conditions = _norm_list(re.findall(r"\((df\[.*?\]\s*[<>=!]+\s*[^\)]+)\)", strategy_code, flags=re.DOTALL))
    entry_tags = _norm_list(re.findall(r"[\"']([A-Za-z0-9_\-]+)[\"']\s*(?:,|\)|\])", _extract(r"entry_tag\s*=\s*(.*)") or ""))
    pair_filters = _norm_list(re.findall(r"(?:metadata\.get\([\"']pair[\"']\)|metadata\[[\"']pair[\"']\]|pair\s+in\s+\[[^\]]+\]|pair\s*==\s*[\"'][^\"']+[\"'])", strategy_code, flags=re.IGNORECASE | re.DOTALL))
    protection_cooldown = _extract_block(r"def\s+protections\s*\([^)]*\).*?:\s*(.*?)(?:\n\s*def\s+|\n\s*class\s+|\Z)")
    cooldown_tokens = _norm_list(re.findall(r"\b(cooldown|stoplossguard|maxdrawdown|lowprofitpairs|protection|protections|cooldownperiod)\b", lc))

    return {
        "minimal_roi": _extract(r"minimal_roi\s*=\s*(\{.*?\})"),
        "stoploss": _extract(r"stoploss\s*=\s*([-\d\.]+)"),
        "trailing_stop": _extract(r"trailing_stop\s*=\s*(True|False)"),
        "use_exit_signal": _extract(r"use_exit_signal\s*=\s*(True|False)"),
        "timeframe": _extract(r"timeframe\s*=\s*[\"']([^\"']+)[\"']"),
        "indicators": indicators,
        "entry_conditions": entry_conditions[:30],
        "entry_tags": entry_tags[:12],
        "pair_filters": pair_filters[:20],
        "protection_cooldown": protection_cooldown,
        "cooldown_tokens": cooldown_tokens,
    }


def _strategy_fingerprint_payload(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "indicators": features.get("indicators", []),
        "entry_conditions": features.get("entry_conditions", []),
        "entry_tags": features.get("entry_tags", []),
        "minimal_roi": features.get("minimal_roi"),
        "stoploss": features.get("stoploss"),
        "trailing_stop": features.get("trailing_stop"),
        "use_exit_signal": features.get("use_exit_signal"),
        "timeframe": features.get("timeframe"),
        "pair_filters": features.get("pair_filters", []),
        "protection_cooldown": features.get("protection_cooldown"),
        "cooldown_tokens": features.get("cooldown_tokens", []),
    }


def build_strategy_fingerprint(features: dict[str, Any]) -> dict[str, Any]:
    payload = _strategy_fingerprint_payload(features)
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return {"hash": digest, "payload": payload}


def _feature_signature(features: dict[str, Any]) -> str:
    return build_strategy_fingerprint(features)["hash"]


def strategy_similarity(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, list[str]]:
    weights = {
        "entry_conditions": 0.28,
        "entry_tags": 0.12,
        "stoploss": 0.12,
        "minimal_roi": 0.12,
        "trailing_stop": 0.08,
        "use_exit_signal": 0.06,
        "timeframe": 0.06,
        "indicators": 0.10,
        "pair_filters": 0.04,
        "protection_cooldown": 0.02,
    }
    score = 0.0
    reasons: list[str] = []
    for k in ["stoploss", "minimal_roi", "trailing_stop", "use_exit_signal", "timeframe", "protection_cooldown"]:
        if str(a.get(k)) == str(b.get(k)) and a.get(k) is not None:
            score += weights[k]
            reasons.append(f"{k} 相同")
    for k in ["entry_conditions", "entry_tags", "indicators", "pair_filters"]:
        sa, sb = set(a.get(k, [])), set(b.get(k, []))
        if sa and sb:
            j = len(sa & sb) / max(1, len(sa | sb))
            score += weights[k] * j
            if j >= 0.66:
                reasons.append(f"{k} 相似")
    return min(1.0, score), reasons


def repeated_fingerprint_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    repeated: list[str] = []
    for field in _strategy_fingerprint_payload(a).keys():
        av = a.get(field)
        bv = b.get(field)
        if isinstance(av, list) or isinstance(bv, list):
            if set(av or []) == set(bv or []) and (av or bv):
                repeated.append(field)
        elif av is not None and str(av) == str(bv):
            repeated.append(field)
    return repeated


def _mutation_expected_changes(mutation_spec: dict[str, Any]) -> list[str]:
    mutation_type = str(mutation_spec.get("mutation_type") or "")
    expected: list[str] = []
    if mutation_type in {"pair_specific_filter", "tag_specific_filter", "add_entry_filter", "tighten_entry_trigger"}:
        expected.append("populate_entry_trend 中必须新增/收紧可检测 entry 条件")
    if mutation_type in {"pair_specific_filter", "tag_specific_filter"} or "ETH/USDT" in json.dumps(mutation_spec, ensure_ascii=False):
        expected.append("必须出现 ETH/USDT 明确分支或等价 pair-specific 条件")
    spec_text = json.dumps(mutation_spec, ensure_ascii=False).lower()
    indicator_names = ["ema20", "ema50", "adx", "atr_pct", "bollinger_middleband", "volume_mean_20"]
    required_indicators = [name for name in indicator_names if name.lower() in spec_text]
    if required_indicators:
        expected.append("populate_indicators 必须计算指标: " + ", ".join(required_indicators))
    structural_names = ["entry_conditions", "pair_filters", "indicators", "minimal_roi", "stoploss", "protection_cooldown"]
    expected.append("至少改变一个可检测策略结构: " + ", ".join(structural_names))
    return expected


def _detected_feature_changes(features: dict[str, Any], matched_features: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for field in _strategy_fingerprint_payload(features).keys():
        av = features.get(field)
        bv = matched_features.get(field)
        if isinstance(av, list) or isinstance(bv, list):
            added = sorted(set(av or []) - set(bv or []))
            removed = sorted(set(bv or []) - set(av or []))
            if added or removed:
                changes.append(f"{field}: added={added[:8]}, removed={removed[:8]}")
        elif str(av) != str(bv):
            changes.append(f"{field}: {bv!r} -> {av!r}")
    return changes


def build_duplicate_diff_summary(
    *,
    version: str,
    matched_version: str,
    similarity_score: float,
    mutation_type: str,
    expected_changes: list[str],
    detected_changes: list[str],
    repeated_fields: list[str],
    current_code: str,
    matched_code: str,
) -> str:
    lines = [
        f"duplicate_version: {version}",
        f"duplicate_with_version: {matched_version or 'unknown'}",
        f"similarity_score: {similarity_score:.4f}",
        f"mutation_type: {mutation_type or 'unknown'}",
        "",
        "expected_changes:",
        *(f"- {item}" for item in expected_changes),
        "",
        "detected_changes:",
        *(f"- {item}" for item in (detected_changes or ["无可检测结构变化"])),
        "",
        "repeated_fingerprint_fields:",
        *(f"- {item}" for item in repeated_fields),
        "",
        "unified_diff:",
    ]
    diff = difflib.unified_diff(
        matched_code.splitlines(),
        current_code.splitlines(),
        fromfile=f"{matched_version or 'matched'}/strategy.py",
        tofile=f"{version}/strategy.duplicate.py",
        lineterm="",
    )
    lines.extend(list(diff)[:400] or ["(代码 diff 为空)"])
    return "\n".join(lines) + "\n"


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
    timeout_attempts = [a for a in attempts if a.get("status") == "timeout"]
    if timeout_attempts and len(timeout_attempts) == len(attempts):
        return f"{display}模型池全部超时：{fallback_error}"
    if failed_attempts and all(int(a.get("status_code") or 0) == 403 and a.get("tos_blocked") for a in failed_attempts):
        return f"{display}模型池全部失败：403 provider TOS blocked"
    return f"{display}模型池全部失败：{fallback_error}"


def _recreate_provider_client(provider: dict[str, Any], timeout_sec: int) -> None:
    api_key = str(provider.get("api_key") or "")
    if not api_key:
        return
    base_url = str(provider.get("base_url") or "") or None
    provider["client"] = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)


def _close_provider_client(provider: dict[str, Any]) -> None:
    client = provider.get("client")
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _create_chat_completion_with_hard_timeout(client: OpenAI, *, model: str, messages: list[dict[str, str]], timeout_sec: int) -> Any:
    """Run the blocking OpenAI-compatible call in a daemon thread and stop waiting at timeout_sec."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(("ok", client.chat.completions.create(model=model, messages=messages, temperature=0.2)), block=False)
        except Exception as exc:  # noqa: BLE001 - propagated to the caller below.
            try:
                result_queue.put(("error", exc), block=False)
            except queue.Full:
                pass

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    try:
        status, payload = result_queue.get(timeout=timeout_sec)
    except queue.Empty as exc:
        raise TimeoutError(f"AI provider call exceeded {timeout_sec} seconds") from exc
    if status == "error":
        raise payload
    return payload


def safe_ask_ai(
    role_runtime: AIRoleRuntime,
    messages: list[dict[str, str]],
    state: dict[str, Any],
) -> str:
    role_runtime.begin_call()
    if role_runtime.role not in {"strategy_advisor", "code_generator"}:
        raise ValueError(f"safe_ask_ai 仅支持 strategy_advisor/code_generator，收到：{role_runtime.role}")
    provider_pool = [item for item in role_runtime.provider_pool if item.get("model") and item.get("client")]
    if not provider_pool:
        provider_pool = [{"name": "legacy_env", "model": model, "client": role_runtime.client, "base_url": ""} for model in role_runtime.model_pool if model]
    model_pool = [str(item.get("model") or "") for item in provider_pool if item.get("model")]
    if not provider_pool:
        raise AIModelPoolExhaustedError(f"{role_runtime.display_name}模型池为空", attempts=[])

    max_attempts = max(1, int(role_runtime.max_attempts_per_call or 1))
    timeout_sec = max(1, int(role_runtime.timeout_sec or 1))
    last_error_message = "未知错误"
    last_status_code: int | None = None

    for attempt_idx in range(max_attempts):
        start_offset = int(getattr(role_runtime, "forced_provider_offset", 0) or 0) % len(provider_pool)
        provider_idx = (start_offset + attempt_idx) % len(provider_pool) if role_runtime.switch_on_error else start_offset
        provider = provider_pool[provider_idx]
        next_provider = provider_pool[(provider_idx + 1) % len(provider_pool)] if len(provider_pool) > 1 else provider
        provider_name = str(provider.get("name") or "legacy_env")
        model = str(provider.get("model") or "")
        client = provider.get("client") or role_runtime.client
        next_model = str(next_provider.get("model") or model)
        next_provider_name = str(next_provider.get("name") or "legacy_env")
        now = time.time()
        last_call = float(state.get("last_ai_call_time", 0.0) or 0.0)
        cooldown = max(0.0, float(state.get("ai_call_cooldown_seconds", 0.0) or 0.0))
        elapsed_since_last = now - last_call if last_call > 0 else -1.0
        wait_before_call = max(0.0, cooldown - elapsed_since_last) if elapsed_since_last >= 0 else 0.0
        print("准备调用 AI：")
        print(f"角色：{role_runtime.role}")
        print(f"模型池：{', '.join(model_pool)}")
        print(f"当前 provider：{provider_name}")
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
                res = _create_chat_completion_with_hard_timeout(client, model=model, messages=messages, timeout_sec=timeout_sec)
            except TimeoutError as exc:
                elapsed = int(time.time() - start_ts)
                last_error_message = f"provider timeout after {timeout_sec} seconds"
                last_status_code = None
                role_runtime.attempts.append({
                    "provider": provider_name,
                    "model": model,
                    "status": "timeout",
                    "error": last_error_message,
                    "elapsed_seconds": elapsed,
                })
                _close_provider_client(provider)
                _recreate_provider_client(provider, timeout_sec)
                if provider.get("name") == "legacy_env":
                    for peer in provider_pool:
                        if peer.get("name") == "legacy_env":
                            _recreate_provider_client(peer, timeout_sec)
                    role_runtime.client = provider.get("client") or role_runtime.client
                print("AI provider 超时，已切换：")
                print(f"角色：{role_runtime.role}")
                print(f"provider：{provider_name}")
                print(f"模型：{model}")
                print(f"超时阈值：{timeout_sec} 秒；实际等待：{elapsed} 秒")
                if attempt_idx + 1 < max_attempts:
                    print(f"provider 超时，已切换到下一个 provider/模型：{next_provider_name}/{next_model}")
                    continue
                raise AIModelPoolExhaustedError(
                    _role_pool_failed_message(role_runtime.role, role_runtime.attempts, last_error_message),
                    status_code=None,
                    attempts=role_runtime.attempts,
                ) from exc
            except (InternalServerError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, TimeoutError, Exception) as exc:
                error_message, status_code = _format_ai_error(exc)
                last_error_message = error_message or exc.__class__.__name__
                last_status_code = status_code
                tos_block = _is_403_provider_tos_block(last_error_message, status_code)
                retriable = tos_block or _is_retriable_ai_error(last_error_message, status_code, exc)
                role_runtime.attempts.append({
                    "provider": provider_name,
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
                print(f"provider：{provider_name}")
                print(f"模型：{model}")
                print(f"错误：{last_error_message}")
                if attempt_idx + 1 < max_attempts:
                    print(f"将在 {wait_sec} 秒后切换到下一个 provider/模型：{next_provider_name}/{next_model}")
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
        role_runtime.used_provider = provider_name
        role_runtime.used_model = model
        role_runtime.attempts.append({"provider": provider_name, "model": model, "status": "success"})
        print("AI 调用成功：")
        print(f"角色：{role_runtime.role}")
        print(f"实际使用 provider：{provider_name}")
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


def _provider_env_prefix(provider_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider_name.strip()).strip("_").upper()
    return f"AI_PROVIDER_{normalized}"


def _looks_like_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return not lowered or lowered in {"你的key", "your_key", "your-api-key", "你的deepseek_key", "你的glm_key"} or lowered.startswith("你的")


def _build_provider_pool_from_env(provider_pool_env: str, timeout_sec: int) -> tuple[list[dict[str, Any]], list[str]]:
    providers: list[dict[str, Any]] = []
    skipped: list[str] = []
    for provider_name in _parse_model_pool(os.getenv(provider_pool_env)):
        prefix = _provider_env_prefix(provider_name)
        provider_type = (os.getenv(f"{prefix}_TYPE") or "openai_compatible").strip().lower()
        base_url = (os.getenv(f"{prefix}_BASE_URL") or "").strip() or None
        api_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
        model = (os.getenv(f"{prefix}_MODEL") or "").strip()
        if provider_type != "openai_compatible":
            skipped.append(f"{provider_name}: 不支持的 TYPE={provider_type}")
            continue
        if _looks_like_placeholder_secret(api_key) or not model or model.startswith("你的") or (base_url and str(base_url).strip().startswith("你的")):
            skipped.append(f"{provider_name}: 缺少 BASE_URL、API_KEY 或 MODEL")
            continue
        providers.append({
            "name": provider_name,
            "type": provider_type,
            "base_url": base_url or "",
            "api_key": api_key,
            "model": model,
            "client": OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec),
        })
    return providers, skipped


def _build_ai_role_runtime(
    model_cfg: dict[str, Any],
    role_name: str,
    timeout_sec: int,
    max_attempts_per_call: int,
    switch_on_error: bool,
) -> AIRoleRuntime:
    default_provider_pool_env_by_role = {
        "strategy_advisor": "STRATEGY_ADVISOR_PROVIDER_POOL",
        "code_generator": "STRATEGY_CODEGEN_PROVIDER_POOL",
        "code_repair": "STRATEGY_CODEGEN_PROVIDER_POOL",
    }
    provider_pool_env = str(model_cfg.get("provider_pool_env") or default_provider_pool_env_by_role.get(role_name, ""))
    provider_pool: list[dict[str, Any]] = []
    skipped_providers: list[str] = []
    if provider_pool_env and os.getenv(provider_pool_env):
        provider_pool, skipped_providers = _build_provider_pool_from_env(provider_pool_env, timeout_sec)
        if provider_pool:
            return AIRoleRuntime(
                role=role_name,
                client=provider_pool[0]["client"],
                model_pool=[str(item.get("model") or "") for item in provider_pool],
                timeout_sec=timeout_sec,
                switch_on_error=switch_on_error,
                max_attempts_per_call=max_attempts_per_call,
                provider_pool=provider_pool,
            )
        skipped_text = "；".join(skipped_providers) if skipped_providers else "未解析到有效 provider"
        raise RuntimeError(f"{provider_pool_env} 已设置，但 {role_name} 没有可用 provider：{skipped_text}")

    base_url = (os.getenv(str(model_cfg.get("base_url_env", ""))) or "").strip() or None
    api_key = (os.getenv(str(model_cfg.get("api_key_env", ""))) or "").strip()
    default_pool_env_by_role = {"strategy_advisor": "CLAUDE_MODEL_POOL", "code_generator": "OPENAI_MODEL_POOL", "code_repair": "OPENAI_MODEL_POOL"}
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
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
    legacy_provider_pool = [{
        "name": "legacy_env",
        "type": "openai_compatible",
        "base_url": base_url or "",
        "api_key": api_key,
        "model": model,
        "client": client,
    } for model in model_pool]
    return AIRoleRuntime(
        role=role_name,
        client=client,
        model_pool=model_pool,
        timeout_sec=timeout_sec,
        switch_on_error=switch_on_error,
        max_attempts_per_call=max_attempts_per_call,
        provider_pool=legacy_provider_pool,
    )


def _print_ai_model_pool_config(advisor_runtime: AIRoleRuntime, code_runtime: AIRoleRuntime, cooldown_seconds: float) -> None:
    def _print_pool(title: str, runtime: AIRoleRuntime) -> None:
        print(title)
        pool = runtime.provider_pool or [{"name": "legacy_env", "model": model, "base_url": ""} for model in runtime.model_pool]
        for idx, item in enumerate(pool, start=1):
            base_url = str(item.get("base_url") or "")
            base_url_text = f" @ {base_url}" if base_url else ""
            print(f"{idx}. {item.get('name', 'legacy_env')} / {item.get('model', '')}{base_url_text}")

    print("\n========== AI 模型池配置 ==========")
    _print_pool("策略顾问 provider/模型池：", advisor_runtime)
    print()
    _print_pool("代码生成 provider/模型池：", code_runtime)
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
        f"{_auto_trade_count_target_prompt_note(runtime_goal)}"
        "重点：减少固定止损吞噬 ROI，不是增加交易数量。禁止/不推荐：increase_trade_frequency,loosen_entry,enable_trailing,adjust_stoploss_only,widen_stoploss。优先：add_entry_filter,tighten_entry_trigger,remove_bad_entry_condition,pair_specific_filter,tag_specific_filter。"
    )



def _v045_small_step_prompt_section(champion: dict[str, Any], runtime_goal: dict[str, Any] | None = None) -> str:
    meta = champion.get("meta", {}) if isinstance(champion, dict) else {}
    name = str((meta or {}).get("strategy_class") or (meta or {}).get("class_name") or "")
    if "MultiCoin_AI_Strategy_20260530_014158_v045" not in name:
        return ""
    trade_target_text = _runtime_trade_target_text(runtime_goal) if runtime_goal is not None else "使用 runtime_goal.target 的最终交易数目标。"
    return (
        "\n========== v045 低频稳健阶段强约束 ==========\n"
        "当前正式 best 是 MultiCoin_AI_Strategy_20260530_014158_v045。下一轮必须围绕 v045 小步优化，禁止大改结构。\n"
        "v045 训练：25 笔，+0.0207 USDT，PF 1.0027，DD 0.66%。\n"
        "v045 验证：202604 +0.4342 USDT / PF 1.0764；202603 +0.7088 USDT / PF 1.0936；202602 +3.3647 USDT / PF 1.8883。\n"
        "只允许 mutation_type: pair_specific_filter, tag_specific_filter, remove_bad_entry_condition, small_entry_filter_adjustment。\n"
        "禁止: increase_trade_frequency, loosen_entry, widen_stoploss, enable_exit_signal, enable_trailing, 大幅修改 ROI。\n"
        f"目标：{trade_target_text} 202604/202603/202602 三个验证月继续全部不亏；最差验证 PF > 1.0；固定止损亏损降低；训练区间不要明显转负。\n"
        "如果必须选择系统允许枚举外的 small_entry_filter_adjustment，请在 mutation_type 用 tighten_entry_trigger 或 add_entry_filter，并在 reason 明确它只是 small_entry_filter_adjustment。\n"
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


def _pair_selection_cfg(runtime_goal: dict[str, Any]) -> dict[str, Any]:
    raw = runtime_goal.get("pair_selection", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "candidate_pairs": [str(p).strip() for p in raw.get("candidate_pairs", []) if str(p).strip()] if isinstance(raw.get("candidate_pairs", []), list) else [],
        "min_pair_trades": int(raw.get("min_pair_trades", 8) or 8),
        "max_pair_drawdown_pct": float(raw.get("max_pair_drawdown_pct", 3) or 3),
        "min_pair_profit_factor": float(raw.get("min_pair_profit_factor", 0.9) or 0.9),
        "prefer_validation_profit_positive": bool(raw.get("prefer_validation_profit_positive", True)),
        "active_pair_count": int(raw.get("active_pair_count", 5) or 5),
        "watch_pair_count": int(raw.get("watch_pair_count", 5) or 5),
        "reevaluate_every_runs": int(raw.get("reevaluate_every_runs", 5) or 5),
    }


def _metric_for_pair(metrics: dict[str, Any], pair: str) -> dict[str, Any]:
    for item in _normalize_pair_metrics(metrics.get("pairs", []) or []):
        if item.get("pair") == pair:
            return item
    return {"pair": pair, "trades": 0, "profit_total_abs": 0.0, "profit_total_pct": 0.0, "profit_factor": 0.0, "winrate": 0.0, "max_drawdown_pct": 0.0}


def _pair_exit_profit_abs(result: dict[str, Any], pair: str, tokens: tuple[str, ...]) -> float:
    total = 0.0
    trades = result.get("trades", []) or result.get("results_per_trade", []) or []
    if not isinstance(trades, list):
        return 0.0
    lowered = tuple(t.lower() for t in tokens)
    for trade in trades:
        if not isinstance(trade, dict) or str(trade.get("pair") or "") != pair:
            continue
        reason = str(trade.get("exit_reason") or trade.get("sell_reason") or "").lower()
        if any(token in reason for token in lowered):
            total += _safe_float(trade.get("profit_abs"))
    return total


def _pair_metric_row(pair: str, train_metrics: dict[str, Any], validation_metrics: list[dict[str, Any]], train_result: dict[str, Any] | None = None) -> dict[str, Any]:
    train_pair = _metric_for_pair(train_metrics, pair)
    validation_pair_rows = []
    for item in validation_metrics:
        m = item.get("metrics", {}) if isinstance(item, dict) else {}
        validation_pair_rows.append(_metric_for_pair(m, pair))
    validation_profit_pcts = [_safe_float(v.get("profit_total_pct")) for v in validation_pair_rows]
    validation_pfs = [_safe_float(v.get("profit_factor")) for v in validation_pair_rows]
    roi_abs = _pair_exit_profit_abs(train_result or {}, pair, ("roi",))
    stoploss_abs = _pair_exit_profit_abs(train_result or {}, pair, ("stop", "stoploss", "stop_loss"))
    stoploss_to_roi_ratio = abs(stoploss_abs) / max(roi_abs, 1e-9) if roi_abs > 0 else (999.0 if stoploss_abs < 0 else 0.0)
    return {
        "pair": pair,
        "train_total_trades": _safe_int(train_pair.get("trades")),
        "train_profit_pct": _safe_float(train_pair.get("profit_total_pct")),
        "train_profit_abs": _safe_float(train_pair.get("profit_total_abs")),
        "train_profit_factor": _safe_float(train_pair.get("profit_factor")),
        "train_max_drawdown_pct": _safe_float(train_pair.get("max_drawdown_pct")),
        "train_roi_profit_abs": roi_abs,
        "train_stoploss_profit_abs": stoploss_abs,
        "validation_avg_profit_pct": sum(validation_profit_pcts) / len(validation_profit_pcts) if validation_profit_pcts else 0.0,
        "validation_avg_profit_factor": sum(validation_pfs) / len(validation_pfs) if validation_pfs else 0.0,
        "validation_worst_profit_pct": min(validation_profit_pcts) if validation_profit_pcts else 0.0,
        "validation_worst_profit_factor": min(validation_pfs) if validation_pfs else 0.0,
        "validation_max_drawdown_pct": max((_safe_float(v.get("max_drawdown_pct")) for v in validation_pair_rows), default=0.0),
        "stoploss_to_roi_ratio": stoploss_to_roi_ratio,
        "validation_periods": validation_pair_rows,
    }


def _score_pair(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, list[str], str]:
    score = 50.0
    reasons: list[str] = []
    train_profit = _safe_float(row.get("train_profit_pct"))
    avg_val = _safe_float(row.get("validation_avg_profit_pct"))
    worst_val = _safe_float(row.get("validation_worst_profit_pct"))
    train_pf = _safe_float(row.get("train_profit_factor"))
    avg_pf = _safe_float(row.get("validation_avg_profit_factor"))
    worst_pf = _safe_float(row.get("validation_worst_profit_factor"))
    trades = _safe_int(row.get("train_total_trades"))
    train_dd = _safe_float(row.get("train_max_drawdown_pct"))
    val_dd = _safe_float(row.get("validation_max_drawdown_pct"))
    sl_roi = _safe_float(row.get("stoploss_to_roi_ratio"))
    score += train_profit * 3.0 + avg_val * 6.0 + worst_val * 4.0
    score += min(train_pf, 3.0) * 8.0 + min(avg_pf, 3.0) * 10.0 + min(worst_pf, 3.0) * 6.0
    if avg_val < 0:
        score -= 35.0
        reasons.append("验证平均收益为负")
    if worst_val < -2.0:
        score -= 30.0
        reasons.append("最差验证区间严重亏损")
    elif worst_val < 0:
        score -= 12.0
        reasons.append("存在亏损验证区间")
    if sl_roi > 1.0:
        score -= min(30.0, 12.0 + (sl_roi - 1.0) * 8.0)
        reasons.append("固定止损亏损大于 ROI 收益")
    min_trades = int(cfg.get("min_pair_trades", 8))
    if trades < min_trades:
        score -= 30.0 + (min_trades - trades) * 2.0
        reasons.append("交易数太少")
    max_dd = float(cfg.get("max_pair_drawdown_pct", 3))
    if max(train_dd, val_dd) > max_dd:
        score -= 25.0 + (max(train_dd, val_dd) - max_dd) * 4.0
        reasons.append("回撤过高")
    if train_pf < float(cfg.get("min_pair_profit_factor", 0.9)):
        score -= 10.0
        reasons.append("训练 PF 偏低")
    if bool(cfg.get("prefer_validation_profit_positive", True)) and avg_val <= 0:
        score -= 12.0
    positives = []
    if avg_val > 0 and worst_val > -0.5:
        positives.append("验证表现稳定")
    if train_pf >= 1.0 and avg_pf >= 1.0:
        positives.append("PF 较好")
    if max(train_dd, val_dd) <= max_dd:
        positives.append("止损/回撤可控")
    if trades >= min_trades:
        positives.append("交易数达标")
    reason = "，".join(positives or reasons or ["综合表现中性"])
    return round(max(0.0, min(100.0, score)), 4), reasons, reason


def _best_strategy_ref() -> tuple[str, str]:
    if not BEST_STRATEGY_FILE.exists():
        raise FileNotFoundError(f"best_strategy.json 不存在：{BEST_STRATEGY_FILE}")
    best = read_json(BEST_STRATEGY_FILE)
    class_name = str(best.get("class_name") or best.get("strategy_class") or "").strip()
    strategy_file = str(best.get("strategy_file") or "").strip()
    if not class_name:
        raise RuntimeError("best_strategy.json 缺少 class_name/strategy_class")
    return class_name, strategy_file


def _run_pair_scan_backtest(config: str, class_name: str, timeframe: str, period: PeriodDef, run_dir: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cmd = ["docker", "compose", "run", "--rm", "freqtrade", "backtesting", "--config", config, "--strategy", class_name, "--timeframe", timeframe, "--timerange", period.timerange, "--export", "trades", "--cache", "none"]
    results_dir = ROOT_DIR / "user_data" / "backtest_results"
    before = set(_list_backtest_zips(results_dir))
    started_at = time.time()
    print(f"正在回测 {label}：{period.timerange}")
    cp = run_cmd(cmd, ROOT_DIR)
    _write_backtest_process_log(run_dir, f"pair_scan_{label}", period.timerange, cp)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"pair-scan 回测失败：{label} {period.timerange}")
    result_zip, candidates = find_backtest_zip_for_strategy(results_dir, class_name, started_at, before)
    if result_zip is None:
        _log_backtest_zip_filter_failure(class_name, candidates)
        raise RuntimeError(f"pair-scan 未找到当前策略回测 zip：{label}")
    local_zip = _copy_backtest_zip_to_version(result_zip, run_dir, f"pair_scan_{label}", period.timerange)
    result, actual_keys = parse_backtest_from_zip(local_zip, class_name, strict=False)
    if result is None:
        raise RuntimeError(f"pair-scan 回测结果解析失败：{label}，实际策略：{actual_keys}")
    metrics = _extract_metrics(result)
    return metrics, result


def print_pair_scan_summary(recommended: dict[str, Any], candidate_count: int) -> None:
    print("\n========== 交易对筛选结果 ==========")
    print(f"候选币种数量：{candidate_count}")
    print("active_pairs：" + (", ".join(item.get("pair", "") for item in recommended.get("active_pairs", [])) or "无"))
    print("watch_pairs：" + (", ".join(item.get("pair", "") for item in recommended.get("watch_pairs", [])) or "无"))
    print("cooldown_pairs：" + (", ".join(item.get("pair", "") for item in recommended.get("cooldown_pairs", [])) or "无"))
    print("推荐原因：")
    for item in recommended.get("active_pairs", []):
        print(f"- {item.get('pair')}: score={item.get('score')}，{item.get('reason')}")


def run_pair_scan(runtime_goal: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> None:
    cfg = _pair_selection_cfg(runtime_goal)
    if not cfg.get("enabled", True):
        raise RuntimeError("pair_selection.enabled=false，无法执行 pair-scan。")
    candidate_pairs = cfg.get("candidate_pairs", []) or []
    if not candidate_pairs:
        raise RuntimeError("optimization_goal.json 中 pair_selection.candidate_pairs 为空。")
    class_name, strategy_file = _best_strategy_ref()
    print("当前模式：交易对筛选 pair-scan")
    print("AI 调用：关闭")
    print(f"source_strategy：{class_name}")
    if strategy_file:
        print(f"策略文件：{strategy_file}")
    train, validations = _build_periods(runtime_goal)
    if not train.timerange:
        raise RuntimeError("缺少训练区间 train_period.timerange，无法继续。")
    base_config = str(runtime_goal.get("config", args.config))
    scan_config = _write_temp_config_with_pairs(base_config, list(candidate_pairs), run_dir, "pair_scan_candidates")
    runtime_goal["config"] = scan_config
    runtime_goal["pairs"] = list(candidate_pairs)
    maybe_download_data(runtime_goal, args, train.timerange)
    timeframe = str(runtime_goal.get("timeframe", args.timeframe))
    train_metrics, train_result = _run_pair_scan_backtest(scan_config, class_name, timeframe, train, run_dir, "train")
    validation_metrics: list[dict[str, Any]] = []
    for period in validations:
        vm, _ = _run_pair_scan_backtest(scan_config, class_name, timeframe, period, run_dir, period.name)
        validation_metrics.append({"period": period.name, "timerange": period.timerange, "metrics": vm})
    rows = []
    for pair in candidate_pairs:
        row = _pair_metric_row(pair, train_metrics, validation_metrics, train_result)
        score, penalty_reasons, reason = _score_pair(row, cfg)
        row.update({"score": score, "penalty_reasons": penalty_reasons, "reason": reason})
        rows.append(row)
    rows.sort(key=lambda item: _safe_float(item.get("score")), reverse=True)
    active_count = max(0, int(cfg.get("active_pair_count", 5)))
    watch_count = max(0, int(cfg.get("watch_pair_count", 5)))
    active_rows = rows[:active_count]
    watch_rows = rows[active_count:active_count + watch_count]
    cooldown_rows = rows[active_count + watch_count:]
    leaderboard = {
        "created_at": datetime.utcnow().isoformat(),
        "source_strategy": class_name,
        "source_strategy_file": strategy_file,
        "train_period": train.timerange,
        "validation_periods": [{"name": p.name, "timerange": p.timerange, "weight": p.weight} for p in validations],
        "candidate_pairs": list(candidate_pairs),
        "items": rows,
    }
    recommended = {
        "created_at": datetime.utcnow().isoformat(),
        "source_strategy": class_name,
        "train_period": train.timerange,
        "validation_periods": [p.timerange for p in validations],
        "active_pairs": [{"pair": r["pair"], "score": r["score"], "reason": r["reason"]} for r in active_rows],
        "watch_pairs": [{"pair": r["pair"], "score": r["score"], "reason": r["reason"]} for r in watch_rows],
        "cooldown_pairs": [{"pair": r["pair"], "score": r["score"], "reason": r["reason"]} for r in cooldown_rows],
        "pair_metrics": {r["pair"]: r for r in rows},
    }
    write_json(run_dir / "pair_leaderboard.json", leaderboard)
    write_json(run_dir / "recommended_pairs.json", recommended)
    write_json(PAIR_LEADERBOARD_FILE, leaderboard)
    write_json(RECOMMENDED_PAIRS_FILE, recommended)
    print_pair_scan_summary(recommended, len(candidate_pairs))
    print(f"pair_leaderboard.json：{run_dir / 'pair_leaderboard.json'}")
    print(f"recommended_pairs.json：{run_dir / 'recommended_pairs.json'}")


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


MANUAL_TASK_ROOT = ROOT_DIR / "ai_manual_tasks"
SENSITIVE_KEY_RE = re.compile(r"(?i)(OKX_API_KEY|OKX_API_SECRET|OKX_API_PASSPHRASE|OPENAI_API_KEY|CLAUDE_API_KEY|API[_-]?KEY|SECRET|PASSPHRASE)\s*[=:]\s*[^\s,'\"]+")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _sanitize_manual_task_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-/")
    if not cleaned:
        raise RuntimeError("manual task name 不能为空。")
    if cleaned in {".", ".."} or ".." in cleaned.split("/"):
        raise RuntimeError(f"manual task name 不安全：{name}")
    return cleaned


def _sanitize_for_manual_task(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(r"(?i)(api[_-]?key|secret|passphrase|token|password)", str(key)):
                out[str(key)] = "<redacted>"
            else:
                out[str(key)] = _sanitize_for_manual_task(item)
        return out
    if isinstance(value, list):
        return [_sanitize_for_manual_task(item) for item in value]
    if isinstance(value, str):
        return SENSITIVE_KEY_RE.sub(lambda m: m.group(1) + "=<redacted>", value)
    return value


def _write_manual_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_manual_task(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    data = _load_json_or_none(path)
    return data if isinstance(data, dict) else {}


def _copy_strategy_or_note(src_value: Any, dest: Path, label: str) -> bool:
    src = Path(str(src_value or "")).expanduser()
    if src and not src.is_absolute():
        src = ROOT_DIR / src
    if src.exists() and src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    dest.write_text(
        f"# {label} strategy file is unavailable.\n"
        f"# Expected source: {src_value or 'not set'}\n",
        encoding="utf-8",
    )
    return False


def _recent_strategy_memory_excerpt(limit: int = 50) -> dict[str, Any]:
    items = _read_json_list_file(MEMORY_FILE)[-limit:] if MEMORY_FILE.exists() else []
    return {"items": items, "limit": limit, "source": str(MEMORY_FILE)}


def _recent_leaderboard_entries(run_limit: int = 8, item_limit: int = 80) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    run_dirs = sorted(RESULT_ROOT.glob("run_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for rd in run_dirs[:run_limit]:
        path = rd / "leaderboard.json"
        if not path.exists():
            continue
        try:
            raw = read_json(path)
        except Exception:
            continue
        for item in (raw.get("items") if isinstance(raw, dict) else []) or []:
            entries.append({
                "run_id": item.get("run_id") or rd.name.replace("run_", ""),
                "version": item.get("version"),
                "strategy_class": item.get("strategy_class"),
                "final_score": item.get("final_score"),
                "is_valid": item.get("is_valid"),
                "is_best": item.get("is_best"),
                "invalid_reason": item.get("invalid_reason"),
                "train_profit_pct": item.get("train_profit_pct"),
                "avg_validation_profit_pct": item.get("avg_validation_profit_pct"),
                "profit_factor": item.get("profit_factor"),
                "max_drawdown_pct": item.get("max_drawdown_pct"),
                "total_trades": item.get("total_trades"),
                "mutation_type": item.get("mutation_type"),
            })
            if len(entries) >= item_limit:
                return {"items": entries, "run_limit": run_limit, "item_limit": item_limit}
    return {"items": entries, "run_limit": run_limit, "item_limit": item_limit}


def _latest_pair_entry_tag_summary() -> list[dict[str, Any]]:
    summaries = sorted(RESULT_ROOT.glob("run_*/v*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in summaries[:40]:
        try:
            data = read_json(path)
        except Exception:
            continue
        pairs = data.get("pair_metrics") or []
        tags = data.get("entry_tag_metrics") or []
        if pairs or tags:
            return [{"source": str(path), "pair_metrics": pairs, "entry_tag_metrics": tags}]
    return []


def _manual_readme_text(task_name: str, task_dir: Path, branch: str) -> str:
    return f"""# 半自动 AI 策略生成任务包：{task_name}

这是半自动策略生成任务包。
服务器不会在本步骤调用 AI。
请先执行 `codex_advisor_instruction.md`，生成 `generated_mutation_spec.json`。
再执行 `codex_codegen_instruction.md`，生成 `generated_strategy.py`。
生成策略后，服务器会拉取代码并执行回测。

## 推荐流程

1. 打开 Codex。
2. 让 Codex 读取本目录的 `codex_advisor_instruction.md`，只生成 `generated_mutation_spec.json`。
3. 再让 Codex 读取 `codex_codegen_instruction.md`，生成 `generated_strategy.py`。
4. 在服务器回到仓库执行：

```bash
git pull
python3 ai_tools/auto_optimize_strategy.py --goal ai_tools/optimization_goal.json --manual-ai-run {task_dir.as_posix()}/generated_strategy.py --manual-ai-task-dir {task_dir.as_posix()}
```

## Git 分支

建议分支：`{branch}`
"""


def _manual_advisor_instruction(runtime_goal: dict[str, Any], best: dict[str, Any], nearest: dict[str, Any], last_run: dict[str, Any]) -> str:
    target = runtime_goal.get("target", {}) or {}
    min_trades = target.get("min_trades", 25)
    max_trades = target.get("max_trades", 80)
    forbidden = last_run.get("forbidden_next_mutation_types") or ["disable_or_adjust_trailing", "扩大止损", "放宽高频入场"]
    problems = last_run.get("current_main_problems") or last_run.get("common_failure_patterns") or ["高频风险", "固定止损亏损吞噬 ROI 收益", "验证区间稳健性不足"]
    return f"""你是 strategy_advisor。请读取本目录上下文文件，只输出 `generated_mutation_spec.json`，不要输出 Python 策略代码。

必须使用的上下文文件：
- `optimization_goal.snapshot.json`
- `current_best_summary.json`
- `nearest_candidate_summary.json`
- `last_run_summary.json`
- `strategy_lessons.json`
- `strategy_blacklist.json`
- `strategy_memory_excerpt.json`
- `leaderboard_recent.json`
- `pair_entry_tag_summary.json`

当前目标交易数：训练区间总交易数 {min_trades}~{max_trades}（不是单币种）。
当前历史 best 指标：`current_best_summary.json`，关键字段：{json.dumps(best.get('train_metrics', {}) or {}, ensure_ascii=False)[:1200]}
nearest_candidate 指标：`nearest_candidate_summary.json`，关键字段：{json.dumps(nearest.get('train_metrics', {}) or {}, ensure_ascii=False)[:1200]}
上轮失败模式：{json.dumps(last_run.get('common_failure_patterns') or last_run.get('last_failure_modes') or last_run.get('for_advisor_next_round') or [], ensure_ascii=False)[:1200]}
当前主要问题：{json.dumps(problems, ensure_ascii=False)}
禁止方向：{json.dumps(forbidden, ensure_ascii=False)}
推荐 mutation_type：优先从 `add_entry_filter`, `tighten_entry_trigger`, `remove_bad_entry_condition`, `pair_specific_filter`, `tag_specific_filter`, `adjust_roi`, `adjust_stoploss`, `reduce_trade_frequency` 中选择一个。

硬性要求：
- 本轮只能做单点小步修改。
- 不允许生成策略代码。
- 不允许启用 exit_signal。
- 不允许做空、杠杆、加仓、马丁格尔。
- 不允许为了交易数而宽松堆叠 OR 造成高频。

只输出以下 JSON object，并写入 `generated_mutation_spec.json`：

```json
{{
  "session_parent_choice": "historical_best | nearest_candidate | baseline",
  "session_parent_reason": "...",
  "mutation_type": "...",
  "goal": "...",
  "changes": [
    {{
      "target": "entry_filter / roi / stoploss / pair_filter / tag_filter",
      "action": "...",
      "reason": "..."
    }}
  ],
  "risk_controls": ["..."],
  "do_not_change": ["..."],
  "expected_effect": {{
    "trade_count": "...",
    "profit_factor": "...",
    "drawdown": "...",
    "stoploss_loss": "..."
  }}
}}
```
"""


def _manual_codegen_instruction() -> str:
    return """你是 code_generator。请读取：
- `generated_mutation_spec.json`
- `current_best_strategy.py`
- `nearest_candidate_strategy.py`
- `optimization_goal.snapshot.json`

然后只生成一个文件：`generated_strategy.py`。

要求：
- 必须是完整可运行 Freqtrade Strategy。
- 只能 long-only spot。
- `can_short = False`。
- `use_exit_signal = False`。
- 不主动使用 `populate_exit_trend` 产生 `exit_long` 信号；如果框架需要该函数，只返回 dataframe，不设置 exit_long=1。
- 不使用杠杆。
- 不使用加仓。
- 不使用马丁格尔。
- 不引入外部网络请求。
- 策略类名先可用占位：`Manual_AI_Generated_Strategy`。
- 服务器导入时会自动重命名为唯一类名。
- 只根据 mutation_spec 做单点小步修改，不要完全重写父策略。
- 输出完整 Python 文件内容，不要输出解释性文字。
"""


def _scan_manual_task_for_sensitive_text(task_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in task_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if SENSITIVE_KEY_RE.search(text):
            hits.append(str(path.relative_to(ROOT_DIR)))
    return hits


def run_manual_git_push(task_dir: Path, task_name: str, branch: str) -> None:
    print("========== Git 自动提交推送 ==========")
    rel_task = task_dir.relative_to(ROOT_DIR)
    commands = [
        ["git", "checkout", "-B", branch],
        ["git", "add", str(rel_task)],
    ]
    for cmd in commands:
        cp = run_cmd(cmd, ROOT_DIR)
        if cp.stdout:
            print(cp.stdout)
        if cp.stderr:
            print(cp.stderr)
        if cp.returncode != 0:
            raise RuntimeError(f"Git 命令失败：{' '.join(cmd)}")
    diff_cp = run_cmd(["git", "diff", "--cached", "--quiet"], ROOT_DIR)
    if diff_cp.returncode == 0:
        print("没有新的任务包变更需要提交，跳过 git commit。")
    else:
        msg = f"Add manual AI task {task_name}"
        cp = run_cmd(["git", "commit", "-m", msg], ROOT_DIR)
        print(cp.stdout)
        if cp.stderr:
            print(cp.stderr)
        if cp.returncode != 0:
            raise RuntimeError("git commit 失败。")
    push_cp = run_cmd(["git", "push", "-u", "origin", branch], ROOT_DIR)
    if push_cp.stdout:
        print(push_cp.stdout)
    if push_cp.stderr:
        print(push_cp.stderr)
    if push_cp.returncode != 0:
        raise RuntimeError("git push 失败。")
    print(f"GitHub 分支：{branch}")


def prepare_manual_ai_task(runtime_goal: dict[str, Any], args: argparse.Namespace) -> Path:
    train, _ = _build_periods(runtime_goal)
    if train.timerange:
        maybe_download_data(runtime_goal, args, train.timerange)
    task_name = _sanitize_manual_task_name(args.manual_task_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    branch = args.manual_git_branch or f"ai-manual/{task_name}"
    task_dir = MANUAL_TASK_ROOT / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    best = _load_json_or_empty(BEST_STRATEGY_FILE)
    nearest = _load_json_or_empty(NEAREST_CANDIDATE_FILE)
    last_run = _load_json_or_empty(LAST_RUN_SUMMARY_FILE)
    _write_manual_json(task_dir / "optimization_goal.snapshot.json", runtime_goal)
    _write_manual_json(task_dir / "current_best_summary.json", best)
    _write_manual_json(task_dir / "nearest_candidate_summary.json", nearest)
    _write_manual_json(task_dir / "last_run_summary.json", last_run)
    _write_manual_json(task_dir / "strategy_lessons.json", {"items": _read_json_list_file(LESSONS_FILE)})
    _write_manual_json(task_dir / "strategy_blacklist.json", {"items": _read_json_list_file(BLACKLIST_FILE)})
    _write_manual_json(task_dir / "strategy_memory_excerpt.json", _recent_strategy_memory_excerpt(50))
    _write_manual_json(task_dir / "leaderboard_recent.json", _recent_leaderboard_entries())
    _write_manual_json(task_dir / "pair_entry_tag_summary.json", _latest_pair_entry_tag_summary())
    _write_manual_json(task_dir / "run_context.json", {
        "task_name": task_name,
        "created_at": datetime.utcnow().isoformat(),
        "mode": "manual_ai_prepare",
        "ai_call": "disabled",
        "train_period": runtime_goal.get("train_period", {}),
        "validation_periods": runtime_goal.get("validation_periods", []),
        "holdout_ranges": runtime_goal.get("holdout_ranges", []),
        "git_branch": branch,
    })
    _copy_strategy_or_note(best.get("strategy_file"), task_dir / "current_best_strategy.py", "current_best")
    _copy_strategy_or_note(nearest.get("strategy_file"), task_dir / "nearest_candidate_strategy.py", "nearest_candidate")
    (task_dir / "README.md").write_text(_manual_readme_text(task_name, task_dir.relative_to(ROOT_DIR), branch), encoding="utf-8")
    (task_dir / "codex_advisor_instruction.md").write_text(_manual_advisor_instruction(runtime_goal, best, nearest, last_run), encoding="utf-8")
    (task_dir / "codex_codegen_instruction.md").write_text(_manual_codegen_instruction(), encoding="utf-8")

    hits = _scan_manual_task_for_sensitive_text(task_dir)
    if hits:
        raise RuntimeError("半自动任务包疑似包含敏感密钥，请检查：" + ", ".join(hits))

    if args.manual_git_push:
        run_manual_git_push(task_dir, task_name, branch)

    print("\n========== 半自动任务包已生成 ==========")
    print(f"任务目录：{task_dir}")
    print(f"Git 分支：{branch}")
    print("下一步：")
    print("1. 打开 Codex")
    print("2. 让 Codex 读取 codex_advisor_instruction.md")
    print("3. 生成 generated_mutation_spec.json")
    print("4. 再让 Codex 读取 codex_codegen_instruction.md")
    print("5. 生成 generated_strategy.py")
    print("6. 回到服务器执行：")
    print("   git pull")
    print(f"   python3 ai_tools/auto_optimize_strategy.py --goal {args.goal} --manual-ai-run {task_dir.relative_to(ROOT_DIR) / 'generated_strategy.py'} --manual-ai-task-dir {task_dir.relative_to(ROOT_DIR)}")
    return task_dir


def _rename_first_strategy_class(code: str, class_name: str) -> str:
    pattern = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*IStrategy\s*\)\s*:")
    if not pattern.search(code):
        raise RuntimeError("生成策略中未找到继承 IStrategy 的策略类。")
    return pattern.sub(f"class {class_name}(IStrategy):", code, count=1)


def run_manual_ai_backtest(runtime_goal: dict[str, Any], args: argparse.Namespace, run_dir: Path) -> None:
    strategy_source = _resolve_repo_path(args.manual_ai_run)
    if not strategy_source.exists():
        raise FileNotFoundError(f"manual-ai-run 策略文件不存在：{strategy_source}")
    print("当前模式：半自动策略回测模式")
    print("AI 调用：关闭")
    print(f"策略来源：{strategy_source}")

    config = str(runtime_goal.get("config", args.config))
    timeframe = str(runtime_goal.get("timeframe", args.timeframe))
    strategy_family = str(runtime_goal.get("strategy_family", args.base_strategy))
    train, validations = _build_periods(runtime_goal)
    if not train.timerange:
        raise RuntimeError("缺少训练区间 train_period.timerange，无法继续。")
    maybe_download_data(runtime_goal, args, train.timerange)

    run_id = run_dir.name.replace("run_", "")
    ver = "v001"
    version_dir = run_dir / ver
    version_dir.mkdir(parents=True, exist_ok=True)
    class_name = f"{strategy_family}_{run_id}_manual_{ver}"
    strategy_file = STRATEGY_DIR / f"{class_name}.py"

    raw_code = strategy_source.read_text(encoding="utf-8")
    code = _rename_first_strategy_class(extract_python_code(raw_code), class_name)
    strategy_file.parent.mkdir(parents=True, exist_ok=True)
    strategy_file.write_text(code, encoding="utf-8")
    shutil.copy2(strategy_file, version_dir / "strategy.py")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(strategy_file, GENERATED_DIR / strategy_file.name)

    mutation_spec = {}
    task_dir = _resolve_repo_path(args.manual_ai_task_dir) if args.manual_ai_task_dir else strategy_source.parent
    spec_path = task_dir / "generated_mutation_spec.json"
    if spec_path.exists():
        try:
            mutation_spec = read_json(spec_path)
            write_json(version_dir / "mutation_spec.json", mutation_spec)
        except Exception as exc:
            print(f"读取 generated_mutation_spec.json 失败，将继续回测：{exc}")
    mutation_type = str(mutation_spec.get("mutation_type", "manual_ai") or "manual_ai")
    spec_hash = hashlib.sha256(json.dumps(mutation_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() if mutation_spec else ""
    features = extract_strategy_features(code)
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    signature = _feature_signature(features)

    iteration_stats = {
        "planned_iterations": 1,
        "advisor_success_count": 0,
        "codegen_success_count": 0,
        "generated_versions_count": 1,
        "train_backtest_count": 0,
        "validation_backtest_count": 0,
        "validation_backtest_total_count": 0,
        "skipped_validation_count": 0,
        "valid_strategy_count": 0,
        "invalid_strategy_count": 0,
        "new_best_update_count": 0,
        "current_iteration_version": ver,
        "strategy_memory_retained_count": len(_read_json_list_file(MEMORY_FILE)) if MEMORY_FILE.exists() else 0,
        "global_stats": {},
        "version_statuses": [],
        "manual_ai_mode": True,
    }
    status = {"version": ver, "strategy_class": class_name, "advisor_status": "半自动跳过", "codegen_status": "半自动外部生成", "syntax_check_status": "未执行", "static_check_status": "未执行", "train_backtest_status": "未执行", "validation_backtest_status": "未执行", "is_valid": False, "is_best": False, "invalid_reason": "", "final_score": 0.0}
    iteration_stats["version_statuses"].append(status)
    backtest_errors: list[dict[str, Any]] = []

    def finish_invalid(reason: str) -> None:
        status["invalid_reason"] = reason
        status["is_valid"] = False
        iteration_stats["invalid_strategy_count"] = 1
        summary = _minimal_round_summary(version=ver, strategy_class=class_name, strategy_file=str(strategy_file), state=_new_round_defaults(), mutation_type=mutation_type, failure_reason=reason)
        summary.update({"manual_ai_mode": True, "manual_strategy_source": str(strategy_source), "backtest_errors": backtest_errors, "ai_models_used": {"strategy_advisor": {"used_model": "manual_disabled", "attempts": []}, "code_generator": {"used_model": "manual_disabled", "attempts": []}}})
        write_json(version_dir / "summary.json", summary)
        write_json(run_dir / "leaderboard.json", {"items": [{"version": ver, "run_id": run_id, "strategy_class": class_name, "strategy_file": str(strategy_file), "is_valid": False, "is_best": False, "invalid_reason": reason, "final_score": 0.0}]})
        update_iteration_global_stats(iteration_stats)
        write_json(run_dir / ITERATION_STATS_FILE_NAME, iteration_stats)
        print_log_saved_summary(args)
        print("\n========== 半自动策略回测完成 ==========")
        print(f"策略文件：{strategy_file}")
        print(f"训练区间结果：未执行（{reason}）")
        print(f"验证区间结果：未执行（{reason}）")
        print("是否成为新 best：否")
        print(f"summary.json：{version_dir / 'summary.json'}")
        print(f"run.log：{run_dir / 'run.log'}")

    print("5. 正在检查 Python 语法……")
    pyc = run_cmd([sys.executable, "-m", "py_compile", str(strategy_file)], ROOT_DIR)
    if pyc.returncode != 0:
        print(pyc.stderr)
        finish_invalid("Python 语法检查失败")
        return
    status["syntax_check_status"] = "成功"
    validate_strategy_class_name(strategy_file, class_name)
    static_ok, static_reason = check_entry_long_static(strategy_file)
    if not static_ok:
        finish_invalid(static_reason or "静态检查失败")
        return
    status["static_check_status"] = "成功"

    target_cfg = runtime_goal.get("target", {}) or {}
    baseline_cfg = runtime_goal.get("baseline", {}) or {}
    min_trades = int(target_cfg.get("min_trades", 25))
    max_trades = int(target_cfg.get("max_trades", 80))
    min_trades_grace_ratio = float(target_cfg.get("min_trades_grace_ratio", 0.8))

    train_cmd = ["docker", "compose", "run", "--rm", "freqtrade", "backtesting", "--config", config, "--strategy", class_name, "--timeframe", timeframe, "--timerange", train.timerange, "--export", "trades", "--cache", "none"]
    results_dir = ROOT_DIR / "user_data" / "backtest_results"
    print(f"6. 正在回测训练区间：{train.timerange}")
    before = set(_list_backtest_zips(results_dir))
    ts = time.time()
    cp = run_cmd(train_cmd, ROOT_DIR)
    _write_backtest_process_log(version_dir, "train", train.timerange, cp)
    (version_dir / "backtest_logs.txt").write_text(f"[Train {train.timerange}]\nSTDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}\n", encoding="utf-8")
    if cp.returncode != 0:
        print(cp.stderr)
        finish_invalid("训练区间回测失败")
        return
    train_zip, train_candidates = find_backtest_zip_for_strategy(results_dir, class_name, ts, before)
    if train_zip is None:
        _log_backtest_zip_filter_failure(class_name, train_candidates)
        actual = list(train_candidates[0].get("actual_strategies") or []) if train_candidates else []
        wrong_zip = str(train_candidates[0].get("zip") or "") if train_candidates else ""
        _record_backtest_error(backtest_errors, stage="train", timerange=train.timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
        finish_invalid("训练区间回测结果 zip 不匹配或缺失")
        return
    train_zip_local = _copy_backtest_zip_to_version(train_zip, version_dir, "train", train.timerange)
    train_result, actual_train_keys = parse_backtest_from_zip(train_zip_local, class_name, strict=False)
    if train_result is None:
        _record_backtest_error(backtest_errors, stage="train", timerange=train.timerange, expected_strategy=class_name, wrong_zip=str(train_zip), actual_strategies=actual_train_keys, error="backtest_parse_failed")
        finish_invalid("训练区间回测结果解析失败")
        return
    train_metrics = _extract_metrics(train_result)
    pair_metrics = _normalize_pair_metrics(train_metrics.get("pairs", []))
    entry_tag_metrics = _normalize_entry_tag_metrics(train_metrics.get("entry_tags", []))
    write_json(version_dir / "train_metrics.json", train_metrics)
    _print_round_table(ver, train.timerange, train_metrics)
    status["train_backtest_status"] = "已训练回测"
    iteration_stats["train_backtest_count"] = 1

    train_score = _score(train_metrics, train)
    train_trades = _safe_int(train_metrics.get("total_trades"))
    hard_invalid_reason: str | None = None
    validation_skip_reason = ""
    trade_under_min = False
    cannot_be_official_best_unless_validation_strong = False
    trade_count_warning = ""
    if train_trades == 0:
        hard_invalid_reason = "训练区间无交易"
        validation_skip_reason = hard_invalid_reason
    elif train_trades < min_trades * min_trades_grace_ratio:
        hard_invalid_reason = "训练区间交易数低于目标下限"
        validation_skip_reason = hard_invalid_reason
    elif train_trades < min_trades:
        trade_under_min = True
        cannot_be_official_best_unless_validation_strong = True
        trade_count_warning = "训练交易数略低于目标，但表现接近打平，建议下一轮略微增加信号。"
    elif train_trades > max_trades * 1.5:
        hard_invalid_reason = "训练区间交易数严重超过目标上限"
        validation_skip_reason = hard_invalid_reason

    validation_metrics: list[dict[str, Any]] = []
    val_scores: list[float] = []
    validation_status = "not_run"
    if hard_invalid_reason:
        print(f"训练区间触发硬约束：{hard_invalid_reason}，跳过所有验证区间回测。")
        iteration_stats["skipped_validation_count"] = 1
        status["validation_backtest_status"] = "跳过验证"
    else:
        for p in validations:
            print(f"7. 正在回测验证区间：{p.timerange}")
            vcmd = train_cmd.copy()
            vcmd[vcmd.index("--timerange") + 1] = p.timerange
            before = set(_list_backtest_zips(results_dir))
            ts = time.time()
            vcp = run_cmd(vcmd, ROOT_DIR)
            _write_backtest_process_log(version_dir, f"validation_{p.name}", p.timerange, vcp)
            with (version_dir / "backtest_logs.txt").open("a", encoding="utf-8") as logf:
                logf.write(f"\n[Validation {p.name} {p.timerange}]\nSTDOUT:\n{vcp.stdout}\n\nSTDERR:\n{vcp.stderr}\n")
            if vcp.returncode != 0:
                print(vcp.stderr)
                _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, error="backtest_failed")
                validation_status = "backtest_failed"
                break
            vzip, v_candidates = find_backtest_zip_for_strategy(results_dir, class_name, ts, before)
            if vzip is None:
                _log_backtest_zip_filter_failure(class_name, v_candidates)
                actual = list(v_candidates[0].get("actual_strategies") or []) if v_candidates else []
                wrong_zip = str(v_candidates[0].get("zip") or "") if v_candidates else ""
                _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
                validation_status = "backtest_parse_failed"
                break
            vzip_local = _copy_backtest_zip_to_version(vzip, version_dir, "validation", p.timerange)
            v_result, actual_v_keys = parse_backtest_from_zip(vzip_local, class_name, strict=False)
            if v_result is None:
                _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, wrong_zip=str(vzip), actual_strategies=actual_v_keys, error="backtest_parse_failed")
                validation_status = "backtest_parse_failed"
                break
            vm = _extract_metrics(v_result)
            validation_metrics.append({"period": p.name, "timerange": p.timerange, "metrics": vm})
            val_scores.append(_score(vm, p))
            iteration_stats["validation_backtest_total_count"] += 1
            _print_round_table(ver, p.timerange, vm)
        if validation_status in {"backtest_failed", "backtest_parse_failed"}:
            status["validation_backtest_status"] = "失败"
        elif validation_metrics:
            validation_status = "completed"
            status["validation_backtest_status"] = "已验证回测"
            iteration_stats["validation_backtest_count"] = 1
        else:
            validation_status = "failed"
            status["validation_backtest_status"] = "失败"
    validation_score = sum(val_scores) / len(val_scores) if val_scores else 0.0
    write_json(version_dir / "validation_metrics.json", {"periods": validation_metrics, "average_score": validation_score})

    overfit_penalty = max(0.0, train_score - validation_score) * 0.3
    baseline_ok, baseline_reason, baseline_dd_penalty = _baseline_gate_and_penalty(train_metrics, runtime_goal)
    final_score, score_penalty_detail = _compute_final_score(train_score, validation_score, train_metrics, validation_metrics, target_cfg)
    final_score = final_score - overfit_penalty - baseline_dd_penalty
    if hard_invalid_reason or train_trades > max_trades:
        final_score = min(final_score, 0.0) if train_trades > max_trades else 0.0
    is_overfit = train_score > validation_score * 1.3 if validation_score else True
    validation_strong = _is_validation_strong(validation_metrics, baseline_cfg, target_cfg)
    is_valid, invalid_reason = _validate_round(train_metrics, validation_metrics, final_score)
    if hard_invalid_reason:
        is_valid = False
        invalid_reason = hard_invalid_reason
    elif trade_under_min and not validation_strong:
        is_valid = False
        invalid_reason = "训练区间交易数略低于目标下限，验证强度不足"
    elif train_trades > max_trades:
        is_valid = False
        invalid_reason = "训练区间交易数超过目标上限"
    if backtest_errors:
        is_valid = False
        is_best = False
        final_score = 0
        invalid_reason = invalid_reason or "回测结果解析失败或 zip 不匹配"
    elif is_valid and not baseline_ok:
        is_valid = False
        invalid_reason = baseline_reason or "未通过 baseline 检查"

    historical_best = _load_json_or_empty(BEST_STRATEGY_FILE)
    historical_score = _safe_float(historical_best.get("final_score")) if historical_best.get("final_score") is not None else -1e18
    is_best = bool(is_valid and final_score > historical_score)
    holdout_metrics: list[dict[str, Any]] = []
    holdout_status = "not_run"
    holdout_reason = "未达到候选 best 条件，未执行 holdout"
    if is_best and runtime_goal.get("holdout_ranges"):
        holdout_status = "completed"
        holdout_reason = ""
        holdout_failed = False
        for idx, h in enumerate(runtime_goal.get("holdout_ranges", []) or [], start=1):
            h_timerange = str(h.get("timerange") or "")
            if not TIMERANGE_RE.match(h_timerange):
                continue
            hcmd = train_cmd.copy()
            hcmd[hcmd.index("--timerange") + 1] = h_timerange
            before = set(_list_backtest_zips(results_dir))
            ts = time.time()
            hcp = run_cmd(hcmd, ROOT_DIR)
            _write_backtest_process_log(version_dir, f"holdout_{idx:02d}", h_timerange, hcp)
            if hcp.returncode != 0:
                _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, error="backtest_failed")
                holdout_failed = True
                break
            hzip, h_candidates = find_backtest_zip_for_strategy(results_dir, class_name, ts, before)
            if hzip is None:
                _log_backtest_zip_filter_failure(class_name, h_candidates)
                actual = list(h_candidates[0].get("actual_strategies") or []) if h_candidates else []
                wrong_zip = str(h_candidates[0].get("zip") or "") if h_candidates else ""
                _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
                holdout_failed = True
                break
            hzip_local = _copy_backtest_zip_to_version(hzip, version_dir, "holdout", h_timerange, str(h.get("label") or f"holdout_{idx:02d}"))
            h_result, actual_h_keys = parse_backtest_from_zip(hzip_local, class_name, strict=False)
            if h_result is None:
                _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, wrong_zip=str(hzip), actual_strategies=actual_h_keys, error="backtest_parse_failed")
                holdout_failed = True
                break
            hm = _extract_metrics(h_result)
            holdout_metrics.append({"label": str(h.get("label") or f"holdout_{idx:02d}"), "timerange": h_timerange, "metrics": hm})
            if _safe_float(hm.get("profit_total_pct")) < -2.5 or _safe_float(hm.get("profit_factor")) < 0.45 or _safe_float(hm.get("max_drawdown_pct")) > 3.0:
                holdout_failed = True
        if holdout_failed:
            is_best = False
            is_valid = False
            invalid_reason = "holdout 复验未通过，降级为 nearest_candidate"
            holdout_status = "failed"
            holdout_reason = "holdout 复验未通过"
        elif not holdout_metrics:
            holdout_status = "not_run"
            holdout_reason = "未找到可执行 holdout 区间，未执行 holdout"

    status.update({"is_valid": bool(is_valid), "is_best": bool(is_best), "invalid_reason": invalid_reason, "final_score": float(final_score)})
    iteration_stats["valid_strategy_count" if is_valid else "invalid_strategy_count"] = 1
    if is_best:
        iteration_stats["new_best_update_count"] = 1

    avg_val_profit, avg_val_pf, max_val_dd = _aggregate_validation_metrics(validation_metrics)
    score_breakdown = {"train_score": train_score, "validation_score": validation_score, "overfit_penalty": overfit_penalty, "baseline_dd_penalty": baseline_dd_penalty, "penalty_detail": score_penalty_detail, "baseline_check": {"passed": baseline_ok, "reason": baseline_reason}}
    reason_detail = [] if is_best else _build_not_best_reason_detail(train_metrics, validation_metrics, final_score, historical_best, target_cfg, invalid_reason)
    summary = {
        "strategy_class": class_name,
        "strategy_file": str(strategy_file),
        "manual_ai_mode": True,
        "manual_strategy_source": str(strategy_source),
        "manual_ai_task_dir": str(task_dir),
        "mutation_type": mutation_type,
        "session_parent_choice": mutation_spec.get("session_parent_choice", "manual"),
        "session_parent_reason": mutation_spec.get("session_parent_reason", ""),
        "changed_items": mutation_spec.get("changes", []),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_status": validation_status,
        "validation_skip_reason": validation_skip_reason,
        "holdout_metrics": holdout_metrics,
        "holdout_status": holdout_status,
        "holdout_reason": holdout_reason,
        "pair_metrics": pair_metrics,
        "entry_tag_metrics": entry_tag_metrics,
        "score_breakdown": score_breakdown,
        "overfit_result": {"is_overfit": is_overfit, "train_score": train_score, "validation_score": validation_score},
        "final_score": final_score,
        "failure_reason": invalid_reason or "通过",
        "reason_detail": reason_detail,
        "is_best": is_best,
        "is_valid": is_valid,
        "invalid_reason": invalid_reason,
        "features": features,
        "spec_hash": spec_hash,
        "code_hash": code_hash,
        "trade_under_min": trade_under_min,
        "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
        "validation_strong": validation_strong,
        "trade_count_warning": trade_count_warning,
        "backtest_errors": backtest_errors,
        "ai_models_used": {"strategy_advisor": {"used_model": "manual_disabled", "attempts": []}, "code_generator": {"used_model": "manual_disabled", "attempts": []}},
    }
    write_json(version_dir / "summary.json", summary)

    leaderboard_entry = {"version": ver, "run_id": run_id, "strategy_class": class_name, "strategy_file": str(strategy_file), "code_hash": code_hash, "created_at": datetime.utcnow().isoformat(), "final_score": final_score, "train_profit_pct": _safe_float(train_metrics.get("profit_total_pct")), "train_profit_abs": _safe_float(train_metrics.get("profit_total_abs")), "avg_validation_profit_pct": avg_val_profit, "avg_validation_profit_factor": avg_val_pf, "max_validation_drawdown_pct": max_val_dd, "validation_metrics": validation_metrics, "trade_under_min": trade_under_min, "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong, "validation_strong": validation_strong, "trade_count_warning": trade_count_warning, "profit_factor": _safe_float(train_metrics.get("profit_factor")), "max_drawdown_pct": _safe_float(train_metrics.get("max_drawdown")) * 100.0, "total_trades": _safe_int(train_metrics.get("total_trades")), "is_overfit": is_overfit, "is_best": is_best, "is_valid": is_valid, "invalid_reason": invalid_reason, "features": features, "spec_hash": spec_hash, "feature_signature": signature, "mutation_type": mutation_type, **_extract_exit_profit_fields(train_metrics)}
    write_json(run_dir / "leaderboard.json", {"items": [leaderboard_entry]})

    memory_items = _read_json_list_file(MEMORY_FILE)
    memory_items.append(leaderboard_entry)
    _write_json_list_file(MEMORY_FILE, memory_items[-200:])
    update_iteration_global_stats(iteration_stats)
    write_json(run_dir / ITERATION_STATS_FILE_NAME, iteration_stats)
    if not is_valid:
        blacklist_items = _read_json_list_file(BLACKLIST_FILE)
        blacklist_items.append({**leaderboard_entry, "failure_reason": invalid_reason, "avoid_next": "避免重复本轮半自动失败结构。"})
        _write_json_list_file(BLACKLIST_FILE, blacklist_items[-200:])
    lessons_items = _read_json_list_file(LESSONS_FILE)
    lessons_items.append({"version": ver, "manual_ai_mode": True, "failure_reason": invalid_reason or "通过", "avoid_next": "继续围绕 best/nearest 做单点小步修改。"})
    _write_json_list_file(LESSONS_FILE, lessons_items[-200:])

    nearest_candidate = None
    if not is_best and train_trades > 0:
        nearest_candidate = {"strategy_class": class_name, "strategy_file": str(strategy_file), "why_nearest": _build_why_nearest(leaderboard_entry, target_cfg, historical_best), "train_metrics": {"profit_total_abs": leaderboard_entry.get("train_profit_abs"), "profit_total_pct": leaderboard_entry.get("train_profit_pct"), "profit_factor": leaderboard_entry.get("profit_factor"), "max_drawdown_pct": leaderboard_entry.get("max_drawdown_pct"), "total_trades": leaderboard_entry.get("total_trades"), "roi_profit_abs": leaderboard_entry.get("roi_profit_abs"), "stop_loss_profit_abs": leaderboard_entry.get("stop_loss_profit_abs")}, "validation_metrics": validation_metrics, "trade_under_min": trade_under_min, "validation_strong": validation_strong, "trade_count_warning": trade_count_warning}
        write_json(NEAREST_CANDIDATE_FILE, nearest_candidate)
    if is_best:
        current_best_saved = {"strategy_class": class_name, "strategy_file": str(strategy_file), "source_run_id": run_id, "version": ver, "train_metrics": train_metrics, "validation_metrics": validation_metrics, "avg_validation_metrics": {"profit_total_pct": avg_val_profit, "profit_factor": avg_val_pf, "max_drawdown_pct": max_val_dd}, "score_breakdown": score_breakdown, "final_score": final_score, "created_at": datetime.utcnow().isoformat(), "why_best": "半自动策略回测 final_score 超过历史 best。", "is_overfit": bool(is_overfit)}
        write_json(BEST_STRATEGY_FILE, current_best_saved)
        shutil.copy2(strategy_file, GENERATED_DIR / f"BEST_{strategy_family}.py")

    last_run_summary = {"run_id": run_id, "manual_ai_mode": True, "best_updated": bool(is_best), "current_best": read_json(BEST_STRATEGY_FILE) if BEST_STRATEGY_FILE.exists() else None, "nearest_candidate": nearest_candidate, "leaderboard_top": [leaderboard_entry], "common_failure_patterns": [] if is_valid else [invalid_reason], "forbidden_next_mutation_types": ["扩大止损", "启用 exit_signal", "启用杠杆/加仓/马丁格尔"], "recommended_next_mutation_types": ["add_entry_filter", "tighten_entry_trigger", "remove_bad_entry_condition", "pair_specific_filter", "tag_specific_filter"]}
    write_json(LAST_RUN_SUMMARY_FILE, last_run_summary)

    if hasattr(args, "_run_start_ts"):
        elapsed = time.time() - float(getattr(args, "_run_start_ts"))
        print("\n========== 本次运行总耗时 ==========")
        print(f"总耗时：{_format_elapsed_seconds(elapsed)}（{elapsed:.1f} 秒）")
    print_log_saved_summary(args)
    print("\n========== 半自动策略回测完成 ==========")
    print(f"策略文件：{strategy_file}")
    print(f"训练区间结果：profit={_safe_float(train_metrics.get('profit_total_pct')):.2f}%, trades={_safe_int(train_metrics.get('total_trades'))}, PF={_safe_float(train_metrics.get('profit_factor')):.4f}")
    print(f"验证区间结果：avg_profit={avg_val_profit:.2f}%, avg_PF={avg_val_pf:.4f}, max_dd={max_val_dd:.2f}%")
    print(f"是否成为新 best：{'是' if is_best else '否'}")
    print(f"summary.json：{version_dir / 'summary.json'}")
    print(f"run.log：{run_dir / 'run.log'}")


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
        provider_pool=list(code_runtime.provider_pool),
    )
    _print_ai_model_pool_config(advisor_runtime, code_runtime, float(args.ai_call_cooldown_seconds))
    print_effective_pair_selection_log(runtime_goal)

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
    loaded_champion_meta = champion.get("meta", {}) or _build_baseline_best(runtime_goal)
    session_state: dict[str, Any] = {
        "official_champion": loaded_champion_meta,
        "in_memory_champion": loaded_champion_meta,
        "session_best": None,
        "session_nearest_candidate": nearest_mem,
        "last_round_summary": None,
        "round_history": [],
        "failed_mutation_types_this_run": [],
        "successful_mutation_types_this_run": [],
        "common_failure_patterns_this_run": [],
        "attempted_mutation_types_this_run": [],
    }
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
    tested_strategy_fingerprints: list[dict[str, Any]] = []
    early_stop_reason = ""
    random_sample_plan = build_random_sample_plan(args, runtime_goal, run_dir)
    print_random_sample_config(random_sample_plan)
    pre_run_ai_review = run_pre_run_ai_review(
        runtime_goal,
        run_dir,
        advisor_runtime,
        code_runtime,
        ai_runtime_state,
    )

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
        "strategy_memory_retained_count": 0,
        "global_stats": {},
        "random_sample_enabled": bool(random_sample_plan.get("enabled")),
        "random_sample_windows_count": len(random_sample_plan.get("windows", []) or []),
        "random_sample_total_backtests": 0,
        "early_stop_triggered": False,
        "early_stop_reason": "",
        "early_stop_checked_counters": {
            "no_new_best": 0,
            "final_score_failures": 0,
            "duplicate_strategies": 0,
        },
        "version_statuses": version_statuses,
    }

    def _is_duplicate_for_codegen_switch(row: dict[str, Any]) -> bool:
        behavior_report = row.get("behavior_duplicate", {}) or {}
        return (
            str(row.get("invalid_reason") or "") == "策略与本次 run 已测试策略高度重复"
            or bool(row.get("is_behavior_duplicate", False))
            or bool(behavior_report.get("is_duplicate", False))
            or str(row.get("not_best_reason") or "") == "策略行为与当前 official_best 几乎一致，不覆盖 best。"
        )

    def _count_trailing_in_rows(rows: list[dict[str, Any]], predicate: Any) -> int:
        count = 0
        for row in reversed(rows):
            if predicate(row):
                count += 1
            else:
                break
        return count

    def _count_trailing(predicate: Any) -> int:
        return _count_trailing_in_rows(version_statuses, predicate)

    def should_early_stop() -> str:
        no_best_count = _count_trailing(lambda row: row.get("advisor_status") != "未执行" and not row.get("is_best"))
        final_score_failure_count = _count_trailing(lambda row: row.get("advisor_status") != "未执行" and _safe_float(row.get("final_score")) <= 0)
        duplicate_count = _count_trailing(_is_duplicate_for_codegen_switch)
        iteration_stats["early_stop_checked_counters"] = {
            "no_new_best": no_best_count,
            "final_score_failures": final_score_failure_count,
            "duplicate_strategies": duplicate_count,
        }
        if int(args.early_stop_patience) > 0 and no_best_count >= int(args.early_stop_patience):
            return f"已连续 {int(args.early_stop_patience)} 轮没有新 best，触发 early stopping。"
        if int(args.early_stop_final_score_failures) > 0 and final_score_failure_count >= int(args.early_stop_final_score_failures):
            return f"已连续 {int(args.early_stop_final_score_failures)} 轮 final_score<=0，触发 early stopping。"
        if int(args.early_stop_duplicate_strategies) > 0 and duplicate_count >= int(args.early_stop_duplicate_strategies):
            return "连续生成重复策略，停止以避免浪费 API。"
        return ""

    def flush_iteration_stats() -> None:
        update_iteration_global_stats(iteration_stats)
        write_json(iteration_stats_path, iteration_stats)
    for i in range(1, iterations + 1):
        early_stop_reason = should_early_stop()
        if early_stop_reason:
            iteration_stats["early_stop_triggered"] = True
            iteration_stats["early_stop_reason"] = early_stop_reason
            print(early_stop_reason)
            print("本次运行提前结束。")
            flush_iteration_stats()
            break
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
            "random_sample_status": "skipped",
            "random_sample_count": 0,
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
        strategy_fingerprint = round_state["strategy_fingerprint"]
        duplicate_report = round_state["duplicate_report"]
        behavior_duplicate = round_state["behavior_duplicate"]
        is_behavior_duplicate = bool(round_state["is_behavior_duplicate"])
        not_best_reason = str(round_state["not_best_reason"])
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
        backtest_errors: list[dict[str, Any]] = round_state["backtest_errors"]
        random_sample_metrics: list[dict[str, Any]] = round_state["random_sample_metrics"]
        random_sample_errors: list[dict[str, Any]] = round_state["random_sample_errors"]
        random_sample_observation: dict[str, Any] = round_state["random_sample_observation"]

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
                "strategy_fingerprint": strategy_fingerprint,
                "duplicate_report": duplicate_report,
                "behavior_duplicate": behavior_duplicate,
                "is_behavior_duplicate": is_behavior_duplicate,
                "not_best_reason": not_best_reason,
                "final_score": final_score,
                "score_breakdown": score_breakdown,
                "invalid_reason": invalid_reason,
                "is_valid": is_valid,
                "is_best": is_best,
                "trade_under_min": trade_under_min,
                "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
                "validation_strong": validation_strong,
                "trade_count_warning": trade_count_warning,
                "backtest_errors": backtest_errors,
                "random_sample_metrics": random_sample_metrics,
                "random_sample_usage": dict(RANDOM_SAMPLE_USAGE),
                "random_sample_observation": random_sample_observation,
                "random_sample_errors": random_sample_errors,
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

        def update_session_state_after_round(
            *,
            summary_path: Path | None = None,
            summary_data: dict[str, Any] | None = None,
            strategy_record: dict[str, Any] | None = None,
            became_best: bool = False,
            nearest_candidate_record: dict[str, Any] | None = None,
        ) -> None:
            round_summary = summary_data
            if round_summary is None and summary_path is not None and summary_path.exists():
                try:
                    round_summary = read_json(summary_path)
                except Exception:
                    round_summary = None
            if not isinstance(round_summary, dict):
                round_summary = {
                    "version": ver,
                    "strategy_class": class_name,
                    "is_valid": bool(is_valid),
                    "is_best": bool(is_best),
                    "invalid_reason": invalid_reason,
                    "failure_reason": failure_reason or invalid_reason,
                    "mutation_type": mutation_type,
                }
            round_summary.setdefault("version", ver)
            round_summary.setdefault("mutation_type", mutation_type)
            session_state["last_round_summary"] = round_summary
            session_state["round_history"].append(round_summary)
            _append_unique(session_state["attempted_mutation_types_this_run"], mutation_type)
            if round_summary.get("is_valid"):
                _append_unique(session_state["successful_mutation_types_this_run"], mutation_type)
            else:
                _append_unique(session_state["failed_mutation_types_this_run"], mutation_type)
                _append_unique(session_state["common_failure_patterns_this_run"], round_summary.get("invalid_reason") or round_summary.get("failure_reason"))

            if became_best and strategy_record:
                session_state["official_champion"] = strategy_record
                session_state["in_memory_champion"] = strategy_record
                session_state["session_best"] = strategy_record
                print(f"第 {ver} 轮成为新 best，下一轮将以该策略作为 in_memory_champion/session_parent 参考。")
                return

            candidate = nearest_candidate_record or strategy_record
            if candidate and not became_best and _is_better_session_nearest(candidate, session_state.get("session_nearest_candidate"), target_cfg, baseline_cfg):
                session_state["session_nearest_candidate"] = candidate
                print(f"第 {ver} 轮未成为 best，但已更新为本次 run 的 nearest_candidate，下一轮将作为参考。")

        print(f"\n========== 第 {i} 轮 / {ver} ==========")
        print("1. 正在生成 mutation_spec（冠军-挑战者小步改动）……")
        print(f"当前策略顾问模型池：{', '.join(advisor_runtime.model_pool)}")
        print(f"当前代码生成模型池：{', '.join(code_runtime.model_pool)}")
        previous_version_statuses = version_statuses[:-1]
        duplicate_streak_before_codegen = _count_trailing_in_rows(previous_version_statuses, _is_duplicate_for_codegen_switch)
        codegen_pool_size = len(code_runtime.provider_pool or code_runtime.model_pool)
        if duplicate_streak_before_codegen >= 1 and codegen_pool_size > 1:
            code_runtime.forced_provider_offset = duplicate_streak_before_codegen % codegen_pool_size
            print(f"已连续 {duplicate_streak_before_codegen} 次生成重复策略，本轮强制切换 codegen provider/模型起点 offset={code_runtime.forced_provider_offset}。")
        else:
            code_runtime.forced_provider_offset = 0
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
        last_round_summary = _strip_random_samples_for_ai_prompt(session_state.get("last_round_summary"))
        last_round_failure = ""
        if isinstance(last_round_summary, dict) and last_round_summary:
            last_round_failure = str(last_round_summary.get("invalid_reason") or last_round_summary.get("failure_reason") or "通过")
        failure_context = f"上一轮失败原因：{last_round_failure or previous_failure_reason}\n" if (last_round_failure or previous_failure_reason) else ""
        failure_context += "最近失败策略共同原因通常不是没有盈利单，而是固定止损或 trailing_stop_loss 吃掉 ROI 收益。\n"
        compact_memory = build_compact_strategy_context(memory_items, baseline_cfg, memory_max_items, memory_max_chars) if memory_enabled else ""
        official_champion = session_state.get("official_champion") or _build_baseline_best(runtime_goal)
        in_memory_champion = session_state.get("in_memory_champion") or official_champion
        session_best_record = session_state.get("session_best")
        session_nearest_record = session_state.get("session_nearest_candidate")
        session_parent_candidates = {
            "historical_best": official_champion,
            "session_best": session_best_record,
            "nearest_candidate": session_nearest_record,
            "baseline": {"strategy_class": "baseline", "train_metrics": baseline_cfg},
        }
        official_champion_name = _strategy_label(official_champion, "baseline")
        in_memory_champion_name = _strategy_label(in_memory_champion, "baseline")
        prompt_has_last_round = bool(last_round_summary)
        pre_run_advisor_injected = bool(pre_run_ai_review)
        pre_run_codegen_injected = bool(pre_run_ai_review and isinstance(pre_run_ai_review.get("codegen_guidance"), dict))
        print("========== 本轮迭代上下文 ==========")
        print(f"当前轮：{ver}")
        print(f"上一轮结果：{_round_summary_label(last_round_summary)}")
        print(f"当前 in_memory_champion：{in_memory_champion_name}")
        print(f"当前 session_best：{_strategy_label(session_best_record)}")
        print(f"当前 session_nearest_candidate：{_strategy_label(session_nearest_record)}")
        print(f"本次 run 已失败 mutation_type：{session_state['failed_mutation_types_this_run'] or '无'}")
        print(f"本次 run 已成功 mutation_type：{session_state['successful_mutation_types_this_run'] or '无'}")
        print(f"本轮 advisor prompt 已包含上一轮结果：{'是' if prompt_has_last_round else '否'}")
        print(f"pre_run_ai_review 已注入 advisor prompt：{'是' if pre_run_advisor_injected else '否'}")
        print(f"pre_run_ai_review.codegen_guidance 已注入 codegen prompt：{'是' if pre_run_codegen_injected else '否'}")
        print(f"当前正式 champion：{official_champion_name}")
        print(f"当前 in_memory_champion：{in_memory_champion_name}")
        print("当前 session_parent 候选：")
        print(f"- historical_best: {_strategy_label(session_parent_candidates['historical_best'])}")
        print(f"- session_best: {_strategy_label(session_parent_candidates['session_best'])}")
        print(f"- nearest_candidate: {_strategy_label(session_parent_candidates['nearest_candidate'])}")
        spec_prompt = _strategy_spec_prompt(class_name, runtime_goal, baseline_cfg, compact_memory, last_round_failure or previous_failure_reason)
        spec_prompt += "\n\n========== 同一次 run 动态 session_state（必须优先于跨 run 旧记忆）==========\n"
        spec_prompt += "当前 official_champion=" + _compact_prompt_json(official_champion, 5000) + "\n"
        spec_prompt += "当前 in_memory_champion=" + _compact_prompt_json(in_memory_champion, 5000) + "\n"
        spec_prompt += "当前 session_best=" + _compact_prompt_json(session_best_record, 5000) + "\n"
        spec_prompt += "当前 session_nearest_candidate=" + _compact_prompt_json(session_nearest_record, 5000) + "\n"
        spec_prompt += "上一轮 last_round_summary=" + _compact_prompt_json(last_round_summary, 5000) + "\n"
        spec_prompt += "本次 run 已尝试 mutation_type=" + _compact_prompt_json(session_state["attempted_mutation_types_this_run"], 2000) + "\n"
        spec_prompt += "本次 run 已失败 mutation_type=" + _compact_prompt_json(session_state["failed_mutation_types_this_run"], 2000) + "\n"
        spec_prompt += "本次 run 已成功 mutation_type=" + _compact_prompt_json(session_state["successful_mutation_types_this_run"], 2000) + "\n"
        spec_prompt += "本次 run 已出现失败模式=" + _compact_prompt_json(session_state["common_failure_patterns_this_run"], 3000) + "\n"
        spec_prompt += "请不要重复上一轮失败方向；必须在 reason/session_parent_reason 中说明本轮为什么选择当前 mutation_type。\n"
        if last_run_summary_mem:
            spec_prompt += "\n跨 run last_run_summary=" + _compact_prompt_json(_strip_random_samples_for_ai_prompt(last_run_summary_mem), 4000) + "\n"
        spec_prompt += "\n跨 run strategy_lessons=" + _compact_prompt_json(lessons_items[-memory_max_items:], 4000) + "\n"
        spec_prompt += "\n跨 run strategy_blacklist=" + _compact_prompt_json(blacklist_items[-memory_max_items:], 4000) + "\n"
        spec_prompt += _format_pre_run_review_for_advisor(pre_run_ai_review)
        spec_prompt += f"\nchampion_strategy_class={_strategy_label(in_memory_champion, 'baseline')}\n"
        spec_prompt += _v045_small_step_prompt_section({"meta": official_champion}, runtime_goal)
        spec_prompt += _format_prompt_guidance_section(
            "========== 用户额外策略顾问规则 ==========",
            _prompt_guidance_rules(runtime_goal, "advisor_rules"),
        )
        spec_prompt += f"\n已失败 mutation_type（避免重复）={sorted(set(used_failed_mutations).union(session_state['failed_mutation_types_this_run']))}\n"
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
            summary_path = write_iteration_summary()
            update_session_state_after_round(summary_path=summary_path)
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
            summary_path = write_iteration_summary()
            update_session_state_after_round(summary_path=summary_path)
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            print(f"strategy_spec 原始返回已保存：{version_dir / 'strategy_spec.raw.txt'}")
            print(f"8. 第 {i} 轮完成：无效，原因：{invalid_reason}")
            continue
        try:
            strategy_spec = extract_json_object(spec_text)
        except (ValueError, json.JSONDecodeError):
            previous_failure_reason = "mutation_spec 不是有效 JSON object。"
            invalid_reason = "mutation_spec JSON 解析失败"
            summary_path = write_iteration_summary()
            update_session_state_after_round(summary_path=summary_path)
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            print(f"strategy_spec 解析失败，请检查：{version_dir / 'strategy_spec.raw.txt'}")
            print(f"8. 第 {i} 轮完成：无效，原因：{invalid_reason}")
            continue
        write_json(version_dir / "mutation_spec.json", strategy_spec)
        print(f"2. mutation_spec 已保存：{version_dir / 'mutation_spec.json'}")
        spec_hash = hashlib.sha256(json.dumps(strategy_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        mutation_type = str(strategy_spec.get("mutation_type", "") or "")
        with (version_dir / "advisor_prompt.txt").open("a", encoding="utf-8") as prompt_audit:
            prompt_audit.write("\n\n========== advisor 返回的本轮选择（审计）==========\n")
            prompt_audit.write(f"本轮选择 mutation_type={mutation_type or '未提供'}\n")
            prompt_audit.write(f"本轮为什么选择当前 mutation_type={strategy_spec.get('reason') or strategy_spec.get('session_parent_reason') or '未提供'}\n")
            prompt_audit.write("完整 mutation_spec=" + json.dumps(strategy_spec, ensure_ascii=False, indent=2) + "\n")
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
            f"5) {_runtime_trade_target_text(runtime_goal)}\n"
            f"{_auto_trade_count_target_prompt_note(runtime_goal)}"
            "6) use_exit_signal 必须为 False，不允许改为 True。\n"
            "7) 仅现货 long only：不做空、不杠杆、不马丁格尔、不无限补仓，不允许 conditions_short。\n"
            "8) 不调用外部 API，不读取手动交易记录。\n"
            "9) 禁止使用过强过滤的全 AND 叠加（如 close>ema200_1h、rsi_1h>55、ema20>ema50>ema100、volume>rolling_mean*1.5 同时成立）。\n"
            "10) 入场逻辑可更宽松，鼓励用 OR 组合：RSI 回调反弹 / EMA 短周期金叉 / 布林带下轨反弹 / MACD 转强 / 成交量不极低。\n"
            f"11) 不允许生成完全无交易策略，也不允许超过 {int(max_trades * 1.5)} 笔训练交易。\n"
            "12) 目标不是追求 0 回撤，而是在足够交易数下综合表现优于 baseline。\n"
            f"{failure_context}"
            f"{compact_memory}\n"
            f"{_v045_small_step_prompt_section({'meta': official_champion}, runtime_goal)}\n"
            "当前 baseline：\n"
            f"- 总收益(USDT)：{baseline_cfg.get('profit_total_abs', -7.43)}\n"
            f"- 收益率(%)：{baseline_cfg.get('profit_total_pct', -0.74)}\n"
            f"- Profit Factor：{baseline_cfg.get('profit_factor', 0.63)}\n"
            f"- 最大回撤(%)：{baseline_cfg.get('max_drawdown_pct', 1.45)}\n"
            f"- 交易数：{baseline_cfg.get('total_trades', 47)}\n"
            "输出要求：\n"
            "- 只输出可运行的完整 Python 策略代码，不要解释。\n"
            "- 必须真实落地 mutation_spec，不能只改 class name、不能只改注释、不能只改变量名。\n"
            "- 必须至少改变一个可由 fingerprint 检测到的策略结构：entry_conditions / pair-specific branch / indicator / ROI / stoploss / protections 之一。\n"
            "- 如果 mutation_spec.mutation_type 是 pair_specific_filter、tag_specific_filter、add_entry_filter、tighten_entry_trigger，必须在 populate_entry_trend 中生成对应的实际条件，并影响 enter_long。\n"
            "- 如果 mutation_spec 要求 ETH/USDT 专属过滤，代码里必须出现明确的 ETH/USDT 分支或等价 pair-specific 条件（例如 metadata['pair'] == 'ETH/USDT' 或 pair 变量判断）。\n"
            "- 如果 mutation_spec 要求新增指标（ema20、ema50、adx、atr_pct、bollinger_middleband、volume_mean_20 等），必须在 populate_indicators 中实际计算这些 dataframe 列，并在入场条件中引用相关列。\n"
            "- 避免把入场条件写成几乎永远不触发的苛刻组合。\n"
        )
        prompt += _format_pre_run_review_for_codegen(pre_run_ai_review)
        prompt += _format_prompt_guidance_section(
            "========== 用户额外代码生成规则 ==========",
            _prompt_guidance_rules(runtime_goal, "codegen_rules"),
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
            summary_path = write_iteration_summary()
            update_session_state_after_round(summary_path=summary_path)
            leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": invalid_reason})
            continue
        (version_dir / "codegen.raw.txt").write_text(response_text, encoding="utf-8")
        status["codegen_status"] = "成功"
        iteration_stats["codegen_success_count"] += 1
        code = extract_python_code(response_text)
        features = extract_strategy_features(code)
        strategy_fingerprint = build_strategy_fingerprint(features)
        signature = strategy_fingerprint["hash"]
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        similarity_threshold = float(args.similarity_threshold)
        similarity_report = {"is_similar": False, "similarity_score": 0.0, "similar_to": "", "similar_type": "none", "reasons": [], "decision": "auto_continue", "user_prompt_required": False}
        parent_pool = [
            ("session_parent", champion.get("meta") if champion.get("meta") else None),
            ("historical_best", session_parent_candidates.get("historical_best")),
            ("session_best", session_parent_candidates.get("session_best")),
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
                summary_path = write_iteration_summary({"similarity_report": similarity_report})
                update_session_state_after_round(summary_path=summary_path)
                leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": "自动拒绝：与失败黑名单高度相似"})
                continue
            if not args.auto_approve:
                similarity_report["user_prompt_required"] = True
                similarity_report["decision"] = "user_prompt"
                print("新策略与历史失败黑名单高度相似，且不是当前 session_parent。")
                print("建议重新生成。")
                if not ask_confirm("是否仍然执行回测", False, args):
                    similarity_report["decision"] = "user_reject"
                    write_json(version_dir / "similarity_report.json", similarity_report)
                    invalid_reason = "用户拒绝回测相似失败策略。"
                    summary_path = write_iteration_summary({"similarity_report": similarity_report})
                    update_session_state_after_round(summary_path=summary_path)
                    leaderboard.append({"version": ver, "run_id": run_id, "strategy_class": class_name, "is_valid": False, "invalid_reason": "用户拒绝回测相似失败策略。"})
                    continue
                similarity_report["decision"] = "user_override_continue"
        print(f"- 是否允许自动继续：{'是' if similarity_report['decision'] in {'auto_continue', 'user_override_continue'} else '否'}")
        write_json(version_dir / "similarity_report.json", similarity_report)

        duplicate_report = {
            "is_duplicate": False,
            "strategy_fingerprint": strategy_fingerprint,
            "matched_hash": "",
            "matched_version": "",
            "matched_strategy_class": "",
            "similarity_score": 0.0,
            "reasons": [],
            "decision": "continue",
            "repeated_fingerprint_fields": [],
            "mutation_type": mutation_type,
            "expected_changes": _mutation_expected_changes(strategy_spec),
            "detected_changes": [],
            "why_rejected": "",
        }
        matched_tested: dict[str, Any] | None = None
        for tested in tested_strategy_fingerprints:
            tested_fp = tested.get("strategy_fingerprint", {}) or {}
            tested_features = tested.get("features", {}) or {}
            fp_equal = str((tested_fp.get("hash") or "")) == str(strategy_fingerprint.get("hash"))
            fp_score, fp_reasons = strategy_similarity(features, tested_features)
            if fp_equal or fp_score >= 0.92:
                duplicate_report.update({
                    "is_duplicate": True,
                    "matched_hash": str(tested_fp.get("hash") or ""),
                    "matched_version": str(tested.get("version") or ""),
                    "matched_strategy_class": str(tested.get("strategy_class") or ""),
                    "similarity_score": 1.0 if fp_equal else round(float(fp_score), 4),
                    "reasons": (["strategy_fingerprint 完全相同"] if fp_equal else fp_reasons),
                    "decision": "skip_backtest",
                    "repeated_fingerprint_fields": repeated_fingerprint_fields(features, tested_features),
                    "detected_changes": _detected_feature_changes(features, tested_features),
                    "why_rejected": "当前策略 fingerprint 与本次 run 已测试策略完全相同或相似度达到重复阈值，跳过正式 strategy.py 保存与回测。",
                })
                matched_tested = tested
                break
        write_json(version_dir / "strategy_fingerprint.json", strategy_fingerprint)
        write_json(version_dir / "duplicate_report.json", duplicate_report)
        print("========== 重复判定拆分 ==========")
        print(f"historical_similarity: similar_to={similarity_report['similar_to'] or '无'}, score={float(similarity_report['similarity_score']):.2f}, decision={similarity_report['decision']}")
        print(f"current_run_similarity: duplicate_with={duplicate_report['matched_version'] or '无'}, score={float(duplicate_report['similarity_score']):.2f}, is_duplicate={'是' if duplicate_report['is_duplicate'] else '否'}")
        print(f"final_duplicate_decision: {duplicate_report['decision']}")
        if duplicate_report["is_duplicate"]:
            duplicate_strategy_path = version_dir / "strategy.duplicate.py"
            duplicate_strategy_path.write_text(code, encoding="utf-8")
            matched_code = ""
            matched_version = duplicate_report.get("matched_version") or ""
            if matched_version:
                matched_strategy_path = run_dir / str(matched_version) / "strategy.py"
                if matched_strategy_path.exists():
                    matched_code = matched_strategy_path.read_text(encoding="utf-8", errors="ignore")
            if not matched_code and matched_tested:
                matched_code = str(matched_tested.get("code") or "")
            duplicate_reason = {
                "duplicate_with_version": duplicate_report.get("matched_version", ""),
                "similarity_score": duplicate_report.get("similarity_score", 0.0),
                "repeated_fingerprint_fields": duplicate_report.get("repeated_fingerprint_fields", []),
                "mutation_type": mutation_type,
                "expected_changes": duplicate_report.get("expected_changes", []),
                "detected_changes": duplicate_report.get("detected_changes", []),
                "why_rejected": duplicate_report.get("why_rejected", ""),
            }
            write_json(version_dir / "duplicate_reason.json", duplicate_reason)
            (version_dir / "duplicate_diff_summary.txt").write_text(
                build_duplicate_diff_summary(
                    version=ver,
                    matched_version=str(duplicate_report.get("matched_version") or ""),
                    similarity_score=float(duplicate_report.get("similarity_score") or 0.0),
                    mutation_type=mutation_type,
                    expected_changes=list(duplicate_report.get("expected_changes") or []),
                    detected_changes=list(duplicate_report.get("detected_changes") or []),
                    repeated_fields=list(duplicate_report.get("repeated_fingerprint_fields") or []),
                    current_code=code,
                    matched_code=matched_code,
                ),
                encoding="utf-8",
            )
            print(f"重复策略草稿已保存：{duplicate_strategy_path}")
            print(f"重复原因已保存：{version_dir / 'duplicate_reason.json'}")
            print(f"重复 diff 摘要已保存：{version_dir / 'duplicate_diff_summary.txt'}")
            invalid_reason = "策略与本次 run 已测试策略高度重复"
            previous_failure_reason = invalid_reason
            status["train_backtest_status"] = "跳过"
            status["validation_backtest_status"] = "跳过"
            status["is_valid"] = False
            status["is_best"] = False
            status["invalid_reason"] = invalid_reason
            status["final_score"] = 0.0
            iteration_stats["invalid_strategy_count"] += 1
            print("策略与本次 run 已测试策略高度重复，跳过训练/验证回测。")
            print(f"重复对象：{duplicate_report['matched_version']} {duplicate_report['matched_strategy_class']}，相似度 {float(duplicate_report['similarity_score']):.2f}")
            summary_path = write_iteration_summary({
                "features": features,
                "strategy_fingerprint": strategy_fingerprint,
                "duplicate_report": duplicate_report,
                "final_score": 0.0,
                "invalid_reason": invalid_reason,
            })
            update_session_state_after_round(summary_path=summary_path)
            leaderboard.append(enrich_leaderboard_entry({
                "version": ver,
                "run_id": run_id,
                "strategy_class": class_name,
                "strategy_file": str(strategy_file),
                "strategy_fingerprint": strategy_fingerprint,
                "duplicate_report": duplicate_report,
                "is_valid": False,
                "is_best": False,
                "invalid_reason": invalid_reason,
                "final_score": 0.0,
            }))
            flush_iteration_stats()
            continue

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
                invalid_reason = "Python 语法检查失败"
                previous_failure_reason = invalid_reason
                summary_path = write_iteration_summary()
                update_session_state_after_round(summary_path=summary_path)
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
                "strategy_fingerprint": strategy_fingerprint,
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
            summary_path = write_iteration_summary({"features": features})
            update_session_state_after_round(summary_path=summary_path, strategy_record=card)
            continue
        status["static_check_status"] = "成功"
        tested_strategy_fingerprints.append({
            "version": ver,
            "strategy_class": class_name,
            "strategy_fingerprint": strategy_fingerprint,
            "features": features,
            "strategy_file": str(version_dir / "strategy.py"),
            "code": code,
        })

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
        _write_backtest_process_log(version_dir, "train", train.timerange, train_cp)
        (version_dir / "backtest_logs.txt").write_text(
            f"[Train {train.timerange}]\nRETURNCODE: {train_cp.returncode}\nSTDOUT:\n{train_cp.stdout}\n\nSTDERR:\n{train_cp.stderr}\n",
            encoding="utf-8",
        )
        if train_cp.returncode != 0:
            print(train_cp.stderr)
            status["train_backtest_status"] = "失败"
            invalid_reason = "训练区间回测失败"
            _record_backtest_error(
                backtest_errors,
                stage="train",
                timerange=train.timerange,
                expected_strategy=class_name,
                error="backtest_failed",
            )
            summary_path = write_iteration_summary({"backtest_errors": backtest_errors})
            update_session_state_after_round(summary_path=summary_path)
            iteration_stats["invalid_strategy_count"] += 1
            status["is_valid"] = False
            status["invalid_reason"] = invalid_reason
            flush_iteration_stats()
            continue
        print("正在解析回测结果……")
        train_zip, train_candidates = find_backtest_zip_for_strategy(results_dir, class_name, train_start_ts, train_before_zips)
        if train_zip is None:
            _log_backtest_zip_filter_failure(class_name, train_candidates)
            actual = []
            wrong_zip = ""
            if train_candidates:
                wrong_zip = str(train_candidates[0].get("zip") or "")
                actual = list(train_candidates[0].get("actual_strategies") or [])
            _record_backtest_error(backtest_errors, stage="train", timerange=train.timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
            _print_backtest_mismatch_summary(backtest_errors)
            status["train_backtest_status"] = "解析失败"
            invalid_reason = "训练区间回测结果 zip 不匹配或缺失"
            summary_path = write_iteration_summary({"backtest_errors": backtest_errors})
            update_session_state_after_round(summary_path=summary_path)
            iteration_stats["invalid_strategy_count"] += 1
            status["is_valid"] = False
            status["invalid_reason"] = invalid_reason
            flush_iteration_stats()
            continue
        train_zip_local = _copy_backtest_zip_to_version(train_zip, version_dir, "train", train.timerange)
        train_result, actual_train_keys = parse_backtest_from_zip(train_zip_local, class_name, strict=False)
        if train_result is None:
            _record_backtest_error(backtest_errors, stage="train", timerange=train.timerange, expected_strategy=class_name, wrong_zip=str(train_zip), actual_strategies=actual_train_keys, error="backtest_parse_failed")
            _print_backtest_mismatch_summary(backtest_errors)
            status["train_backtest_status"] = "解析失败"
            invalid_reason = "训练区间回测结果解析失败"
            summary_path = write_iteration_summary({"backtest_errors": backtest_errors})
            update_session_state_after_round(summary_path=summary_path)
            iteration_stats["invalid_strategy_count"] += 1
            status["is_valid"] = False
            status["invalid_reason"] = invalid_reason
            flush_iteration_stats()
            continue
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
                _write_backtest_process_log(version_dir, f"validation_{p.name}", p.timerange, val_cp)
                with (version_dir / "backtest_logs.txt").open("a", encoding="utf-8") as logf:
                    logf.write(f"\n[Validation {p.name} {p.timerange}]\nRETURNCODE: {val_cp.returncode}\nSTDOUT:\n{val_cp.stdout}\n\nSTDERR:\n{val_cp.stderr}\n")
                if val_cp.returncode != 0:
                    print(val_cp.stderr)
                    _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, error="backtest_failed")
                    validation_status = "backtest_failed"
                    invalid_reason = "验证区间回测失败"
                    break
                print("正在解析回测结果……")
                val_zip, val_candidates = find_backtest_zip_for_strategy(results_dir, class_name, val_start_ts, val_before_zips)
                if val_zip is None:
                    _log_backtest_zip_filter_failure(class_name, val_candidates)
                    actual = []
                    wrong_zip = ""
                    if val_candidates:
                        wrong_zip = str(val_candidates[0].get("zip") or "")
                        actual = list(val_candidates[0].get("actual_strategies") or [])
                    _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
                    _print_backtest_mismatch_summary(backtest_errors)
                    validation_status = "backtest_parse_failed"
                    invalid_reason = "验证区间回测结果 zip 不匹配或缺失"
                    break
                val_zip_local = _copy_backtest_zip_to_version(val_zip, version_dir, "validation", p.timerange)
                val_result, actual_val_keys = parse_backtest_from_zip(val_zip_local, class_name, strict=False)
                if val_result is None:
                    _record_backtest_error(backtest_errors, stage="validation", timerange=p.timerange, expected_strategy=class_name, wrong_zip=str(val_zip), actual_strategies=actual_val_keys, error="backtest_parse_failed")
                    _print_backtest_mismatch_summary(backtest_errors)
                    validation_status = "backtest_parse_failed"
                    invalid_reason = "验证区间回测结果解析失败"
                    break
                vm = _extract_metrics(val_result)
                validation_metrics.append({"period": p.name, "timerange": p.timerange, "metrics": vm})
                iteration_stats["validation_backtest_total_count"] += 1
                _print_round_table(ver, p.timerange, vm)
                val_scores.append(_score(vm, p))
            if validation_status in {"backtest_failed", "backtest_parse_failed"}:
                status["validation_backtest_status"] = "失败"
            elif validation_metrics:
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
        if backtest_errors:
            is_valid = False
            is_best = False
            final_score = 0
            invalid_reason = invalid_reason or "回测结果解析失败或 zip 不匹配"
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
        avg_validation_profit_pct, avg_validation_profit_factor, max_validation_drawdown_pct = _aggregate_validation_metrics(validation_metrics)
        is_best = is_valid and final_score > 0 and (best is None or final_score > float(best["final_score"]))
        not_best_reason = ""
        official_gate_reasons = _official_best_hard_gate_reasons(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            avg_validation_profit_pct=avg_validation_profit_pct,
            avg_validation_profit_factor=avg_validation_profit_factor,
            max_validation_drawdown_pct=max_validation_drawdown_pct,
            current_official_best=session_state.get("official_champion"),
        )
        if is_best and official_gate_reasons:
            is_best = False
            not_best_reason = "；".join(official_gate_reasons)
            print("official best 硬门槛未通过，本轮只能作为 nearest_candidate，不能覆盖 best_strategy.json。")
            for reason in official_gate_reasons:
                print(f"- {reason}")
        behavior_duplicate = _behavior_duplicate_report(
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            official_best=session_state.get("official_champion"),
        )
        is_behavior_duplicate = bool(behavior_duplicate.get("is_duplicate"))
        if is_best and is_behavior_duplicate:
            is_best = False
            not_best_reason = "策略行为与当前 official_best 几乎一致，不覆盖 best。"
            print("行为级重复检测：与当前 official_best 指标几乎一致，不更新 best。")
        reason_detail = [] if is_best else (
            [not_best_reason] if not_best_reason else _build_not_best_reason_detail(
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                final_score=final_score,
                champion_meta=champion.get("meta", {}) or {},
                target_cfg=target_cfg,
                invalid_reason=invalid_reason,
            )
        )

        status["is_valid"] = bool(is_valid)
        status["is_best"] = bool(is_best)
        status["invalid_reason"] = str(invalid_reason or "")
        status["not_best_reason"] = not_best_reason
        status["behavior_duplicate"] = behavior_duplicate
        status["is_behavior_duplicate"] = is_behavior_duplicate
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
            session_best = {"version": ver, "class_name": class_name, "final_score": final_score, "is_valid": is_valid, "invalid_reason": invalid_reason, "not_best_reason": not_best_reason}
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
            "not_best_reason": not_best_reason,
            "official_best_hard_gate_reasons": official_gate_reasons,
            "reason_detail": reason_detail,
            "behavior_duplicate": behavior_duplicate,
            "is_behavior_duplicate": is_behavior_duplicate,
            "is_best": is_best,
            "is_valid": is_valid,
            "invalid_reason": invalid_reason,
            "similarity_report": read_json(version_dir / "similarity_report.json") if (version_dir / "similarity_report.json").exists() else {},
            "strategy_fingerprint": strategy_fingerprint,
            "duplicate_report": duplicate_report,
            "trade_under_min": trade_under_min,
            "cannot_be_official_best_unless_validation_strong": cannot_be_official_best_unless_validation_strong,
            "validation_strong": validation_strong,
            "allow_near_min_trades_best": allow_near_min_trades_best,
            "min_trades_grace_ratio": min_trades_grace_ratio,
            "trade_count_warning": trade_count_warning,
            "backtest_errors": backtest_errors,
        }
        summary_path = write_iteration_summary(summary)
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
            "strategy_fingerprint": strategy_fingerprint,
            "duplicate_report": duplicate_report,
            "behavior_duplicate": behavior_duplicate,
            "not_best_reason": not_best_reason,
            "official_best_hard_gate_reasons": official_gate_reasons,
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
                "strategy_fingerprint": strategy_fingerprint,
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
        if is_best and "MultiCoin_AI_Strategy_20260530_014158_v045" in str((session_state.get("official_champion") or {}).get("strategy_class") or ""):
            required_v045_holdouts = [
                {"label": "v045_holdout_202601", "timerange": "20260101-20260131"},
                {"label": "v045_holdout_20260115_20260215", "timerange": "20260115-20260215"},
            ]
            existing_holdout_timeranges = {str(item.get("timerange") or "") for item in holdout_ranges if isinstance(item, dict)}
            for required_holdout in required_v045_holdouts:
                if required_holdout["timerange"] not in existing_holdout_timeranges:
                    holdout_ranges.append(required_holdout)
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
                _write_backtest_process_log(version_dir, f"holdout_{h_label}", h_timerange, hcp)
                if hcp.returncode != 0:
                    _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, error="backtest_failed")
                    holdout_failed = True
                    holdout_status = "backtest_failed"
                    holdout_reason = "holdout 回测失败"
                    break
                hzip, h_candidates = find_backtest_zip_for_strategy(results_dir, class_name, ts, before)
                if hzip is None:
                    _log_backtest_zip_filter_failure(class_name, h_candidates)
                    actual = []
                    wrong_zip = ""
                    if h_candidates:
                        wrong_zip = str(h_candidates[0].get("zip") or "")
                        actual = list(h_candidates[0].get("actual_strategies") or [])
                    _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, wrong_zip=wrong_zip, actual_strategies=actual, error="wrong_strategy_zip_detected" if wrong_zip else "zip_missing")
                    _print_backtest_mismatch_summary(backtest_errors)
                    holdout_failed = True
                    holdout_status = "backtest_parse_failed"
                    holdout_reason = "holdout 回测结果 zip 不匹配或缺失"
                    break
                hzip_local = _copy_backtest_zip_to_version(hzip, version_dir, "holdout", h_timerange, h_label)
                h_result, actual_h_keys = parse_backtest_from_zip(hzip_local, class_name, strict=False)
                if h_result is None:
                    _record_backtest_error(backtest_errors, stage="holdout", timerange=h_timerange, expected_strategy=class_name, wrong_zip=str(hzip), actual_strategies=actual_h_keys, error="backtest_parse_failed")
                    _print_backtest_mismatch_summary(backtest_errors)
                    holdout_failed = True
                    holdout_status = "backtest_parse_failed"
                    holdout_reason = "holdout 回测结果解析失败"
                    break
                hm = _extract_metrics(h_result)
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

        if random_sample_plan.get("enabled"):
            random_sample_metrics, random_sample_errors, random_sample_status = run_random_sample_backtests(
                plan=random_sample_plan,
                train_cmd=train_cmd,
                results_dir=results_dir,
                version_dir=version_dir,
                class_name=class_name,
            )
            random_sample_observation = summarize_random_sample_observation(random_sample_metrics)
            status["random_sample_status"] = random_sample_status
            status["random_sample_count"] = len(random_sample_metrics)
            iteration_stats["random_sample_total_backtests"] += len(random_sample_metrics) + len(random_sample_errors)
            write_json(version_dir / "random_sample_metrics.json", {
                "usage": dict(RANDOM_SAMPLE_USAGE),
                "metrics": random_sample_metrics,
                "observation": random_sample_observation,
                "errors": random_sample_errors,
            })
            print_random_sample_observation(random_sample_metrics)
        else:
            status["random_sample_status"] = "skipped"
            status["random_sample_count"] = 0
        if not is_best and not reason_detail:
            if not_best_reason:
                reason_detail = [not_best_reason]
            else:
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
            "not_best_reason": not_best_reason,
            "official_best_hard_gate_reasons": official_gate_reasons,
            "reason_detail": reason_detail,
            "behavior_duplicate": behavior_duplicate,
            "is_behavior_duplicate": is_behavior_duplicate,
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
            update_session_state_after_round(summary_path=summary_path)
            flush_iteration_stats()
            continue
        leaderboard_entry["is_best"] = bool(is_best)
        leaderboard_entry["is_valid"] = bool(is_valid)
        leaderboard_entry["invalid_reason"] = invalid_reason
        leaderboard_entry["reason_detail"] = reason_detail
        round_strategy_record = _strategy_record_from_round(
            version=ver,
            class_name=class_name,
            strategy_file=strategy_file,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            final_score=final_score,
            is_valid=is_valid,
            invalid_reason=invalid_reason,
            features=features,
        )
        round_strategy_record.update({
            "source_run_id": run_id,
            "avg_validation_metrics": {
                "profit_total_pct": avg_validation_profit_pct,
                "profit_factor": avg_validation_profit_factor,
                "max_drawdown_pct": max_validation_drawdown_pct,
            },
            "score_breakdown": score_breakdown,
            "strategy_fingerprint": strategy_fingerprint,
            "duplicate_report": duplicate_report,
            "behavior_duplicate": behavior_duplicate,
            "not_best_reason": not_best_reason,
            "mutation_type": mutation_type,
            "failure_reason": failure_reason,
        })
        if is_best:
            best = round_data
            iteration_stats["new_best_update_count"] += 1
            best_summary_path = summary_path
            write_json(run_dir / "best_strategy.json", best)
            shutil.copy2(strategy_file, GENERATED_DIR / f"BEST_{strategy_family}.py")
            champion = {"meta": round_strategy_record, "code": code}
            historical_best_mem = round_strategy_record
        update_session_state_after_round(summary_path=summary_path, strategy_record=round_strategy_record, became_best=is_best)
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
        return (not row.get("is_valid")) or bool(row.get("official_best_hard_gate_reasons"))

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
    session_nearest_final = session_state.get("session_nearest_candidate")
    if _is_better_session_nearest(session_nearest_final, nearest_candidate, target_cfg, baseline_cfg):
        nearest_candidate = session_nearest_final
    if nearest_candidate:
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
    all_backtest_errors: list[dict[str, Any]] = []
    random_sample_run_observations: list[dict[str, Any]] = []
    for summary_file in sorted(run_dir.glob("v*/summary.json")):
        try:
            summary_for_errors = read_json(summary_file)
            all_backtest_errors.extend(summary_for_errors.get("backtest_errors", []) or [])
            observation = summary_for_errors.get("random_sample_observation") or {}
            if observation.get("count"):
                random_sample_run_observations.append({
                    "version": summary_file.parent.name,
                    "strategy_class": summary_for_errors.get("strategy_class", ""),
                    **observation,
                })
        except Exception:
            continue
    _print_backtest_mismatch_summary(all_backtest_errors)
    common_failure_patterns = _build_common_failure_patterns(invalid_rows, target_cfg)
    if session_state.get("common_failure_patterns_this_run"):
        common_failure_patterns = sorted(set(common_failure_patterns).union(session_state["common_failure_patterns_this_run"]))[:10]
    nearest_advisor_notes = _build_nearest_advisor_notes(nearest_candidate)
    session_best = session_state.get("session_best") or session_best
    last_run_summary = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(),
        "target": target_cfg,
        "official_best": current_best_saved if 'current_best_saved' in locals() else None,
        "historical_best": historical_best_mem,
        "final_official_champion": session_state.get("official_champion"),
        "final_in_memory_champion": session_state.get("in_memory_champion"),
        "nearest_candidate": nearest_candidate,
        "session_best": session_best,
        "session_nearest_candidate": session_state.get("session_nearest_candidate"),
        "round_history": session_state.get("round_history", []),
        "early_stop": {
            "triggered": bool(iteration_stats.get("early_stop_triggered")),
            "reason": iteration_stats.get("early_stop_reason", ""),
            "counters": iteration_stats.get("early_stop_checked_counters", {}),
        },
        "failed_mutation_types_this_run": session_state.get("failed_mutation_types_this_run", []),
        "successful_mutation_types_this_run": session_state.get("successful_mutation_types_this_run", []),
        "attempted_mutation_types_this_run": session_state.get("attempted_mutation_types_this_run", []),
        "trade_count_warning": trade_count_warning,
        "failed_versions": [r.get("version") for r in invalid_rows],
        "common_failure_patterns": common_failure_patterns,
        "backtest_errors": all_backtest_errors,
        "random_sample_observation_only": True,
        "random_sample_summary": {
            "enabled": bool(random_sample_plan.get("enabled")),
            "windows_count": len(random_sample_plan.get("windows", []) or []),
            "used_for_final_score": False,
            "used_for_best_selection": False,
            "used_for_ai_prompt": False,
            "observations": random_sample_run_observations,
        },
        "nearest_advisor_notes": nearest_advisor_notes,
        "recommended_next_mutation_types": ["add_entry_filter", "tighten_entry_trigger", "remove_bad_entry_condition", "pair_specific_filter", "tag_specific_filter"],
        "forbidden_next_mutation_types": sorted(set(used_failed_mutations).union(session_state.get("failed_mutation_types_this_run", []))),
        "worst_pairs": sorted(best_pair_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")))[:5],
        "best_pairs": sorted(best_pair_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")), reverse=True)[:5],
        "worst_entry_tags": sorted(best_entry_tag_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")))[:5],
        "best_entry_tags": sorted(best_entry_tag_metrics, key=lambda x: _safe_float(x.get("profit_total_abs")), reverse=True)[:5],
        "lessons_for_next_run": [
            "不要重新生成完全不同策略",
            "优先围绕 nearest_candidate 和 historical_best 做单点小步调整",
            _runtime_trade_target_text(runtime_goal),
            f"如果 nearest_candidate 交易数超标，下一轮目标是压回 {_runtime_trade_target_values(runtime_goal)['min_trades']}~{_runtime_trade_target_values(runtime_goal)['max_trades']}，且避免高于 {_runtime_trade_target_values(runtime_goal)['max_trades']}。",
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
    write_json(run_dir / "last_run_summary.json", last_run_summary)

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
        session_state["official_champion"] = current_best_saved
        session_state["in_memory_champion"] = current_best_saved
        if session_state.get("session_best"):
            session_state["session_best"] = current_best_saved
            session_best = current_best_saved
        last_run_summary["official_best"] = current_best_saved
        last_run_summary["final_official_champion"] = session_state.get("official_champion")
        last_run_summary["final_in_memory_champion"] = session_state.get("in_memory_champion")
        last_run_summary["session_best"] = session_best
        write_json(LAST_RUN_SUMMARY_FILE, last_run_summary)
        write_json(run_dir / "last_run_summary.json", last_run_summary)
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
        print(f"- {_runtime_trade_target_text(runtime_goal)}")
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
        session_best_label = _strategy_label(session_best)
        official_best_label = _strategy_label(current_best_saved) if current_best_saved else ""
        if current_best_saved and session_best_label and session_best_label == official_best_label:
            print(f"本轮 session best：策略名={session_best_label}；final_score={_safe_float(session_best.get('final_score')):.4f}；已成为正式 best")
        else:
            reason = session_best.get("not_best_reason") or session_best.get("invalid_reason") or ("通过但未超过 official_best 硬门槛/综合得分" if session_best.get("is_valid") else "未通过有效性约束")
            print(f"本轮 session best：策略名={session_best_label}；final_score={_safe_float(session_best.get('final_score')):.4f}；未成为正式 best 原因：{reason}")
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
    print(f"当前策略迭代版本：{iteration_stats.get('current_iteration_version')}（仅本次 run 内版本号）")
    print(f"strategy_memory 当前保留条数：{iteration_stats.get('strategy_memory_retained_count')}")
    global_stats = iteration_stats.get("global_stats") or {}
    print("\n========== 全项目累计统计 ==========")
    print(f"累计 run 总数：{global_stats.get('run_count', 0)}")
    print(f"累计有效 run 数：{global_stats.get('nonempty_run_count', 0)}")
    print(f"累计尝试版本数/v目录数：{global_stats.get('version_dir_count', 0)}")
    print(f"累计生成 strategy.py 数：{global_stats.get('strategy_file_count', 0)}")
    print(f"累计生成 mutation_spec.json 数：{global_stats.get('mutation_spec_count', 0)}")
    print(f"累计训练回测版本数：{global_stats.get('train_backtested_count', 0)}")
    print(f"累计验证回测版本数：{global_stats.get('validation_backtested_count', 0)}")
    print(f"累计 summary.json 数：{global_stats.get('summary_count', 0)}")
    print(f"累计有效策略数：{global_stats.get('valid_strategy_count', 0)}")
    print(f"累计 new best 次数：{global_stats.get('new_best_count', 0)}")
    print(f"strategy_memory 当前保留条数：{global_stats.get('strategy_memory_retained_count', iteration_stats.get('strategy_memory_retained_count'))}")
    print("说明：strategy_memory 保留条数不是累计策略版本数。")
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



def acquire_auto_optimize_lock(allow_concurrent_runs: bool) -> Any | None:
    if allow_concurrent_runs:
        return None
    AUTO_OPTIMIZE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = AUTO_OPTIMIZE_LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("检测到已有自动优化任务正在运行。")
        print("为避免回测 zip 互相污染，本次任务已中止。")
        print("如确认没有任务在运行，可删除锁文件：")
        print(f"rm -f {AUTO_OPTIMIZE_LOCK_FILE}")
        lock_fp.close()
        return False
    lock_fp.seek(0)
    lock_fp.truncate()
    lock_fp.write(f"pid={os.getpid()} started_at={datetime.utcnow().isoformat()}Z\n")
    lock_fp.flush()
    return lock_fp


def release_auto_optimize_lock(lock_fp: Any | None) -> None:
    if not lock_fp:
        return
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fp.close()

def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="ai_tools/optimization_goal.json")
    parser.add_argument("--mode", choices=["optimize", "pair-scan"], default="optimize", help="运行模式：optimize=AI 策略优化，pair-scan=用当前 best 策略筛选交易对")
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
    parser.add_argument("--setup", type=lambda value: parse_cli_y_or_n(value, "--setup"), default="n", help="是否启动后直接进入交互式设置：y=进入，n=跳过；默认 n")
    parser.add_argument("--confirm", type=lambda value: parse_cli_y_or_n(value, "--confirm"), default="n", help="是否开启中途人工确认：y=询问，n=自动选择默认跳过；默认 n")
    parser.add_argument("--no-wizard", action="store_true")
    parser.add_argument("--save-goal", action="store_true")
    parser.add_argument("--config", default="user_data/config.5coins.json")
    parser.add_argument("--pairs-file", default=None, help="optimize 模式下读取指定 recommended_pairs.json，并使用 active_pairs 覆盖本次 pair_whitelist")
    parser.add_argument("--ignore-recommended-pairs", action="store_true", default=False, help="optimize 模式下忽略 user_data/ai_memory/recommended_pairs.json，使用原始 config 的 pair_whitelist")
    parser.add_argument("--refresh-pairs", action="store_true", default=False, help="optimize 模式开始前先执行一次 pair-scan，生成最新 recommended_pairs.json 并用于本次优化")
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
    parser.add_argument("--manual-ai-prepare", action="store_true", default=False, help="只生成半自动 AI 任务包，不调用 AI，不回测新策略")
    parser.add_argument("--manual-task-name", default=None, help="半自动任务包目录名；默认 run_YYYYMMDD_HHMMSS")
    parser.add_argument("--manual-ai-run", default=None, help="读取 Codex 生成的策略文件，执行本地回测和 best 判断；不调用 AI")
    parser.add_argument("--manual-ai-task-dir", default=None, help="半自动任务包目录，可用于读取 generated_mutation_spec.json 等上下文")
    parser.add_argument("--manual-git-push", action="store_true", default=False, help="生成任务包后自动 git add / commit / push")
    parser.add_argument("--manual-git-branch", default=None, help="半自动任务包 Git 分支；默认 ai-manual/<task_name>")
    parser.add_argument("--random-sample-windows", type=int, default=0, help="每个策略在正式训练/验证/holdout/best 判断后额外随机采样回测窗口数；默认 0 关闭")
    parser.add_argument("--early-stop-patience", type=int, default=12, help="连续 N 轮没有新 best 时提前停止；默认 12，<=0 关闭")
    parser.add_argument("--early-stop-final-score-failures", type=int, default=8, help="连续 N 轮 final_score<=0 时提前停止；默认 8，<=0 关闭")
    parser.add_argument("--early-stop-duplicate-strategies", type=int, default=3, help="连续 N 轮策略 fingerprint 与本次 run 已测策略高度重复时提前停止；默认 3，<=0 关闭")
    parser.add_argument("--random-sample-min-days", type=int, default=25, help="随机采样窗口最小天数，默认 25")
    parser.add_argument("--random-sample-max-days", type=int, default=35, help="随机采样窗口最大天数，默认 35")
    parser.add_argument("--random-sample-data-start", default=None, help="随机采样数据起始日期 YYYYMMDD，例如 20260101")
    parser.add_argument("--random-sample-data-end", default=None, help="随机采样数据结束日期 YYYYMMDD，例如 20260601")
    parser.add_argument("--random-sample-seed", default=None, help="随机采样种子；默认 None")
    log_push_group = parser.add_mutually_exclusive_group()
    log_push_group.add_argument("--push-logs-to-git", dest="push_logs_to_git", action="store_true", default=None, help="强制开启运行日志推送到独立日志仓库")
    log_push_group.add_argument("--no-push-logs-to-git", dest="push_logs_to_git", action="store_false", help="强制关闭运行日志推送到独立日志仓库")
    parser.add_argument("--log-repo-path", default=None, help="覆盖 LOG_REPO_PATH，指定独立日志仓库本地路径")
    parser.add_argument("--allow-concurrent-runs", action="store_true", default=False, help="允许多个 auto_optimize_strategy.py 并发运行（默认禁止，避免回测 zip 污染）")
    args = parser.parse_args()
    if args.manual_ai_prepare and args.manual_ai_run:
        parser.error("--manual-ai-prepare 和 --manual-ai-run 不能同时使用")
    if args.mode == "pair-scan" and (args.manual_ai_prepare or args.manual_ai_run):
        parser.error("--mode pair-scan 不能与 --manual-ai-prepare/--manual-ai-run 同时使用")
    if args.mode == "pair-scan" and args.pairs_file:
        parser.error("--pairs-file 仅用于 optimize 模式，pair-scan 请使用 pair_selection.candidate_pairs")
    if args.mode == "pair-scan" and args.ignore_recommended_pairs:
        parser.error("--ignore-recommended-pairs 仅用于 optimize 模式")
    if args.mode == "pair-scan" and args.refresh_pairs:
        parser.error("--refresh-pairs 仅用于 optimize 模式；pair-scan 模式本身已执行交易对筛选")
    if args.random_sample_windows < 0:
        parser.error("--random-sample-windows 不能小于 0")
    if args.random_sample_windows > 0:
        if not args.random_sample_data_start or not args.random_sample_data_end:
            parser.error("启用 --random-sample-windows 时必须同时提供 --random-sample-data-start 和 --random-sample-data-end")
        if args.random_sample_min_days <= 0 or args.random_sample_max_days < args.random_sample_min_days:
            parser.error("--random-sample-min-days 必须大于 0，且不能大于 --random-sample-max-days")
        try:
            start = _parse_yyyymmdd(args.random_sample_data_start, "--random-sample-data-start")
            end = _parse_yyyymmdd(args.random_sample_data_end, "--random-sample-data-end")
        except ValueError as exc:
            parser.error(str(exc))
        if end <= start:
            parser.error("--random-sample-data-end 必须晚于 --random-sample-data-start")
    lock_fp = acquire_auto_optimize_lock(args.allow_concurrent_runs)
    if lock_fp is False:
        return
    configure_log_repo_args(args)

    run_dir = RESULT_ROOT / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args._run_start_ts = time.time()
    log_ctx = setup_terminal_logging(run_dir, args)
    args._log_context = log_ctx
    try:
        print_log_start_banner(log_ctx)
        print_interaction_config(args)
        check_log_repo_startup(args)
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

        if args.mode == "optimize":
            apply_pairs_file_override(runtime_goal, args, run_dir)
            apply_auto_trade_count_target(runtime_goal)
            maybe_reset_best_strategy(args.reset_best)
        elif args.reset_best:
            print("当前为 pair-scan 模式：忽略 --reset-best，不覆盖 best_strategy.json。")

        if args.manual_ai_prepare:
            effective_iterations = 0
        elif args.manual_ai_run:
            effective_iterations = 1
        elif args.mode == "pair-scan":
            effective_iterations = 0
        else:
            effective_iterations = args.iterations if args.iterations is not None else int(runtime_goal.get("max_iterations", 5))
        runtime_goal["max_iterations"] = int(effective_iterations)
        print_prompt_guidance_summary(runtime_goal)
        print(f"本次实际迭代轮数：{int(effective_iterations)}")

        if args.save_goal:
            write_json(goal_path, runtime_goal)
            print(f"已保存修改后的目标配置到：{goal_path}")

        write_json(run_dir / "goal.runtime.json", runtime_goal)
        write_json(run_dir / "goal.json", goal)
        print(f"运行时配置已保存：{run_dir / 'goal.runtime.json'}")

        if args.mode == "pair-scan":
            run_pair_scan(runtime_goal, args, run_dir)
        elif args.manual_ai_prepare:
            prepare_manual_ai_task(runtime_goal, args)
            print_log_saved_summary(args)
        elif args.manual_ai_run:
            run_manual_ai_backtest(runtime_goal, args, run_dir)
        else:
            run_auto_optimization(runtime_goal, args, run_dir)
    except BaseException:
        if log_ctx is not None:
            print("\n========== 异常 traceback ==========")
        traceback.print_exc()
        raise
    finally:
        if log_ctx is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                _update_latest_log(log_ctx.latest_log_path, log_ctx.global_log_path)
            except OSError as exc:
                print(f"更新 latest.log 失败：{exc}", file=sys.stderr)
        log_push_result = push_run_logs_to_log_repo(run_dir, args)
        print_log_repo_push_summary(log_push_result)
        restore_terminal_logging(log_ctx)
        release_auto_optimize_lock(lock_fp)


if __name__ == "__main__":
    main()
