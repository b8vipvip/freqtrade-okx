# -*- coding: utf-8 -*-
"""Shared utility functions for AI analysis and strategy generation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


def load_settings() -> Dict[str, str]:
    load_dotenv()
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-5.5").strip(),
        "BACKTEST_FILE": os.getenv("BACKTEST_FILE", "user_data/backtest_results/backtest-result.json").strip(),
        "AI_ANALYSIS_FILE": os.getenv("AI_ANALYSIS_FILE", "user_data/backtest_results/ai-analysis.txt").strip(),
        "AI_STRATEGY_FILE": os.getenv("AI_STRATEGY_FILE", "user_data/strategies/AI_Generated_Strategy.py").strip(),
    }


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text_file(path: str, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def ensure_file_exists(path: str, message: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{message}: {path}")
