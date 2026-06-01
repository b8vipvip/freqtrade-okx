from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_tools" / "auto_optimize_strategy.py"
spec = importlib.util.spec_from_file_location("auto_optimize_strategy_confirm", MODULE_PATH)
assert spec and spec.loader
optimizer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = optimizer
spec.loader.exec_module(optimizer)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"confirm": "n", "start": None, "assume_yes": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_per_iteration_confirm_n_auto_skips_and_logs_separate_context(capsys) -> None:
    result = optimizer.ask_confirm("是否仍然执行回测", False, _args(confirm="n"))

    output = capsys.readouterr().out
    assert result is False
    assert "[per_iteration_confirm]" in output
    assert "--confirm n" in output


def test_setup_wizard_final_confirm_ignores_confirm_n_and_waits_for_user(monkeypatch, capsys) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)

    result = optimizer.ask_setup_wizard_final_confirm(_args(confirm="n"))

    output = capsys.readouterr().out
    assert result is True
    assert prompts == ["是否开始自动优化（默认：是，输入 y/n）："]
    assert "[setup_wizard_final_confirm]" in output
    assert "不会自动取消最终启动" in output
    assert "自动选择不确认/跳过" not in output


def test_setup_wizard_final_confirm_can_be_automated_with_start_or_assume_yes(monkeypatch) -> None:
    def fail_input(prompt: str) -> str:
        raise AssertionError(f"should not prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fail_input)

    assert optimizer.ask_setup_wizard_final_confirm(_args(start="y")) is True
    assert optimizer.ask_setup_wizard_final_confirm(_args(start="n")) is False
    assert optimizer.ask_setup_wizard_final_confirm(_args(assume_yes=True)) is True
