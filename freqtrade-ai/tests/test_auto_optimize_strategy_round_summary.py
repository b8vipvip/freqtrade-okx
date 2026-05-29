from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ai_tools" / "auto_optimize_strategy.py"
spec = importlib.util.spec_from_file_location("auto_optimize_strategy", MODULE_PATH)
assert spec and spec.loader
optimizer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = optimizer
spec.loader.exec_module(optimizer)


def _metrics(total_trades: int, profit: float = 2.0) -> dict:
    return {
        "total_trades": total_trades,
        "profit_total_abs": profit,
        "profit_total": profit / 100.0,
        "profit_total_pct": profit,
        "profit_factor": 1.2,
        "max_drawdown": 0.01,
        "max_drawdown_pct": 1.0,
        "winrate": 0.55,
        "pairs": [],
        "entry_tags": [],
    }


def _summary_for_state(state: dict) -> dict:
    return optimizer._minimal_round_summary(
        version="v001",
        strategy_class="TestStrategy_v001",
        strategy_file="user_data/strategies/TestStrategy_v001.py",
        state=state,
        mutation_type="unit_test",
        failure_reason=state.get("invalid_reason", ""),
    )


def test_total_trades_zero_skips_validation_and_holdout_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state["train_metrics"] = _metrics(0)
    optimizer._apply_train_hard_constraint_status(state, "训练区间无交易", "训练区间无交易")

    summary = _summary_for_state(state)

    assert summary["validation_metrics"] == []
    assert summary["validation_status"] == "skipped"
    assert summary["holdout_metrics"] == []
    assert summary["holdout_status"] == "skipped"
    assert summary["final_score"] == 0
    assert summary["invalid_reason"] == "训练区间无交易"


def test_total_trades_below_min_trades_skips_validation_and_holdout_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state["train_metrics"] = _metrics(10)
    optimizer._apply_train_hard_constraint_status(
        state,
        "训练区间交易数低于目标下限",
        "训练区间交易数低于目标下限",
    )

    summary = _summary_for_state(state)

    assert summary["validation_metrics"] == []
    assert summary["validation_status"] == "skipped"
    assert summary["validation_skip_reason"] == "训练区间交易数低于目标下限"
    assert summary["holdout_metrics"] == []
    assert summary["holdout_status"] == "skipped"
    assert summary["holdout_reason"] == "训练区间触发硬约束，跳过验证和 holdout"


def test_total_trades_above_severe_max_skips_validation_and_holdout_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state["train_metrics"] = _metrics(121)
    optimizer._apply_train_hard_constraint_status(
        state,
        "训练区间交易数严重超过目标上限",
        "训练区间交易数严重超过目标上限",
    )

    summary = _summary_for_state(state)

    assert summary["validation_status"] == "skipped"
    assert summary["validation_metrics"] == []
    assert summary["holdout_status"] == "skipped"
    assert summary["holdout_metrics"] == []
    assert summary["is_valid"] is False


def test_train_passes_but_validation_fails_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state.update(
        {
            "train_metrics": _metrics(40),
            "validation_metrics": [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(0)}],
            "validation_status": "completed",
            "holdout_status": "not_run",
            "holdout_reason": "未达到候选 best 条件，未执行 holdout",
            "is_valid": False,
            "invalid_reason": "所有验证区间无交易",
            "final_score": 0,
        }
    )

    summary = _summary_for_state(state)

    assert summary["validation_status"] == "completed"
    assert summary["holdout_metrics"] == []
    assert summary["holdout_status"] == "not_run"
    assert summary["invalid_reason"] == "所有验证区间无交易"


def test_train_and_validation_pass_but_holdout_not_executed_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state.update(
        {
            "train_metrics": _metrics(40),
            "validation_metrics": [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(20)}],
            "validation_status": "completed",
            "is_valid": True,
            "is_best": False,
            "final_score": 12.5,
        }
    )

    summary = _summary_for_state(state)

    assert summary["validation_status"] == "completed"
    assert summary["holdout_metrics"] == []
    assert summary["holdout_status"] == "not_run"
    assert summary["holdout_reason"] == "未达到候选 best 条件，未执行 holdout"


def test_train_and_validation_pass_with_holdout_executed_without_unboundlocalerror() -> None:
    state = optimizer._new_round_defaults()
    state.update(
        {
            "train_metrics": _metrics(40),
            "validation_metrics": [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(20)}],
            "validation_status": "completed",
            "holdout_metrics": [{"label": "holdout_01", "timerange": "20240201-20240229", "metrics": _metrics(18)}],
            "holdout_status": "completed",
            "holdout_reason": "",
            "is_valid": True,
            "is_best": True,
            "final_score": 15.0,
        }
    )

    summary = _summary_for_state(state)

    assert summary["holdout_status"] == "completed"
    assert len(summary["holdout_metrics"]) == 1
    assert summary["is_best"] is True
    assert summary["is_valid"] is True


def test_near_min_trades_can_continue_as_candidate_with_warning() -> None:
    state = optimizer._new_round_defaults()
    state.update(
        {
            "train_metrics": _metrics(24, profit=-0.02),
            "validation_metrics": [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(30, profit=3.0)}],
            "validation_status": "completed",
            "trade_under_min": True,
            "cannot_be_official_best_unless_validation_strong": True,
            "validation_strong": True,
            "trade_count_warning": "训练交易数略低于目标，但表现接近打平，建议下一轮在保持质量基础上略微增加信号。",
            "is_valid": False,
            "invalid_reason": "训练区间交易数略低于目标下限，仅作为候选参考",
            "final_score": 8.0,
        }
    )

    summary = _summary_for_state(state)

    assert summary["validation_status"] == "completed"
    assert summary["trade_under_min"] is True
    assert summary["cannot_be_official_best_unless_validation_strong"] is True
    assert summary["validation_strong"] is True
    assert summary["trade_count_warning"] == "训练交易数略低于目标，但表现接近打平，建议下一轮在保持质量基础上略微增加信号。"
    assert summary["holdout_status"] == "not_run"


def test_validation_strong_requires_profit_pf_and_drawdown_targets() -> None:
    baseline = {"profit_total_pct": -0.74, "profit_factor": 0.63}
    target = {"max_drawdown_pct": 3.0}
    strong_validation = [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(30, profit=2.0)}]
    weak_validation = [{"period": "valid_01", "timerange": "20240101-20240131", "metrics": _metrics(30, profit=-1.0)}]

    assert optimizer._is_validation_strong(strong_validation, baseline, target) is True
    assert optimizer._is_validation_strong(weak_validation, baseline, target) is False
