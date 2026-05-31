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


def test_not_best_reason_detail_explains_mixed_validation_and_stoploss() -> None:
    train = _metrics(58, profit=-0.7638)
    train.update({
        "profit_total_abs": -7.6382,
        "profit_factor": 0.5795,
        "max_drawdown_pct": 1.4238,
        "roi_profit_abs": 9.0999,
        "stop_loss_profit_abs": -18.1642,
    })
    validations = [
        {"period": "valid_202604", "timerange": "20260401-20260430", "metrics": {**_metrics(60, profit=0.4043), "profit_total_abs": 4.0427, "profit_factor": 1.3618}},
        {"period": "valid_202603", "timerange": "20260301-20260331", "metrics": {**_metrics(42, profit=-1.5265), "profit_total_abs": -15.2649, "profit_factor": 0.3929}},
        {"period": "valid_202602", "timerange": "20260201-20260228", "metrics": {**_metrics(39, profit=-1.0237), "profit_total_abs": -10.2367, "profit_factor": 0.4764}},
    ]
    champion = {"strategy_class": "historical_best", "train_metrics": {"profit_total_pct": 0.03, "profit_factor": 1.0585, "max_drawdown_pct": 0.46}}

    detail = optimizer._build_not_best_reason_detail(train, validations, -41.9608, champion, {"min_profit_factor": 1.0, "max_drawdown_pct": 3.0}, "final_score<=0")

    joined = "\n".join(detail)
    assert "训练区间收益低于 historical_best" in joined
    assert "训练区间 PF 低于 historical_best" in joined
    assert "验证区间表现不稳定" in joined
    assert "最差验证 PF 仅 0.3929" in joined
    assert "固定止损亏损吞噬 ROI 收益" in joined
    assert "final_score=-41.9608" in joined


def test_common_failure_patterns_are_data_driven_not_fixed() -> None:
    rows = [
        {
            "total_trades": 24,
            "profit_factor": 0.9,
            "max_drawdown_pct": 1.0,
            "roi_profit_abs": 9.0,
            "stop_loss_profit_abs": -18.0,
            "validation_metrics": [
                {"period": "202604", "metrics": {"profit_total_abs": 4.0}},
                {"period": "202603", "metrics": {"profit_total_abs": -15.0}},
            ],
        },
        {
            "total_trades": 60,
            "profit_factor": 0.58,
            "max_drawdown_pct": 1.4,
            "roi_profit_abs": 9.1,
            "stop_loss_profit_abs": -18.2,
            "validation_metrics": [
                {"period": "202604", "metrics": {"profit_total_abs": 4.0}},
                {"period": "202603", "metrics": {"profit_total_abs": -15.0}},
            ],
        },
    ]

    patterns = optimizer._build_common_failure_patterns(rows, {"min_trades": 25, "max_trades": 80, "min_profit_factor": 1.0, "max_drawdown_pct": 3.0})

    assert "交易数过高" not in patterns
    assert "验证区间全部亏损" not in patterns
    assert "验证区间表现不稳定" in patterns
    assert "固定止损吞噬 ROI" in patterns
    assert "PF 低" in patterns


def test_random_sample_plan_respects_bounds_overlap_and_holdout(tmp_path) -> None:
    args = optimizer.argparse.Namespace(
        random_sample_windows=3,
        random_sample_min_days=25,
        random_sample_max_days=35,
        random_sample_data_start="20260101",
        random_sample_data_end="20260430",
        random_sample_seed="unit-test",
    )
    runtime_goal = {"holdout_ranges": [{"label": "holdout", "timerange": "20260201-20260301"}]}

    plan = optimizer.build_random_sample_plan(args, runtime_goal, tmp_path)

    assert (tmp_path / "random_sample_plan.json").exists()
    assert plan["enabled"] is True
    assert len(plan["windows"]) == 3
    for window in plan["windows"]:
        assert 25 <= window["days"] <= 35
        assert window["timerange"] != "20260201-20260301"
    for idx, left in enumerate(plan["windows"]):
        for right in plan["windows"][idx + 1 :]:
            assert optimizer._timerange_overlap_days(left["timerange"], right["timerange"]) <= 7


def test_strip_random_samples_for_ai_prompt_removes_observation_only_fields() -> None:
    payload = {
        "final_score": 12.3,
        "random_sample_metrics": [{"label": "random_001"}],
        "nested": {"random_sample_summary": {"enabled": True}, "keep": "value"},
    }

    stripped = optimizer._strip_random_samples_for_ai_prompt(payload)

    assert stripped == {"final_score": 12.3, "nested": {"keep": "value"}}


def test_strategy_fingerprint_includes_exit_timeframe_pair_and_protection_fields() -> None:
    code = '''
class Demo(IStrategy):
    timeframe = "15m"
    minimal_roi = {"0": 0.02, "60": 0.01}
    stoploss = -0.03
    trailing_stop = False
    use_exit_signal = False

    @property
    def protections(self):
        return [{"method": "CooldownPeriod", "stop_duration_candles": 4}]

    def populate_entry_trend(self, df, metadata):
        pair = metadata.get("pair")
        cond = (df["rsi"] < 30) & (df["volume"] > 0) & (pair == "BTC/USDT")
        df.loc[cond, ["enter_long", "enter_tag"]] = (1, "dip")
        return df
'''

    features = optimizer.extract_strategy_features(code)
    fingerprint = optimizer.build_strategy_fingerprint(features)

    assert fingerprint["hash"]
    payload = fingerprint["payload"]
    assert payload["minimal_roi"] == '{"0": 0.02, "60": 0.01}'
    assert payload["stoploss"] == "-0.03"
    assert payload["trailing_stop"] == "False"
    assert payload["use_exit_signal"] == "False"
    assert payload["timeframe"] == "15m"
    assert "cooldownperiod" in payload["cooldown_tokens"]
    assert payload["pair_filters"]


def test_minimal_round_summary_writes_fingerprint_and_duplicate_report() -> None:
    state = optimizer._new_round_defaults()
    state["strategy_fingerprint"] = {"hash": "abc123", "payload": {"timeframe": "15m"}}
    state["duplicate_report"] = {"is_duplicate": True, "decision": "skip_backtest"}
    state["invalid_reason"] = "策略与本次 run 已测试策略高度重复"

    summary = _summary_for_state(state)

    assert summary["strategy_fingerprint"]["hash"] == "abc123"
    assert summary["duplicate_report"]["is_duplicate"] is True
    assert summary["invalid_reason"] == "策略与本次 run 已测试策略高度重复"


def test_provider_pool_from_env_builds_openai_compatible_clients(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_ADVISOR_PROVIDER_POOL", "apihost_claude_opus47,deepseek_official")
    monkeypatch.setenv("AI_PROVIDER_APIHOST_CLAUDE_OPUS47_BASE_URL", "https://apihost.cn/v1")
    monkeypatch.setenv("AI_PROVIDER_APIHOST_CLAUDE_OPUS47_API_KEY", "sk-test")
    monkeypatch.setenv("AI_PROVIDER_APIHOST_CLAUDE_OPUS47_MODEL", "claude-opus-4-7")
    monkeypatch.setenv("AI_PROVIDER_APIHOST_CLAUDE_OPUS47_TYPE", "openai_compatible")
    monkeypatch.setenv("AI_PROVIDER_DEEPSEEK_OFFICIAL_BASE_URL", "你的deepseek_base_url")
    monkeypatch.setenv("AI_PROVIDER_DEEPSEEK_OFFICIAL_API_KEY", "你的deepseek_key")
    monkeypatch.setenv("AI_PROVIDER_DEEPSEEK_OFFICIAL_MODEL", "你的deepseek模型名")
    monkeypatch.setenv("AI_PROVIDER_DEEPSEEK_OFFICIAL_TYPE", "openai_compatible")

    runtime = optimizer._build_ai_role_runtime(
        {"provider_pool_env": "STRATEGY_ADVISOR_PROVIDER_POOL"},
        "strategy_advisor",
        timeout_sec=5,
        max_attempts_per_call=3,
        switch_on_error=True,
    )

    assert runtime.model_pool == ["claude-opus-4-7"]
    assert runtime.provider_pool[0]["name"] == "apihost_claude_opus47"
    assert runtime.provider_pool[0]["base_url"] == "https://apihost.cn/v1"


def test_provider_pool_env_errors_when_only_placeholders(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_CODEGEN_PROVIDER_POOL", "glm_official")
    monkeypatch.setenv("AI_PROVIDER_GLM_OFFICIAL_BASE_URL", "你的glm_base_url")
    monkeypatch.setenv("AI_PROVIDER_GLM_OFFICIAL_API_KEY", "你的glm_key")
    monkeypatch.setenv("AI_PROVIDER_GLM_OFFICIAL_MODEL", "你的glm模型名")
    monkeypatch.setenv("AI_PROVIDER_GLM_OFFICIAL_TYPE", "openai_compatible")

    try:
        optimizer._build_ai_role_runtime(
            {"provider_pool_env": "STRATEGY_CODEGEN_PROVIDER_POOL"},
            "code_generator",
            timeout_sec=5,
            max_attempts_per_call=3,
            switch_on_error=True,
        )
    except RuntimeError as exc:
        assert "没有可用 provider" in str(exc)
    else:
        raise AssertionError("placeholder provider pool should fail fast")


def test_pair_score_penalizes_negative_validation_stoploss_and_low_trades() -> None:
    cfg = {
        "min_pair_trades": 8,
        "max_pair_drawdown_pct": 3,
        "min_pair_profit_factor": 0.9,
        "prefer_validation_profit_positive": True,
    }
    row = {
        "pair": "BAD/USDT",
        "train_total_trades": 3,
        "train_profit_pct": -0.5,
        "train_profit_factor": 0.5,
        "train_max_drawdown_pct": 4.0,
        "validation_avg_profit_pct": -1.2,
        "validation_avg_profit_factor": 0.4,
        "validation_worst_profit_pct": -3.0,
        "validation_worst_profit_factor": 0.2,
        "validation_max_drawdown_pct": 5.0,
        "stoploss_to_roi_ratio": 2.0,
    }

    score, penalties, reason = optimizer._score_pair(row, cfg)

    assert score < 20
    assert "验证平均收益为负" in penalties
    assert "最差验证区间严重亏损" in penalties
    assert "固定止损亏损大于 ROI 收益" in penalties
    assert "交易数太少" in penalties
    assert "回撤过高" in penalties
    assert reason


def test_pair_metric_row_combines_train_validation_and_exit_reason_profit() -> None:
    train_metrics = {
        "pairs": [
            {"key": "SOL/USDT", "trades": 10, "profit_total_abs": 5, "profit_total": 0.01, "profit_factor": 1.4, "max_drawdown_pct": 1.2},
        ]
    }
    validations = [
        {"period": "v1", "timerange": "20240101-20240131", "metrics": {"pairs": [{"key": "SOL/USDT", "trades": 6, "profit_total": 0.02, "profit_factor": 1.2, "max_drawdown_pct": 0.8}]}},
        {"period": "v2", "timerange": "20240201-20240229", "metrics": {"pairs": [{"key": "SOL/USDT", "trades": 4, "profit_total": -0.01, "profit_factor": 0.8, "max_drawdown_pct": 1.5}]}},
    ]
    result = {
        "trades": [
            {"pair": "SOL/USDT", "exit_reason": "roi", "profit_abs": 3.0},
            {"pair": "SOL/USDT", "exit_reason": "stop_loss", "profit_abs": -1.5},
            {"pair": "ETH/USDT", "exit_reason": "roi", "profit_abs": 99.0},
        ]
    }

    row = optimizer._pair_metric_row("SOL/USDT", train_metrics, validations, result)

    assert row["train_total_trades"] == 10
    assert row["train_profit_pct"] == 1.0
    assert row["validation_avg_profit_pct"] == 0.5
    assert row["validation_worst_profit_pct"] == -1.0
    assert row["train_roi_profit_abs"] == 3.0
    assert row["train_stoploss_profit_abs"] == -1.5
    assert row["stoploss_to_roi_ratio"] == 0.5
