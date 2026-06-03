from pathlib import Path
import sys

AI_TOOLS_PATH = Path(__file__).resolve().parents[1] / "ai_tools"
if str(AI_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(AI_TOOLS_PATH))

from prompt_similarity import merge_config, review_prompts


def test_prompt_similarity_uses_mutation_intent_and_retries_once(tmp_path: Path) -> None:
    cfg = merge_config({"prompt_similarity_filter": {"enabled": True, "threshold": 0.95}})
    assert cfg["max_retries"] == 1
    assert cfg["compare_fields"] == ["normalized_mutation_intent", "codegen_prompt_fingerprint"]

    registry = tmp_path / "registry.json"
    first = {
        "strategy_family": "trend_following",
        "mutation_type": "add_regime_filter",
        "indicators_to_add": ["adx"],
        "entry_conditions_to_change": ["only enter when adx rising"],
        "exit_conditions_to_change": [],
        "pair_specific_rules": {"BTC/USDT": "stricter"},
        "estimated_trade_count_guard": {"min_trades": 20},
        "implementation_intent_summary": "add choppy filter",
    }
    review_prompts(
        run_dir=tmp_path,
        registry_path=registry,
        fields={"mutation_spec": first, "advisor_prompt": "shared boilerplate A", "codegen_prompt": "parent code A"},
        config=cfg,
        run_id="run_a",
        version="v001",
    )

    report = review_prompts(
        run_dir=tmp_path,
        registry_path=registry,
        fields={"mutation_spec": dict(first), "advisor_prompt": "different boilerplate B", "codegen_prompt": "different parent code B"},
        config=cfg,
        run_id="run_b",
        version="v001",
    )
    assert report["advisor_prompt_similarity"] == 1.0
    assert report["decision"] == "retry_advisor_only"
    assert report["similarity_basis"]["advisor"] == "normalized_mutation_intent_only"

    after_retry = review_prompts(
        run_dir=tmp_path,
        registry_path=registry,
        fields={"mutation_spec": dict(first)},
        config=cfg,
        run_id="run_b",
        version="v001",
        retry_index=1,
    )
    assert after_retry["decision"] == "force_continue_after_retry_limit"
    assert after_retry["max_retries"] == 1


def test_prompt_similarity_ignores_full_prompt_boilerplate(tmp_path: Path) -> None:
    cfg = {"threshold": 0.95, "compare_fields": ["normalized_mutation_intent", "codegen_prompt_fingerprint"], "max_retries": 1}
    registry = tmp_path / "registry.json"
    review_prompts(
        run_dir=tmp_path,
        registry_path=registry,
        fields={
            "mutation_spec": {"strategy_family": "mean_reversion", "mutation_type": "rsi_rebound", "indicators_to_add": ["rsi"]},
            "advisor_prompt": "SAME PUBLIC TEMPLATE" * 100,
            "codegen_prompt": "SAME PARENT CODE" * 100,
        },
        config=cfg,
        run_id="run_a",
        version="v001",
    )
    report = review_prompts(
        run_dir=tmp_path,
        registry_path=registry,
        fields={
            "mutation_spec": {"strategy_family": "breakout", "mutation_type": "volume_breakout", "indicators_to_add": ["volume_mean_20"]},
            "advisor_prompt": "SAME PUBLIC TEMPLATE" * 100,
            "codegen_prompt": "SAME PARENT CODE" * 100,
        },
        config=cfg,
        run_id="run_b",
        version="v001",
    )
    assert report["advisor_prompt_similarity"] < 0.95
    assert report["decision"] == "continue"
