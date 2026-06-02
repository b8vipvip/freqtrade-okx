# -*- coding: utf-8 -*-
"""Multi-role AI investment committee orchestration.

The chairman persona is risk-first/value-investing-inspired, but never claims to be
or represent Warren Buffett.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from provider_config import auto_provider_pools_enabled, provider_pool_names_for_env

DEFAULT_AI_COMMITTEE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "roles": [
        "macro_analyst",
        "geopolitical_risk_analyst",
        "crypto_market_analyst",
        "quant_risk_manager",
        "strategy_engineer",
        "adversarial_reviewer",
        "final_chairman",
    ],
    "final_chairman_provider": "apihost_claude_opus47",
    "max_rounds": 2,
    "require_structured_json": True,
}

ROLE_DESCRIPTIONS = {
    "macro_analyst": "关注利率、美元、股市风险偏好、流动性。",
    "geopolitical_risk_analyst": "关注战争、制裁、贸易战、政策不确定性。",
    "crypto_market_analyst": "关注 BTC/ETH 趋势、ETF、资金费率、清算、稳定币流动性。",
    "quant_risk_manager": "关注 stoploss_to_roi_ratio、worst_month、PF、交易数、回撤。",
    "strategy_engineer": "把观点转成 Freqtrade 可实现的指标和条件。",
    "adversarial_reviewer": "指出过拟合、未来函数、不可实现规则、重复 prompt、无效 mutation。",
    "final_chairman": "价值投资、风险优先、先控制亏损、只在高胜率场景出手的主席；不得冒充巴菲特。",
}


def merge_config(goal: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_AI_COMMITTEE_CONFIG)
    raw = (goal or {}).get("ai_committee", {}) if isinstance(goal, dict) else {}
    if isinstance(raw, dict):
        cfg.update(raw)
    env_enabled = os.getenv("AI_COMMITTEE_ENABLED")
    if env_enabled is not None:
        cfg["enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
    if os.getenv("AI_COMMITTEE_ROLES"):
        cfg["roles"] = [r.strip() for r in os.getenv("AI_COMMITTEE_ROLES", "").split(",") if r.strip()]
    auto_chairman_pool = provider_pool_names_for_env("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL") if auto_provider_pools_enabled() else []
    if auto_chairman_pool:
        cfg["final_chairman_provider_pool"] = auto_chairman_pool
        cfg["final_chairman_provider"] = auto_chairman_pool[0]
    elif os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL"):
        pool = [p.strip() for p in os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL", "").split(",") if p.strip()]
        cfg["final_chairman_provider_pool"] = pool
        if pool:
            cfg["final_chairman_provider"] = pool[0]
    elif os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER"):
        cfg["final_chairman_provider"] = os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER")
    if os.getenv("AI_COMMITTEE_MAX_ROUNDS"):
        cfg["max_rounds"] = int(os.getenv("AI_COMMITTEE_MAX_ROUNDS", "2") or 2)
    return cfg


def _failure_focus(failure_type: str) -> str:
    if failure_type in {"worst_month_loss", "stoploss_to_roi_high", "low_trade_profitable_near_miss"}:
        return failure_type
    if failure_type == "low_trade_profitable":
        return "low_trade_profitable_near_miss"
    return "stoploss_to_roi_high"


def _role_output(role: str, ctx: dict[str, Any]) -> dict[str, Any]:
    failure_type = str(ctx.get("failure_type") or "general_improvement")
    macro = ((ctx.get("market_intel") or {}).get("macro_regime") or {}).get("risk_on_off", "neutral")
    crypto = ((ctx.get("market_intel") or {}).get("crypto_regime") or {}).get("trend", "range")
    base = {
        "role": role,
        "main_view": f"{ROLE_DESCRIPTIONS.get(role, role)} 当前 failure_type={failure_type}, macro={macro}, crypto={crypto}。",
        "risks": [],
        "recommended_strategy_constraints": [],
        "blocked_ideas": [],
        "confidence": 0.62,
    }
    if role == "macro_analyst":
        base["risks"] = ["美元/利率/股指风险偏好切换可能放大加密回撤"]
        base["recommended_strategy_constraints"] = ["加入可回测的 regime 与波动过滤", "risk_off 时降低 BTC/OP 入场频率"]
    elif role == "geopolitical_risk_analyst":
        base["risks"] = ["战争、制裁、监管标题党不可直接编码为交易规则"]
        base["blocked_ideas"] = ["把新闻关键词写入策略代码", "使用验证月新闻反向拟合规则"]
    elif role == "crypto_market_analyst":
        base["risks"] = ["资金费率/清算/ETF 流出可能导致假突破"]
        base["recommended_strategy_constraints"] = ["避免 choppy market", "BTC/OP 使用更严格 entry threshold"]
    elif role == "quant_risk_manager":
        base["risks"] = ["stoploss_to_roi_ratio 过高", "worst_month_loss", "交易数失控或过低"]
        base["recommended_strategy_constraints"] = ["控制固定止损暴露", "目标交易数 20~60，理想 25~45"]
        base["blocked_ideas"] = ["仅调整 ROI", "用 trailing 替代 fixed stoploss"]
    elif role == "strategy_engineer":
        base["recommended_strategy_constraints"] = ["populate_indicators 计算 regime/choppy/volatility filters", "populate_entry_trend 落地 BTC/OP pair-specific threshold"]
    elif role == "adversarial_reviewer":
        base["risks"] = ["prompt 重复", "不可实现新闻规则", "未来函数/验证集污染"]
        base["blocked_ideas"] = ["不可回测的新闻条件", "删除风险过滤器", "等价 mutation"]
    return base


def _build_directive(ctx: dict[str, Any], role_outputs: list[dict[str, Any]], retry_hint: str = "") -> dict[str, Any]:
    active_pairs = [str(p) for p in ctx.get("active_pairs", [])]
    pair_rules: dict[str, Any] = {}
    for pair in active_pairs:
        if pair == "BTC/USDT":
            pair_rules[pair] = {"entry_threshold": "stricter", "reason": "risk-first committee prioritizes lower BTC stoploss exposure"}
        elif pair == "OP/USDT":
            pair_rules[pair] = {"entry_threshold": "much_stricter", "max_trade_share": "reduced", "reason": "OP high-frequency stoploss exposure must be capped"}
    failure_type = str(ctx.get("failure_type") or "")
    too_few_recovery = failure_type in {"too_few_trades", "training_trade_count_below_min", "zero_trade"}
    allowed_mutations = ["pair_specific_filter", "add_entry_filter", "tighten_entry_trigger", "reduce_trade_frequency", "cooldown_or_protection"]
    blocked_mutations = ["adjust_roi_only", "replace_stoploss_with_trailing", "remove_risk_filter", "increase_trade_frequency", "news_keyword_rule"]
    if too_few_recovery:
        allowed_mutations = ["controlled_widen_entry", "widen_entry_controlled", "remove_overstrict_pair_filter", "pair_specific_filter"]
        blocked_mutations = ["add_more_global_filters", "tighten_entry_trigger", "add_entry_filter", "reduce_trade_frequency", "adjust_roi_only", "replace_stoploss_with_trailing", "news_keyword_rule"]
    return {
        "preferred_parent": "nearest_candidate",
        "failure_focus": _failure_focus(failure_type),
        "strategy_family": "trend_following_regime_filter" if not retry_hint else "trend_following_regime_filter_diversified",
        "allowed_mutations": allowed_mutations,
        "blocked_mutations": blocked_mutations,
        "too_few_trades_recovery_mode": too_few_recovery,
        "pair_specific_rules": pair_rules,
        "trade_count_target": {"min": 20, "ideal_min": 25, "ideal_max": 45, "max": 60},
        "risk_constraints": ["reduce stoploss_to_roi_ratio", "do not increase OP high frequency trades", "do not remove risk filters", "validation intel may only become abstract postmortem metrics"],
        "codegen_requirements": [
            "populate_entry_trend must include pair-specific BTC/OP thresholds when those pairs are active",
            "populate_indicators may calculate regime/choppy/volatility filters only when mutation_spec includes estimated_trade_count_guard",
            "risk-reducing does not mean no-trade; any mutation must satisfy min_trades >= 20 and target total trades 20~60 (ideal 25~45)",
            "BTC/OP may be tightened, but SOL/BNB/DOGE opportunities must be preserved",
            "must not be equivalent to previous strategy",
            "do not implement news-based rules or external API calls",
        ] + ([f"diversification_retry_hint: {retry_hint}"] if retry_hint else []),
        "prompt_directive": (
            "Use a value-investing-inspired, risk-first committee style: first avoid large losses, do not trade merely for count, "
            "control BTC/OP stoploss exposure without pushing global training trades below 20, never use validation/holdout news as generation input, and only express market intel as backtestable candle/volume/volatility proxies."
        ),
        "final_chairman_decision": "Proceed only if the mutation is materially different, risk-reducing, implementable in Freqtrade without news/live external data, and preserves min_trades >= 20; risk-reducing does not mean no-trade.",
        "do_not_impersonate": "This is not Warren Buffett advice and must not claim to be from Buffett.",
    }


def run_committee(
    *,
    run_dir: Path,
    market_intel: dict[str, Any] | None,
    pair_attribution: dict[str, Any] | None,
    last_run_summary: dict[str, Any] | None,
    nearest_candidate: dict[str, Any] | None,
    historical_best: dict[str, Any] | None,
    current_goal: dict[str, Any],
    active_pairs: list[str],
    failure_type: str,
    config: dict[str, Any] | None = None,
    retry_hint: str = "",
) -> dict[str, Any]:
    cfg = dict(DEFAULT_AI_COMMITTEE_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    out_dir = run_dir / "committee"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = {
        "market_intel": market_intel or {},
        "pair_attribution": pair_attribution or {},
        "last_run_summary": last_run_summary or {},
        "nearest_candidate": nearest_candidate or {},
        "historical_best": historical_best or {},
        "current_goal": current_goal,
        "active_pairs": active_pairs,
        "failure_type": failure_type,
    }
    roles = [str(r) for r in cfg.get("roles", []) if str(r)] or list(DEFAULT_AI_COMMITTEE_CONFIG["roles"])
    non_final = [r for r in roles if r != "final_chairman"]
    role_outputs = [_role_output(role, ctx) for role in non_final]
    directive = _build_directive(ctx, role_outputs, retry_hint=retry_hint)
    role_outputs.append({
        "role": "final_chairman",
        "main_view": directive["final_chairman_decision"],
        "risks": directive["risk_constraints"],
        "recommended_strategy_constraints": directive["allowed_mutations"],
        "blocked_ideas": directive["blocked_mutations"],
        "confidence": 0.72,
    })
    transcript = ["# AI 投资委员会会议纪要", "", f"generated_at: {datetime.now(timezone.utc).isoformat()}", "", "> 风格说明：价值投资/风险优先/先控制亏损；不冒充巴菲特本人。", ""]
    for item in role_outputs:
        transcript.extend([f"## {item['role']}", item.get("main_view", ""), "", "- risks: " + json.dumps(item.get("risks", []), ensure_ascii=False), "- constraints: " + json.dumps(item.get("recommended_strategy_constraints", []), ensure_ascii=False), "- blocked: " + json.dumps(item.get("blocked_ideas", []), ensure_ascii=False), ""])
    (out_dir / "role_outputs.json").write_text(json.dumps(role_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "committee_transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    (out_dir / "final_strategy_directive.json").write_text(json.dumps(directive, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"role_outputs": role_outputs, "final_strategy_directive": directive, "committee_consensus": transcript}


def directive_prompt_section(market_intel: dict[str, Any] | None, directive: dict[str, Any] | None) -> str:
    if not directive:
        return ""
    safe = {
        "market_regime": (market_intel or {}).get("macro_regime", {}),
        "crypto_regime": (market_intel or {}).get("crypto_regime", {}),
        "final_strategy_directive": directive,
        "data_isolation": {
            "train_intel_used_for_prompt": True,
            "validation_intel_used_for_prompt": False,
            "holdout_intel_used_for_prompt": False,
        },
    }
    return "\n\n========== AI 投资委员会 strategy_directive（仅训练区间情报，可注入 prompt）==========\n" + json.dumps(safe, ensure_ascii=False, indent=2) + "\n"
