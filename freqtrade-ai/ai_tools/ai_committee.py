# -*- coding: utf-8 -*-
"""Concurrent six-role AI investment committee orchestration."""

from __future__ import annotations

import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from provider_config import auto_provider_pools_enabled, provider_pool_names_for_env

DEFAULT_AI_COMMITTEE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "parallel_enabled": True,
    "role_count": 6,
    "max_parallel_calls": 6,
    "role_timeout_seconds": 180,
    "min_successful_roles": 3,
    "final_chairman_provider": "GS88_GPT55",
    "require_structured_json": True,
    "roles": [
        "pessimistic_risk_officer",
        "optimistic_trend_analyst",
        "aggressive_momentum_trader",
        "conservative_value_chair_candidate",
        "macro_liquidity_analyst",
        "quantitative_stat_arb_reviewer",
    ],
}

ROLE_SPECS: dict[str, dict[str, str]] = {
    "pessimistic_risk_officer": {
        "stance": "pessimistic",
        "title": "悲观风控官",
        "focus": "寻找爆仓风险、止损吞噬 ROI、最差验证月、过拟合、低质量入场；宁愿少交易，也不要扩大风险；但必须遵守 min_trades>=20，不允许 no-trade 策略。",
    },
    "optimistic_trend_analyst": {
        "stance": "optimistic",
        "title": "乐观趋势分析师",
        "focus": "寻找 BTC/SOL/BNB/DOGE 等趋势延续、波动突破、顺势机会；风险可控时保留交易机会；防止盲目放宽导致高频。",
    },
    "aggressive_momentum_trader": {
        "stance": "aggressive",
        "title": "激进动量交易员",
        "focus": "寻找短周期动量、突破确认、波动扩张机会；可以提出更激进 entry，但必须说明风控边界；不允许训练交易数超过 max_trades 或高频亏损。",
    },
    "conservative_value_chair_candidate": {
        "stance": "conservative",
        "title": "保守价值/风险优先分析师",
        "focus": "关注长期稳健、风险收益比、最差月表现、低回撤；少做低确定性交易，优先降低固定止损损失；不能把交易数压到 20 以下。",
    },
    "macro_liquidity_analyst": {
        "stance": "macro",
        "title": "宏观与流动性分析师",
        "focus": "仅使用 market_intel 的训练期情报，关注美元、利率、ETF、股市风险偏好、加密流动性；禁止验证/holdout 新闻进入 prompt；禁止新闻关键词交易规则，只能抽象为 market regime 约束。",
    },
    "quantitative_stat_arb_reviewer": {
        "stance": "quant",
        "title": "量化统计审查员",
        "focus": "分析交易数、PF、胜率、ROI、stoploss_to_roi_ratio、pair attribution；必须从历史 run、nearest_candidate、last_run_summary、pair attribution 提出量化建议；必须明确 estimated_trade_count_guard。",
    },
}

PAIR_VIEW_TEMPLATE = {pair: {} for pair in ["BTC/USDT", "OP/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]}
HARD_BLOCKED_MUTATIONS = ["replace_stoploss_with_trailing", "adjust_roi_only", "remove_risk_filter", "news_keyword_rule"]
RECOVERY_FAILURE_TYPES = {"too_few_trades", "training_trade_count_below_min", "zero_or_near_zero_trade", "zero_trade"}
RECOVERY_ALLOWED = ["controlled_widen_entry", "relax_global_filter", "pair_specific_restore_non_bad_pairs", "reduce_overstrict_regime_filter"]
RECOVERY_BLOCKED = ["add_more_global_filters", "tighten_all_pairs", "reduce_trade_frequency", "add_choppy_filter_without_trade_guard"]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def _csv(raw: str | None) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def merge_config(goal: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_AI_COMMITTEE_CONFIG)
    raw = (goal or {}).get("ai_committee", {}) if isinstance(goal, dict) else {}
    if isinstance(raw, dict):
        cfg.update(raw)
    if os.getenv("AI_COMMITTEE_ENABLED") is not None:
        cfg["enabled"] = _env_flag("AI_COMMITTEE_ENABLED", bool(cfg.get("enabled")))
    cfg["parallel_enabled"] = _env_flag("AI_COMMITTEE_PARALLEL_ENABLED", bool(cfg.get("parallel_enabled", True)))
    cfg["role_count"] = _env_int("AI_COMMITTEE_ROLE_COUNT", int(cfg.get("role_count", 6) or 6))
    cfg["max_parallel_calls"] = _env_int("AI_COMMITTEE_MAX_PARALLEL_CALLS", int(cfg.get("max_parallel_calls", 6) or 6))
    cfg["role_timeout_seconds"] = _env_int("AI_COMMITTEE_ROLE_TIMEOUT_SECONDS", int(cfg.get("role_timeout_seconds", 180) or 180))
    cfg["min_successful_roles"] = _env_int("AI_COMMITTEE_MIN_SUCCESSFUL_ROLES", int(cfg.get("min_successful_roles", 3) or 3))
    if os.getenv("AI_COMMITTEE_ROLES"):
        cfg["roles"] = _csv(os.getenv("AI_COMMITTEE_ROLES"))
    analyst_pool = provider_pool_names_for_env("AI_COMMITTEE_ANALYST_PROVIDER_POOL") if (os.getenv("AI_COMMITTEE_ANALYST_PROVIDER_POOL") or auto_provider_pools_enabled()) else []
    if analyst_pool:
        cfg["analyst_provider_pool"] = analyst_pool
    chairman_pool = provider_pool_names_for_env("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL") if (os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL") or auto_provider_pools_enabled()) else []
    if chairman_pool:
        cfg["final_chairman_provider_pool"] = chairman_pool
        cfg["final_chairman_provider"] = chairman_pool[0]
    elif os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER"):
        cfg["final_chairman_provider"] = os.getenv("AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER")
    return cfg


def _guard(reason: str = "保持训练交易数 20~60，理想 25~45。") -> dict[str, Any]:
    return {"min_trades": 20, "ideal_min_trades": 25, "ideal_max_trades": 45, "max_trades": 60, "will_reduce_below_min": False, "reason": reason}


def _normalize_role_output(data: dict[str, Any], role: str, provider: str = "offline", model: str = "offline") -> dict[str, Any]:
    spec = ROLE_SPECS.get(role, {})
    out = {
        "role": role,
        "stance": data.get("stance") or spec.get("stance") or "quant",
        "provider": data.get("provider") or provider,
        "model": data.get("model") or model,
        "summary": str(data.get("summary") or data.get("main_view") or spec.get("focus") or ""),
        "key_risks": list(data.get("key_risks") or data.get("risks") or []),
        "key_opportunities": list(data.get("key_opportunities") or []),
        "pair_specific_view": data.get("pair_specific_view") if isinstance(data.get("pair_specific_view"), dict) else dict(PAIR_VIEW_TEMPLATE),
        "allowed_mutations": list(data.get("allowed_mutations") or data.get("recommended_strategy_constraints") or []),
        "blocked_mutations": sorted(set(list(data.get("blocked_mutations") or data.get("blocked_ideas") or []) + HARD_BLOCKED_MUTATIONS)),
        "estimated_trade_count_guard": data.get("estimated_trade_count_guard") if isinstance(data.get("estimated_trade_count_guard"), dict) else _guard(),
        "risk_constraints": list(data.get("risk_constraints") or []),
        "confidence": float(data.get("confidence", 0.62) or 0.62),
    }
    for pair in PAIR_VIEW_TEMPLATE:
        out["pair_specific_view"].setdefault(pair, {})
    out["estimated_trade_count_guard"].setdefault("min_trades", 20)
    out["estimated_trade_count_guard"].setdefault("ideal_min_trades", 25)
    out["estimated_trade_count_guard"].setdefault("ideal_max_trades", 45)
    out["estimated_trade_count_guard"].setdefault("max_trades", 60)
    out["estimated_trade_count_guard"].setdefault("will_reduce_below_min", False)
    out["estimated_trade_count_guard"].setdefault("reason", "未说明，默认保持交易数护栏。")
    return out


def _offline_role_output(role: str, ctx: dict[str, Any]) -> dict[str, Any]:
    spec = ROLE_SPECS.get(role, {})
    failure_type = str(ctx.get("failure_type") or "general_improvement")
    allowed = ["pair_specific_filter", "risk_guarded_entry_refinement", "regime_filter_with_trade_guard"]
    blocked = ["adjust_roi_only", "replace_stoploss_with_trailing", "remove_risk_filter", "news_keyword_rule"]
    if (_env_flag("TOO_FEW_TRADES_RECOVERY_ENABLED", True) and failure_type in RECOVERY_FAILURE_TYPES):
        allowed = RECOVERY_ALLOWED + [x for x in allowed if x not in RECOVERY_ALLOWED]
        blocked = sorted(set(blocked + RECOVERY_BLOCKED))
    return _normalize_role_output({
        "summary": f"{spec.get('title', role)}：{spec.get('focus', '')} 当前 failure_type={failure_type}，建议所有 mutation 都带交易数护栏。",
        "key_risks": ["stoploss_to_roi_ratio 过高", "全局过滤过严导致 too_few_trades", "验证/holdout 信息污染"],
        "key_opportunities": ["SOL/BNB/DOGE 可保留趋势/动量机会", "BTC/OP 可采用 pair-specific 更严格门槛"],
        "allowed_mutations": allowed,
        "blocked_mutations": blocked,
        "risk_constraints": ["min_trades>=20", "ideal 25~45", "max_trades<=60", "不得生成新闻关键词规则"],
        "confidence": 0.64,
    }, role=role)


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads((text or "").strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if m:
        data = json.loads(m.group(1))
        if isinstance(data, dict):
            return data
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text or ""):
        if ch == "{":
            try:
                data, _ = decoder.raw_decode(text[idx:])
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    raise ValueError("role output is not a JSON object")


def _clone_runtime(runtime: Any, offset: int, timeout_seconds: int) -> Any:
    cloned = copy.copy(runtime)
    cloned.attempts = []
    cloned.used_model = ""
    cloned.used_provider = ""
    cloned.forced_provider_offset = offset
    cloned.timeout_sec = timeout_seconds
    return cloned


def _role_prompt(role: str, ctx: dict[str, Any]) -> str:
    spec = ROLE_SPECS.get(role, {})
    safe_ctx = {
        "market_intel_train_only": ctx.get("market_intel", {}),
        "pair_attribution": ctx.get("pair_attribution", {}),
        "last_run_summary": ctx.get("last_run_summary", {}),
        "nearest_candidate": ctx.get("nearest_candidate", {}),
        "historical_best": ctx.get("historical_best", {}),
        "active_pairs": ctx.get("active_pairs", []),
        "failure_type": ctx.get("failure_type", ""),
        "too_few_trades_recovery": (_env_flag("TOO_FEW_TRADES_RECOVERY_ENABLED", True) and ctx.get("failure_type") in RECOVERY_FAILURE_TYPES),
    }
    return (
        f"你是专业、资深、理性、可审计的量化投资委员会分析师：{role}（{spec.get('title', '')}）。\n"
        f"立场/重点：{spec.get('focus', '')}\n"
        "只服务于最终量化策略目标，不允许玄学、情绪化、故事化。不得使用 validation/holdout 新闻或情报。\n"
        "硬约束：min_trades>=20，ideal 25~45，max_trades<=60；不允许 no-trade 风控；不允许 replace_stoploss_with_trailing；不允许 adjust_roi_only；不允许 remove_risk_filter；不允许 news_keyword_rule。\n"
        "如果上一轮 too_few_trades，必须允许 controlled_widen_entry/relax_global_filter/pair_specific_restore_non_bad_pairs/reduce_overstrict_regime_filter，禁止继续收紧所有 pair。\n"
        "必须只输出 JSON，字段严格包括 role, stance, provider, model, summary, key_risks, key_opportunities, pair_specific_view, allowed_mutations, blocked_mutations, estimated_trade_count_guard, risk_constraints, confidence。\n"
        "上下文 JSON（仅训练区间情报）：\n" + json.dumps(safe_ctx, ensure_ascii=False, indent=2)[:50000]
    )


def _run_live_role(role: str, ctx: dict[str, Any], runtime: Any, ask_ai: Callable[..., str], state: dict[str, Any], offset: int, timeout_seconds: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    role_runtime = _clone_runtime(runtime, offset, timeout_seconds)
    started = time.time()
    try:
        raw = ask_ai(role_runtime, [{"role": "user", "content": _role_prompt(role, ctx)}], state=state)
        parsed = _json_from_text(raw)
        provider = getattr(role_runtime, "used_provider", "") or ""
        model = getattr(role_runtime, "used_model", "") or ""
        output = _normalize_role_output(parsed, role=role, provider=provider, model=model)
        output["elapsed_seconds"] = round(time.time() - started, 3)
        print(f"committee role {role}: provider={provider}, model={model}, elapsed={output['elapsed_seconds']}s")
        return output, None
    except Exception as exc:  # noqa: BLE001 - failed roles are audited, not fatal
        failure = {"role": role, "error": str(exc), "elapsed_seconds": round(time.time() - started, 3), "attempts": getattr(role_runtime, "attempts", [])}
        print(f"committee role {role} failed: {failure['error']}")
        return None, failure


def _build_directive(ctx: dict[str, Any], role_outputs: list[dict[str, Any]], failed_roles: list[dict[str, Any]], chairman_meta: dict[str, Any] | None = None, retry_hint: str = "") -> dict[str, Any]:
    failure_type = str(ctx.get("failure_type") or "general_improvement")
    too_few = (_env_flag("TOO_FEW_TRADES_RECOVERY_ENABLED", True) and failure_type in RECOVERY_FAILURE_TYPES)
    allowed: list[str] = []
    blocked: list[str] = []
    for item in role_outputs:
        allowed.extend([str(x) for x in item.get("allowed_mutations", [])])
        blocked.extend([str(x) for x in item.get("blocked_mutations", [])])
    if not allowed:
        allowed = ["pair_specific_filter", "risk_guarded_entry_refinement", "regime_filter_with_trade_guard"]
    if too_few:
        allowed = RECOVERY_ALLOWED + [x for x in allowed if x not in RECOVERY_ALLOWED]
        blocked = sorted(set(blocked + RECOVERY_BLOCKED))
    blocked = sorted(set(blocked + HARD_BLOCKED_MUTATIONS))
    allowed = [x for x in dict.fromkeys(allowed) if x not in blocked or x in RECOVERY_ALLOWED]
    adopted = []
    rejected = []
    for item in role_outputs:
        adopted.append({"role": item.get("role"), "adopted": item.get("allowed_mutations", [])[:5], "reason": "符合可回测、交易数护栏和风险控制硬约束。"})
        for mut in item.get("blocked_mutations", [])[:5]:
            rejected.append({"role": item.get("role"), "rejected": mut, "reason": "违反或接近硬约束/过拟合/不可回测风险，主席拒绝。"})
    directive = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chairman_provider": (chairman_meta or {}).get("provider", "offline"),
        "chairman_model": (chairman_meta or {}).get("model", "offline"),
        "chairman_elapsed_seconds": (chairman_meta or {}).get("elapsed_seconds", 0),
        "successful_role_count": len(role_outputs),
        "failed_role_count": len(failed_roles),
        "failed_roles": failed_roles,
        "adopted_views": adopted,
        "rejected_views": rejected,
        "preferred_parent": "nearest_candidate",
        "failure_focus": failure_type,
        "strategy_family": "trend_following_regime_filter",
        "allowed_mutations": allowed,
        "blocked_mutations": blocked,
        "too_few_trades_recovery_mode": too_few,
        "trade_count_target": {"min": 20, "ideal_min": 25, "ideal_max": 45, "max": 60},
        "estimated_trade_count_guard": _guard("主席硬约束：不允许把交易数压到 20 以下，理想 25~45，最高 60。"),
        "pair_specific_rules": {
            "BTC/USDT": {"bias": "stricter quality/risk gates if prior attribution is weak"},
            "OP/USDT": {"bias": "avoid low-quality entries and high stoploss exposure"},
            "SOL/USDT": {"bias": "preserve reasonable trend/momentum opportunities"},
            "BNB/USDT": {"bias": "preserve reasonable trend/momentum opportunities"},
            "DOGE/USDT": {"bias": "preserve reasonable volatility breakout opportunities without high frequency"},
        },
        "risk_constraints": ["min_trades>=20", "ideal 25~45", "max_trades<=60", "no no-trade risk control", "no replace_stoploss_with_trailing", "no adjust_roi_only", "no remove_risk_filter", "no news_keyword_rule", "no validation/holdout intel in prompt"],
        "codegen_requirements": [
            "populate_entry_trend must implement pair-specific BTC/OP thresholds when required",
            "regime/choppy/volatility filters must include estimated_trade_count_guard and pair-specific bypass/less strict path",
            "must preserve SOL/BNB/DOGE opportunities and at least 3 pairs can potentially trigger entry",
            "risk-reducing does not mean no-trade; satisfy min_trades >= 20 and target 20~60 (ideal 25~45)",
            "do not implement news-based rules or external API calls",
        ],
        "prompt_directive": "主席结论：明确处理分歧，采纳可回测且符合交易数护栏的 pair-specific 风控/趋势机会建议，拒绝 no-trade、新闻关键词、纯 ROI、删除风控、过严全局过滤。",
        "final_chairman_decision": "Proceed only if mutation is materially different, risk-reducing, implementable in Freqtrade, and preserves min_trades>=20; risk-reducing does not mean no-trade.",
        "may_reduce_below_min": False,
    }
    if retry_hint:
        directive["retry_hint"] = retry_hint
    return directive


def _chairman_prompt(ctx: dict[str, Any], role_outputs: list[dict[str, Any]], failed_roles: list[dict[str, Any]]) -> str:
    return (
        "你是 AI 投资委员会主席。读取 6 个角色输出，生成 final_strategy_directive JSON。不能简单平均，必须明确 adopted_views/rejected_views 以及拒绝原因。\n"
        "硬约束：min_trades>=20；ideal 25~45；max_trades<=60；不允许 no-trade 风控；不允许 replace_stoploss_with_trailing；不允许 adjust_roi_only；不允许 remove_risk_filter；不允许 news_keyword_rule；不允许 validation/holdout intel 进入 prompt。\n"
        "只输出 JSON。\n"
        + json.dumps({"role_outputs": role_outputs, "failed_roles": failed_roles, "failure_type": ctx.get("failure_type")}, ensure_ascii=False, indent=2)[:60000]
    )


def _write_outputs(out_dir: Path, role_outputs: list[dict[str, Any]], failures: list[dict[str, Any]], directive: dict[str, Any]) -> list[str]:
    transcript = ["# AI 投资委员会会议纪要", "", f"generated_at: {datetime.now(timezone.utc).isoformat()}", "", f"successful_roles: {len(role_outputs)}", f"failed_roles: {len(failures)}", ""]
    for item in role_outputs:
        transcript += [f"## {item.get('role')} ({item.get('stance')})", f"provider/model: {item.get('provider')}/{item.get('model')} elapsed={item.get('elapsed_seconds', 0)}s", item.get("summary", ""), "", "allowed_mutations: " + json.dumps(item.get("allowed_mutations", []), ensure_ascii=False), "blocked_mutations: " + json.dumps(item.get("blocked_mutations", []), ensure_ascii=False), ""]
    transcript += ["## final_chairman", json.dumps(directive, ensure_ascii=False, indent=2)]
    (out_dir / "role_outputs.json").write_text(json.dumps(role_outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "committee_transcript.md").write_text("\n".join(transcript), encoding="utf-8")
    (out_dir / "final_strategy_directive.json").write_text(json.dumps(directive, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "committee_failures.json").write_text(json.dumps({"failed_roles": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript


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
    analyst_runtime: Any | None = None,
    chairman_runtime: Any | None = None,
    ask_ai_func: Callable[..., str] | None = None,
    ai_state: dict[str, Any] | None = None,
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
    roles = [r for r in (cfg.get("roles") or []) if r in ROLE_SPECS][: int(cfg.get("role_count", 6) or 6)] or list(ROLE_SPECS)[:6]
    parallel_enabled = bool(cfg.get("parallel_enabled", True))
    max_workers = max(1, int(cfg.get("max_parallel_calls", 6) or 6))
    timeout_seconds = max(1, int(cfg.get("role_timeout_seconds", 180) or 180))
    min_success = max(1, int(cfg.get("min_successful_roles", 3) or 3))
    provider_pool = list(cfg.get("analyst_provider_pool") or [])
    if provider_pool and len(provider_pool) < len(roles):
        print(f"AI 投资委员会 provider 数量不足，允许复用 provider：providers={len(provider_pool)}, roles={len(roles)}")
    print(f"AI 投资委员会并发模式：{'开启' if parallel_enabled else '关闭'}")
    print(f"AI 投资委员会角色数量：{len(roles)}")
    role_outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if analyst_runtime is not None and ask_ai_func is not None:
        if parallel_enabled:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = {executor.submit(_run_live_role, role, ctx, analyst_runtime, ask_ai_func, ai_state or {}, idx, timeout_seconds): role for idx, role in enumerate(roles)}
                for fut in as_completed(futs):
                    output, failure = fut.result()
                    if output:
                        role_outputs.append(output)
                    if failure:
                        failures.append(failure)
        else:
            for idx, role in enumerate(roles):
                output, failure = _run_live_role(role, ctx, analyst_runtime, ask_ai_func, ai_state or {}, idx, timeout_seconds)
                if output:
                    role_outputs.append(output)
                if failure:
                    failures.append(failure)
    else:
        role_outputs = [_offline_role_output(role, ctx) for role in roles]
    role_outputs.sort(key=lambda x: roles.index(str(x.get("role"))) if str(x.get("role")) in roles else 999)
    print(f"AI 投资委员会成功角色数量：{len(role_outputs)}")
    print(f"AI 投资委员会失败角色数量：{len(failures)}")
    if len(role_outputs) < min_success:
        print(f"成功角色少于 {min_success}，fallback 到 offline conservative committee。")
        role_outputs = [_offline_role_output(role, ctx) for role in roles]
        failures.append({"role": "fallback", "error": "live successful roles below minimum; used offline conservative committee"})
    chairman_meta = {"provider": "offline", "model": "offline", "elapsed_seconds": 0}
    directive: dict[str, Any] | None = None
    if chairman_runtime is not None and ask_ai_func is not None and len(role_outputs) >= min_success:
        chairman = _clone_runtime(chairman_runtime, 0, timeout_seconds)
        started = time.time()
        try:
            raw = ask_ai_func(chairman, [{"role": "user", "content": _chairman_prompt(ctx, role_outputs, failures)}], state=ai_state or {})
            parsed = _json_from_text(raw)
            chairman_meta = {"provider": getattr(chairman, "used_provider", "") or "", "model": getattr(chairman, "used_model", "") or "", "elapsed_seconds": round(time.time() - started, 3)}
            directive = _build_directive(ctx, role_outputs, failures, chairman_meta, retry_hint=retry_hint)
            # Preserve local hard constraints while letting chairman add audit fields.
            for key in ["adopted_views", "rejected_views", "final_chairman_decision", "prompt_directive"]:
                if key in parsed:
                    directive[key] = parsed[key]
        except Exception as exc:  # noqa: BLE001
            failures.append({"role": "final_chairman", "error": str(exc), "attempts": getattr(chairman, "attempts", [])})
    if directive is None:
        directive = _build_directive(ctx, role_outputs, failures, chairman_meta, retry_hint=retry_hint)
    print(f"chairman provider/model/耗时：{directive.get('chairman_provider')}/{directive.get('chairman_model')}/{directive.get('chairman_elapsed_seconds')}s")
    print("final allowed_mutations:" + json.dumps(directive.get("allowed_mutations", []), ensure_ascii=False))
    print("final blocked_mutations:" + json.dumps(directive.get("blocked_mutations", []), ensure_ascii=False))
    print("estimated_trade_count_guard:" + json.dumps(directive.get("estimated_trade_count_guard", {}), ensure_ascii=False))
    print(f"是否可能导致交易数低于 min：{'yes' if directive.get('may_reduce_below_min') else 'no'}")
    transcript = _write_outputs(out_dir, role_outputs, failures, directive)
    return {"role_outputs": role_outputs, "final_strategy_directive": directive, "committee_consensus": transcript, "failed_roles": failures}


def directive_prompt_section(market_intel: dict[str, Any] | None, directive: dict[str, Any] | None) -> str:
    if not directive:
        return ""
    safe = {
        "market_regime": (market_intel or {}).get("macro_regime", {}),
        "crypto_regime": (market_intel or {}).get("crypto_regime", {}),
        "final_strategy_directive": directive,
        "data_isolation": {"train_intel_used_for_prompt": True, "validation_intel_used_for_prompt": False, "holdout_intel_used_for_prompt": False},
    }
    return "\n\n========== AI 投资委员会 strategy_directive（仅训练区间情报，可注入 prompt）==========\n" + json.dumps(safe, ensure_ascii=False, indent=2) + "\n"
