# -*- coding: utf-8 -*-
"""Concurrent multi-candidate strategy code generation and AI review."""

from __future__ import annotations

import copy
import hashlib
import json
import ast
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_CODEGEN_COMMITTEE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "candidate_count": 3,
    "max_parallel_calls": 3,
    "timeout_seconds": 240,
    "reviewer_enabled": True,
    "reviewer_provider_pool": ["GS88_GPT55", "GS88_GPT54", "GS88_CODEX_AUTO_REVIEW", "OPENROUTER_GPT53C"],
}

CODEGEN_ROLES: list[dict[str, str]] = [
    {
        "role": "risk_control_codegen",
        "candidate": "candidate_A",
        "focus": "重点实现风控、pair-specific threshold、避免 BTC/OP 低质量入场；必须防止交易数低于 20。",
    },
    {
        "role": "regime_filter_codegen",
        "candidate": "candidate_B",
        "focus": "重点实现 market regime、choppy market filter、volatility contraction/breakout filter；不允许过度全局过滤；必须保留 SOL/BNB/DOGE 的合理交易机会。",
    },
    {
        "role": "minimal_diff_codegen",
        "candidate": "candidate_C",
        "focus": "重点做最小可控改动；尽量保留父策略结构，只改 mutation_spec 要求的关键位置，避免大改导致不可控。",
    },
]

FORBIDDEN_BOLLINGER_PATTERNS = ["bollinger['middle']", 'bollinger["middle"]', "bollinger['mid']", 'bollinger["mid"]']


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


def merge_config(goal: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CODEGEN_COMMITTEE_CONFIG)
    raw = (goal or {}).get("codegen_committee", {}) if isinstance(goal, dict) else {}
    if isinstance(raw, dict):
        cfg.update(raw)
    cfg["enabled"] = env_flag("CODEGEN_COMMITTEE_ENABLED", bool(cfg.get("enabled", True)))
    cfg["candidate_count"] = env_int("CODEGEN_COMMITTEE_CANDIDATE_COUNT", int(cfg.get("candidate_count", 3) or 3))
    cfg["max_parallel_calls"] = env_int("CODEGEN_COMMITTEE_MAX_PARALLEL_CALLS", int(cfg.get("max_parallel_calls", 3) or 3))
    cfg["timeout_seconds"] = env_int("CODEGEN_COMMITTEE_TIMEOUT_SECONDS", int(cfg.get("timeout_seconds", 240) or 240))
    cfg["reviewer_enabled"] = env_flag("CODEGEN_REVIEWER_ENABLED", bool(cfg.get("reviewer_enabled", True)))
    if os.getenv("CODEGEN_REVIEWER_PROVIDER_POOL"):
        cfg["reviewer_provider_pool"] = [p.strip() for p in os.getenv("CODEGEN_REVIEWER_PROVIDER_POOL", "").split(",") if p.strip()]
    return cfg


def extract_python_code(content: str) -> str:
    m = re.search(r"```python\s*(.*?)```", content or "", flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```\s*(.*?)```", content or "", flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else (content or "")).strip() + "\n"


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        parsed = json.loads(m.group(1))
        if isinstance(parsed, dict):
            return parsed
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    raise ValueError("reviewer response is not JSON object")


def _clone_runtime(runtime: Any, offset: int = 0, timeout_seconds: int | None = None) -> Any:
    cloned = copy.copy(runtime)
    cloned.attempts = []
    cloned.used_model = ""
    cloned.used_provider = ""
    cloned.forced_provider_offset = offset
    if timeout_seconds:
        cloned.timeout_sec = timeout_seconds
    return cloned


def _provider_model_from_runtime(runtime: Any) -> tuple[str, str]:
    return str(getattr(runtime, "used_provider", "") or ""), str(getattr(runtime, "used_model", "") or "")


def _fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def build_role_prompt(base_prompt: str, role_cfg: dict[str, str], retry_hint: str = "") -> str:
    recovery_block = (
        "\nTOO_FEW_TRADES_RECOVERY CODEGEN HARD RULES:\n"
        "- 如果实现 regime/choppy filter，必须提供 pair-specific bypass 或 less strict path，避免所有 pair 无交易。\n"
        "- 不允许所有 entry 条件都是 AND 严格叠加导致仅 1 笔交易。\n"
        "- 必须保留至少 3 个 pair 可能触发 entry；只能对 BTC/OP 保持更严格，必须恢复 SOL/BNB/DOGE 交易机会。\n"
    )
    return (
        f"你是并发代码生成委员会角色：{role_cfg['role']}。\n"
        f"角色重点：{role_cfg['focus']}\n"
        "必须只输出完整 Python 策略代码；不得输出解释。必须真实实现 mutation_spec，不能只改类名/注释/变量名。\n"
        "硬约束：min_trades>=20，理想 25~45，max_trades<=60；不允许 no-trade 风控；不允许 replace_stoploss_with_trailing；不允许 adjust_roi_only；不允许 remove_risk_filter；不允许 news_keyword_rule。\n"
        + recovery_block
        + (f"\nRETRY_REVIEWER_REJECTION_REASONS:\n{retry_hint}\n" if retry_hint else "")
        + "\nBASE_CODEGEN_PROMPT:\n"
        + base_prompt
    )


def lightweight_check(code: str, class_name: str, mutation_spec: dict[str, Any] | None, parent_code: str = "") -> dict[str, Any]:
    stripped = (code or "").strip()
    checks = {
        "has_complete_python_code": bool(stripped and "class " in stripped and "populate_entry_trend" in stripped),
        "contains_correct_class_name": bool(re.search(rf"class\s+{re.escape(class_name)}\s*\(", stripped)),
        "inherits_istrategy": bool(re.search(r"class\s+\w+\s*\([^)]*IStrategy[^)]*\)", stripped)),
        "has_populate_indicators": "def populate_indicators" in stripped,
        "has_populate_entry_trend": "def populate_entry_trend" in stripped,
        "has_populate_exit_trend": "def populate_exit_trend" in stripped,
        "obvious_syntax_truncation": stripped.count("(") != stripped.count(")") or stripped.count("[") != stripped.count("]") or stripped.endswith((",", ".", "and", "or")),
        "python_ast_parse_ok": False,
        "forbidden_bollinger_middle_direct_read": any(p in stripped for p in FORBIDDEN_BOLLINGER_PATTERNS),
        "forbidden_replace_stoploss_with_trailing": "replace_stoploss_with_trailing" in stripped.lower(),
        "forbidden_remove_stoploss": bool(re.search(r"stoploss\s*=\s*(None|0|0\.0)", stripped)) or "remove_stoploss" in stripped.lower(),
        "forbidden_remove_risk_filter": "remove_risk_filter" in stripped.lower(),
        "forbidden_news_keyword_rule": "news_keyword_rule" in stripped.lower() or bool(re.search(r"news|headline|keyword", stripped, flags=re.IGNORECASE)),
        "only_changed_class_comments_or_names": False,
        "obviously_missing_mutation_spec": False,
        "possible_no_trade_filter": False,
    }
    syntax_error = ""
    if stripped:
        try:
            ast.parse(stripped)
            checks["python_ast_parse_ok"] = True
        except SyntaxError as exc:
            syntax_error = f"SyntaxError: {exc.msg} line={exc.lineno} offset={exc.offset}"
        except Exception as exc:  # noqa: BLE001
            syntax_error = f"parse_error: {exc}"
    if parent_code:
        def scrub(x: str) -> str:
            x = re.sub(r"class\s+\w+", "class X", x)
            x = re.sub(r"#.*", "", x)
            x = re.sub(r"\s+", "", x)
            return x
        checks["only_changed_class_comments_or_names"] = scrub(parent_code) == scrub(stripped)
    spec_text = json.dumps(mutation_spec or {}, ensure_ascii=False).lower()
    code_text = stripped.lower()
    tokens = [tok for tok in re.findall(r"[a-zA-Z_]{4,}", spec_text) if tok not in {"mutation_type", "reason", "true", "false", "null"}]
    meaningful_hits = sum(1 for tok in set(tokens[:80]) if tok in code_text)
    checks["obviously_missing_mutation_spec"] = bool(tokens) and meaningful_hits == 0
    entry_match = re.search(r"def\s+populate_entry_trend\s*\(.*?\):(?P<body>.*?)(?:\n\s*def\s+|\nclass\s+|\Z)", stripped, flags=re.DOTALL)
    entry_body = entry_match.group("body") if entry_match else ""
    checks["possible_no_trade_filter"] = entry_body.count(" & ") + entry_body.count(" and ") >= 10 or all(pair in entry_body for pair in ["BTC/USDT", "OP/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"])
    hard_errors: list[str] = []
    if not checks["has_complete_python_code"]:
        hard_errors.append("missing_complete_python_strategy_class")
    if not checks["contains_correct_class_name"]:
        hard_errors.append("missing_expected_strategy_class")
    if not checks["inherits_istrategy"]:
        hard_errors.append("strategy_class_not_inheriting_IStrategy")
    for fn_key, fn_name in [
        ("has_populate_indicators", "populate_indicators"),
        ("has_populate_entry_trend", "populate_entry_trend"),
        ("has_populate_exit_trend", "populate_exit_trend"),
    ]:
        if not checks[fn_key]:
            hard_errors.append(f"missing_{fn_name}")
    if checks["obvious_syntax_truncation"] or not checks["python_ast_parse_ok"]:
        hard_errors.append(syntax_error or "obvious_syntax_error_or_truncation")
    if checks["forbidden_bollinger_middle_direct_read"]:
        hard_errors.append("forbidden_bollinger_middle_direct_read")
    for key in ["forbidden_replace_stoploss_with_trailing", "forbidden_remove_stoploss", "forbidden_remove_risk_filter", "forbidden_news_keyword_rule"]:
        if checks[key]:
            hard_errors.append(key)
    if checks["only_changed_class_comments_or_names"]:
        hard_errors.append("duplicate_strategy_only_changed_class_comments_or_names")
    if checks["obviously_missing_mutation_spec"]:
        hard_errors.append("obviously_missing_mutation_spec")
    soft_warnings: list[str] = []
    if checks["possible_no_trade_filter"]:
        soft_warnings.append("possible_low_trade_or_overstrict_filter")
    passed = not hard_errors
    return {"passed": passed, "checks": checks, "hard_errors": hard_errors, "soft_warnings": soft_warnings}



def _function_body(code: str, function_name: str) -> str:
    match = re.search(rf"def\s+{re.escape(function_name)}\s*\(.*?\):(?P<body>.*?)(?:\n\s*def\s+|\nclass\s+|\Z)", code or "", flags=re.DOTALL)
    body = match.group("body") if match else ""
    body = re.sub(r"#.*", "", body)
    body = re.sub(r"\s+", " ", body).strip().lower()
    return body


def _code_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", (a or "").lower()))
    tb = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", (b or "").lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _candidate_duplicate_report(item: dict[str, Any], accepted: list[dict[str, Any]], parent_code: str, mutation_spec: dict[str, Any]) -> dict[str, Any]:
    code = str(item.get("code") or "")
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""
    parent_hash = hashlib.sha256((parent_code or "").encode("utf-8")).hexdigest() if parent_code else ""
    ind_body = _function_body(code, "populate_indicators")
    entry_body = _function_body(code, "populate_entry_trend")
    parent_ind = _function_body(parent_code, "populate_indicators")
    parent_entry = _function_body(parent_code, "populate_entry_trend")
    reasons: list[str] = []
    if code_hash and code_hash == parent_hash:
        reasons.append("generated_code_hash_matches_parent")
    if parent_code and _code_similarity(code, parent_code) >= 0.995:
        reasons.append("generated_code_ast_similarity_matches_parent")
    if parent_ind and ind_body == parent_ind and parent_entry and entry_body == parent_entry:
        reasons.append("populate_indicators_and_entry_trend_unchanged")
    for prev in accepted:
        prev_code = str(prev.get("code") or "")
        if code_hash and code_hash == hashlib.sha256(prev_code.encode("utf-8")).hexdigest():
            reasons.append(f"generated_code_hash_matches_{prev.get('candidate')}")
            break
        if _code_similarity(code, prev_code) >= 0.995:
            reasons.append(f"generated_code_similarity_matches_{prev.get('candidate')}")
            break
    spec_text = json.dumps(mutation_spec or {}, ensure_ascii=False).lower()
    code_text = code.lower()
    spec_tokens = {tok for tok in re.findall(r"[a-zA-Z_]{4,}", spec_text) if tok not in {"mutation_type", "reason", "true", "false", "null"}}
    if spec_tokens and not any(tok in code_text for tok in list(spec_tokens)[:80]):
        reasons.append("mutation_spec_implementation_diff_missing")
    return {"is_duplicate": bool(reasons), "reasons": reasons, "code_hash": code_hash, "populate_indicators_hash": hashlib.sha256(ind_body.encode()).hexdigest(), "populate_entry_trend_hash": hashlib.sha256(entry_body.encode()).hexdigest()}

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one_candidate(
    *,
    role_cfg: dict[str, str],
    base_prompt: str,
    retry_hint: str,
    code_runtime: Any,
    ask_ai: Callable[..., str],
    ai_state: dict[str, Any],
    version_dir: Path,
    class_name: str,
    mutation_spec: dict[str, Any],
    parent_code: str,
    provider_offset: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    candidate = role_cfg["candidate"]
    cand_dir = version_dir / "codegen_candidates" / candidate
    cand_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_role_prompt(base_prompt, role_cfg, retry_hint=retry_hint)
    (cand_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    runtime = _clone_runtime(code_runtime, provider_offset, timeout_seconds)
    start = time.time()
    raw = ""
    error = ""
    try:
        raw = ask_ai(runtime, [{"role": "user", "content": prompt}], state=ai_state)
        code = extract_python_code(raw)
    except Exception as exc:  # noqa: BLE001 - per-candidate failure must not crash the round
        code = ""
        error = str(exc)
    elapsed = round(time.time() - start, 3)
    provider, model = _provider_model_from_runtime(runtime)
    checks = lightweight_check(code, class_name, mutation_spec, parent_code)
    metadata = {
        "candidate": candidate,
        "role": role_cfg["role"],
        "provider": provider,
        "model": model,
        "elapsed_seconds": elapsed,
        "prompt_fingerprint": _fingerprint(prompt),
        "code_hash": hashlib.sha256((code or "").encode("utf-8")).hexdigest() if code else "",
        "lightweight_check": checks,
        "error": error,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (cand_dir / "raw_response.txt").write_text(raw or error, encoding="utf-8")
    (cand_dir / "strategy.py").write_text(code, encoding="utf-8")
    _write_json(cand_dir / "metadata.json", metadata)
    return {"candidate": candidate, "role": role_cfg["role"], "code": code, "raw_response": raw, "metadata": metadata, "prompt": prompt}


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    if isinstance(value, list):
        return [_coerce_string_list(item)[0] for item in value if _coerce_string_list(item)]
    return [str(value)]


def _reviewer_decision_with_hard_gate(ai_decision: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    for item in candidates:
        meta = item.get("metadata", {}) or {}
        lw = meta.get("lightweight_check", {}) or {}
        hard_errors = _coerce_string_list(lw.get("hard_errors"))
        soft_warnings = _coerce_string_list(lw.get("soft_warnings"))
        checks = lw.get("checks", {}) or {}
        score = 100
        score -= 1000 if hard_errors else 0
        score -= 15 * len(soft_warnings)
        score -= 25 if meta.get("error") else 0
        score -= 10 if checks.get("possible_no_trade_filter") else 0
        ai_rank = next((r for r in ai_decision.get("ranking", []) if isinstance(r, dict) and str(r.get("candidate")) == str(item.get("candidate"))), {}) if isinstance(ai_decision.get("ranking"), list) else {}
        if ai_rank:
            try:
                score += max(-20, min(20, int(float(ai_rank.get("score", 0))) - 50))
            except Exception:
                pass
        ranking.append({
            "candidate": item.get("candidate"),
            "score": score,
            "hard_errors": hard_errors,
            "soft_warnings": soft_warnings,
            "ai_reviewer_notes": ai_rank,
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
        })
    ranking.sort(key=lambda x: x.get("score", -9999), reverse=True)
    selectable = [r for r in ranking if not r.get("hard_errors")]
    if selectable:
        selected = str(selectable[0].get("candidate") or "")
        hard_rejected = [r.get("candidate") for r in ranking if r.get("hard_errors")]
        non_selected = [r.get("candidate") for r in ranking if r.get("candidate") != selected]
        return {
            **ai_decision,
            "selected_candidate": selected,
            "ranking": ranking,
            "rejected_candidates": hard_rejected,
            "non_selected_candidates": non_selected,
            "reviewer_hard_rejected_count": len(hard_rejected),
            "reviewer_non_selected_count": len(non_selected),
            "reviewer_selected_count": 1,
            "selected_candidate_had_hard_errors": False,
            "all_rejected": False,
            "selection_reason": ai_decision.get("selection_reason") or "hard-gated reviewer selected least-bad non-hard-error candidate for syntax/backtest",
            "hard_gate_policy": "Only hard errors can block backtest; soft warnings cannot reject all candidates.",
        }
    return {
        **ai_decision,
        "selected_candidate": None,
        "ranking": ranking,
        "rejected_candidates": [r.get("candidate") for r in ranking],
        "non_selected_candidates": [r.get("candidate") for r in ranking],
        "reviewer_hard_rejected_count": len(ranking),
        "reviewer_non_selected_count": len(ranking),
        "reviewer_selected_count": 0,
        "all_rejected": True,
        "all_rejected_reason": "all candidates have hard_errors",
        "selection_reason": ai_decision.get("selection_reason") or "all candidates have hard_errors; retry allowed once",
        "hard_gate_policy": "Only hard errors can block backtest; retry only when every candidate has hard errors.",
    }




def _candidate_mentions(text: Any) -> set[str]:
    return {m.group(0) for m in re.finditer(r"candidate_[A-Z]", str(text or ""))}


def _ensure_selection_consistency(decision: dict[str, Any], selected: str) -> dict[str, Any]:
    fixed = dict(decision)
    fixed["selected_candidate"] = selected
    reason = str(fixed.get("selection_reason") or "")
    mentioned = _candidate_mentions(reason)
    inconsistent = bool(mentioned and selected not in mentioned)
    fixed["consistency_check"] = {
        "selected_candidate": selected,
        "selection_reason_mentions": sorted(mentioned),
        "selection_reason_consistent": not inconsistent,
        "action": "selection_reason_rewritten_to_structured_selected_candidate" if inconsistent else "ok",
    }
    if inconsistent:
        print(f"WARNING codegen reviewer selection_reason mentions {sorted(mentioned)} but structured selected_candidate={selected}; using structured selected_candidate.")
        fixed["original_selection_reason"] = reason
        fixed["selection_reason"] = f"Structured selected_candidate={selected} is authoritative; original reviewer reason mentioned inconsistent candidate(s) {sorted(mentioned)} and was rewritten before saving."
    elif selected and selected not in reason:
        fixed["selection_reason"] = (reason + f" Selected candidate: {selected}.").strip() if reason else f"Selected candidate: {selected}."
    return fixed


def offline_review(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return _reviewer_decision_with_hard_gate({
        "compliance_check": {
            "implements_mutation_spec": True,
            "not_duplicate": True,
            "no_forbidden_bollinger": True,
            "estimated_trade_count_guard_ok": True,
            "does_not_remove_risk_filters": True,
            "does_not_replace_stoploss_with_trailing": True,
        },
        "required_minor_patch": None,
        "selection_reason": "offline hard-gated reviewer selected highest lightweight-check score",
    }, candidates)


def run_codegen_committee(
    *,
    version_dir: Path,
    base_prompt: str,
    class_name: str,
    mutation_spec: dict[str, Any],
    final_strategy_directive: dict[str, Any] | None,
    parent_strategy_summary: dict[str, Any] | str | None,
    parent_code: str,
    code_runtime: Any,
    reviewer_runtime: Any | None,
    ask_ai: Callable[..., str],
    ai_state: dict[str, Any],
    config: dict[str, Any] | None = None,
    prompt_review_func: Callable[..., dict[str, Any]] | None = None,
    prompt_review_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CODEGEN_COMMITTEE_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    roles = CODEGEN_ROLES[: max(1, int(cfg.get("candidate_count", 3) or 3))]
    max_workers = max(1, int(cfg.get("max_parallel_calls", 3) or 3))
    timeout_seconds = max(1, int(cfg.get("timeout_seconds", 240) or 240))
    print(f"codegen committee enabled: yes")
    print(f"codegen committee candidate_count: {len(roles)}")
    all_results: list[dict[str, Any]] = []
    retry_triggered = False
    retry_hint = ""
    for attempt in range(2):
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, role_cfg in enumerate(roles):
                futures.append(executor.submit(
                    _run_one_candidate,
                    role_cfg=role_cfg,
                    base_prompt=base_prompt,
                    retry_hint=retry_hint,
                    code_runtime=code_runtime,
                    ask_ai=ask_ai,
                    ai_state=ai_state,
                    version_dir=version_dir,
                    class_name=class_name,
                    mutation_spec=mutation_spec,
                    parent_code=parent_code,
                    provider_offset=idx,
                    timeout_seconds=timeout_seconds,
                ))
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x.get("candidate", ""))
        # Per-candidate duplicate review: code/diff evidence has priority; prompt fingerprint is audit-only.
        if prompt_review_func:
            survivors = []
            for item in results:
                kwargs = dict(prompt_review_kwargs or {})
                fields = dict(kwargs.pop("fields", {}) or {})
                fields["mutation_spec"] = mutation_spec
                report = prompt_review_func(fields=fields, version=f"{kwargs.pop('version')}_{item.get('candidate')}", **kwargs)
                duplicate_report = _candidate_duplicate_report(item, survivors, parent_code, mutation_spec)
                item["metadata"]["prompt_similarity_report"] = report
                item["metadata"]["code_duplicate_report"] = duplicate_report
                item["metadata"]["prompt_duplicate"] = bool(duplicate_report.get("is_duplicate"))
                item["metadata"]["prompt_fingerprint_audit_only"] = bool(report.get("prompt_duplicate") or report.get("decision") in {"retry_advisor_only", "prompt_duplicate_stop", "force_continue_after_retry_limit"})
                _write_json(version_dir / "codegen_candidates" / str(item.get("candidate")) / "metadata.json", item["metadata"])
                if not item["metadata"].get("prompt_duplicate"):
                    survivors.append(item)
            if survivors:
                results = survivors
            else:
                all_results.extend(results)
                if attempt == 0:
                    retry_triggered = True
                    retry_hint = "All candidates were code-duplicate/prompt_duplicate. Regenerate with a different implementation path: change indicators, populate_indicators, populate_entry_trend and pair-specific thresholds; do not reuse shared template-only changes."
                    print("all codegen candidates duplicate; regenerating once with different implementation path requirement.")
                    continue
                return {"status": "prompt_duplicate", "all_candidates_prompt_duplicate": True, "candidates": all_results, "retry_triggered": retry_triggered}
        all_results.extend(results)
        decision = offline_review(results)
        if cfg.get("reviewer_enabled") and reviewer_runtime is not None:
            review_prompt = (
                "你是 AI code reviewer。读取 mutation_spec、final_strategy_directive、父策略摘要、候选代码和轻量检查结果，选择唯一最佳候选。只输出 JSON。\n"
                "必须输出字段：selected_candidate, ranking, rejected_candidates, compliance_check, selection_reason, required_minor_patch。\n"
                "重要审核政策：只有硬错误才能拒绝候选进入回测。硬错误仅包括：无完整 Python strategy class、语法明显错误、缺 populate_indicators/populate_entry_trend/populate_exit_trend、未继承 IStrategy、明显没有实现 mutation_spec、明显违反 forbidden rule（replace_stoploss_with_trailing/remove_stoploss/remove_risk_filter/news_keyword_rule）、明显 duplicate strategy。\n"
                "参数可能不优、交易数可能偏低、风险可能偏高均为 soft_warnings，不得因此 all reject；必须从非硬错误候选中选择 least-bad candidate。只有全部候选都有 hard_errors 时 selected_candidate 才能为 null。\n"
                f"mutation_spec={json.dumps(mutation_spec, ensure_ascii=False)}\n"
                f"final_strategy_directive={json.dumps(final_strategy_directive or {}, ensure_ascii=False)}\n"
                f"parent_strategy_summary={json.dumps(parent_strategy_summary or {}, ensure_ascii=False)[:6000]}\n"
                f"candidates={json.dumps([{k: v for k, v in item.items() if k in {'candidate','role','code','metadata'}} for item in results], ensure_ascii=False)[:60000]}\n"
            )
            review_runtime = _clone_runtime(reviewer_runtime, 0, timeout_seconds)
            review_start = time.time()
            try:
                raw_decision = ask_ai(review_runtime, [{"role": "user", "content": review_prompt}], state=ai_state)
                ai_decision = extract_json_object(raw_decision)
                decision = _reviewer_decision_with_hard_gate(ai_decision, results)
                (version_dir / "codegen_review.raw.txt").write_text(raw_decision, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                decision["reviewer_error"] = str(exc)
            decision["reviewer_provider"] = getattr(review_runtime, "used_provider", "")
            decision["reviewer_model"] = getattr(review_runtime, "used_model", "")
            decision["reviewer_elapsed_seconds"] = round(time.time() - review_start, 3)
            print(f"reviewer provider/model/elapsed: {decision.get('reviewer_provider')}/{decision.get('reviewer_model')}/{decision.get('reviewer_elapsed_seconds')}s")
        for rank in decision.get("ranking", []) or []:
            print(f"codegen_review candidate={rank.get('candidate')} score={rank.get('score')} hard_errors={rank.get('hard_errors', [])} soft_warnings={rank.get('soft_warnings', [])}")
        selected = decision.get("selected_candidate")
        valid_names = {str(item.get("candidate")) for item in results}
        if selected in valid_names:
            selected_item = next(item for item in results if item.get("candidate") == selected)
            decision = _ensure_selection_consistency(decision, str(selected))
            _write_json(version_dir / "codegen_review_decision.json", decision)
            print(f"selected_candidate: {selected}")
            print(f"selection_reason: {decision.get('selection_reason') or ''}")
            print(f"selected_candidate_consistency: {json.dumps(decision.get('consistency_check', {}), ensure_ascii=False)}")
            print(f"codegen retry triggered: {'yes' if retry_triggered else 'no'}")
            return {"status": "selected", "selected_candidate": selected, "selected_code": selected_item.get("code", ""), "decision": decision, "candidates": all_results, "retry_triggered": retry_triggered}
        if attempt == 0:
            retry_triggered = True
            retry_hint = str(decision.get("selection_reason") or decision.get("rejected_candidates") or "all candidates rejected")
            print(f"codegen reviewer all_rejected: {decision.get('all_rejected_reason') or decision.get('selection_reason') or decision.get('rejected_candidates')}")
            print("codegen reviewer rejected all candidates with hard_errors; retrying once with rejection reasons.")
            continue
        _write_json(version_dir / "codegen_review_decision.json", decision)
        print(f"codegen reviewer all_rejected final: {decision.get('all_rejected_reason') or decision.get('selection_reason') or decision.get('rejected_candidates')}")
        return {"status": "all_rejected", "decision": decision, "candidates": all_results, "retry_triggered": retry_triggered}
    return {"status": "all_rejected", "candidates": all_results, "retry_triggered": retry_triggered}
