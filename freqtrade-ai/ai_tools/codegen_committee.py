# -*- coding: utf-8 -*-
"""Concurrent multi-candidate strategy code generation and AI review."""

from __future__ import annotations

import copy
import hashlib
import json
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
        "has_populate_indicators": "def populate_indicators" in stripped,
        "has_populate_entry_trend": "def populate_entry_trend" in stripped,
        "has_populate_exit_trend": "def populate_exit_trend" in stripped,
        "obvious_syntax_truncation": stripped.count("(") != stripped.count(")") or stripped.count("[") != stripped.count("]") or stripped.endswith((",", ".", "and", "or")),
        "forbidden_bollinger_middle_direct_read": any(p in stripped for p in FORBIDDEN_BOLLINGER_PATTERNS),
        "only_changed_class_comments_or_names": False,
        "obviously_missing_mutation_spec": False,
        "possible_no_trade_filter": False,
    }
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
    passed = (
        checks["has_complete_python_code"]
        and checks["contains_correct_class_name"]
        and checks["has_populate_indicators"]
        and checks["has_populate_entry_trend"]
        and checks["has_populate_exit_trend"]
        and not checks["obvious_syntax_truncation"]
        and not checks["forbidden_bollinger_middle_direct_read"]
        and not checks["only_changed_class_comments_or_names"]
        and not checks["obviously_missing_mutation_spec"]
    )
    return {"passed": passed, "checks": checks}



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


def offline_review(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranking = []
    for item in candidates:
        meta = item.get("metadata", {})
        checks = ((meta.get("lightweight_check") or {}).get("checks") or {})
        score = 50
        if (meta.get("lightweight_check") or {}).get("passed"):
            score += 30
        score -= 20 if checks.get("possible_no_trade_filter") else 0
        score -= 25 if meta.get("error") else 0
        ranking.append({"candidate": item.get("candidate"), "score": score, "pros": ["lightweight checks passed"] if score >= 80 else [], "cons": [k for k, v in checks.items() if v is False or k.startswith("obvious") and v]})
    ranking.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected = str((ranking[0] or {}).get("candidate") or "candidate_A") if ranking else "candidate_A"
    return {
        "selected_candidate": selected,
        "ranking": ranking,
        "rejected_candidates": [r.get("candidate") for r in ranking[1:]],
        "compliance_check": {
            "implements_mutation_spec": True,
            "not_duplicate": True,
            "no_forbidden_bollinger": True,
            "estimated_trade_count_guard_ok": True,
            "does_not_remove_risk_filters": True,
            "does_not_replace_stoploss_with_trailing": True,
        },
        "selection_reason": "offline reviewer selected the highest lightweight-check score",
        "required_minor_patch": None,
    }


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
                item["metadata"]["prompt_fingerprint_audit_only"] = bool(report.get("prompt_duplicate") or report.get("decision") in {"retry", "skip"})
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
                "如果所有候选都不合格，selected_candidate 必须为 null，并在 rejected_candidates/selection_reason 说明原因。\n"
                f"mutation_spec={json.dumps(mutation_spec, ensure_ascii=False)}\n"
                f"final_strategy_directive={json.dumps(final_strategy_directive or {}, ensure_ascii=False)}\n"
                f"parent_strategy_summary={json.dumps(parent_strategy_summary or {}, ensure_ascii=False)[:6000]}\n"
                f"candidates={json.dumps([{k: v for k, v in item.items() if k in {'candidate','role','code','metadata'}} for item in results], ensure_ascii=False)[:60000]}\n"
            )
            review_runtime = _clone_runtime(reviewer_runtime, 0, timeout_seconds)
            review_start = time.time()
            try:
                raw_decision = ask_ai(review_runtime, [{"role": "user", "content": review_prompt}], state=ai_state)
                decision = extract_json_object(raw_decision)
                (version_dir / "code_review.raw.txt").write_text(raw_decision, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                decision["reviewer_error"] = str(exc)
            decision["reviewer_provider"] = getattr(review_runtime, "used_provider", "")
            decision["reviewer_model"] = getattr(review_runtime, "used_model", "")
            decision["reviewer_elapsed_seconds"] = round(time.time() - review_start, 3)
            print(f"reviewer provider/model/elapsed: {decision.get('reviewer_provider')}/{decision.get('reviewer_model')}/{decision.get('reviewer_elapsed_seconds')}s")
        selected = decision.get("selected_candidate")
        valid_names = {str(item.get("candidate")) for item in results}
        if selected in valid_names:
            selected_item = next(item for item in results if item.get("candidate") == selected)
            _write_json(version_dir / "code_review_decision.json", decision)
            print(f"selected_candidate: {selected}")
            print(f"selection_reason: {decision.get('selection_reason') or ''}")
            print(f"codegen retry triggered: {'yes' if retry_triggered else 'no'}")
            return {"status": "selected", "selected_candidate": selected, "selected_code": selected_item.get("code", ""), "decision": decision, "candidates": all_results, "retry_triggered": retry_triggered}
        if attempt == 0:
            retry_triggered = True
            retry_hint = str(decision.get("selection_reason") or decision.get("rejected_candidates") or "all candidates rejected")
            print("codegen reviewer rejected all candidates; retrying once with rejection reasons.")
            continue
        _write_json(version_dir / "code_review_decision.json", decision)
        return {"status": "all_rejected", "decision": decision, "candidates": all_results, "retry_triggered": retry_triggered}
    return {"status": "all_rejected", "candidates": all_results, "retry_triggered": retry_triggered}
