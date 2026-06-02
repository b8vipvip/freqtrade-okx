# -*- coding: utf-8 -*-
"""Prompt fingerprint registry and duplicate review."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_SIMILARITY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "threshold": 0.95,
    "compare_fields": ["market_intel_summary", "committee_consensus", "final_strategy_directive", "mutation_spec", "normalized_mutation_intent", "codegen_prompt_fingerprint"],
    "retry_on_duplicate": True,
    "max_retries": 2,
}


def merge_config(goal: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_PROMPT_SIMILARITY_CONFIG)
    raw = (goal or {}).get("prompt_similarity_filter", {}) if isinstance(goal, dict) else {}
    if isinstance(raw, dict):
        cfg.update(raw)
    env_enabled = os.getenv("PROMPT_SIMILARITY_FILTER_ENABLED")
    if env_enabled is not None:
        cfg["enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
    if os.getenv("PROMPT_SIMILARITY_THRESHOLD"):
        cfg["threshold"] = float(os.getenv("PROMPT_SIMILARITY_THRESHOLD", "0.95") or 0.95)
    if os.getenv("PROMPT_SIMILARITY_MAX_RETRIES"):
        cfg["max_retries"] = int(os.getenv("PROMPT_SIMILARITY_MAX_RETRIES", "2") or 2)
    if os.getenv("PROMPT_SIMILARITY_COMPARE_FIELDS"):
        cfg["compare_fields"] = [x.strip() for x in os.getenv("PROMPT_SIMILARITY_COMPARE_FIELDS", "").split(",") if x.strip()]
    return cfg


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = value.lower()
    value = re.sub(r"\b20\d{6}_\d{6}\b|\bv\d{3,}\b|run_[0-9a-f_\-]+", " <id> ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()) if len(tok) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fingerprint_value(value: Any) -> dict[str, Any]:
    norm = normalize_text(value)
    tokens = sorted(token_set(norm))
    return {"hash": hashlib.sha256(norm.encode("utf-8")).hexdigest(), "normalized_text": norm, "tokens": tokens, "token_count": len(tokens)}



def normalized_mutation_intent(fields: dict[str, Any]) -> dict[str, Any]:
    spec = fields.get("mutation_spec") if isinstance(fields, dict) else {}
    directive = fields.get("final_strategy_directive") if isinstance(fields, dict) else {}
    if not isinstance(spec, dict):
        spec = {}
    if not isinstance(directive, dict):
        directive = {}
    return {
        "mutation_type": spec.get("mutation_type") or directive.get("mutation_type") or "",
        "indicators_to_add": spec.get("indicators_to_add") or spec.get("indicator_changes") or directive.get("indicators_to_add") or [],
        "entry_conditions_to_change": spec.get("entry_conditions_to_change") or spec.get("entry_changes") or directive.get("entry_conditions_to_change") or [],
        "pair_specific_rules": spec.get("pair_specific_rules") or directive.get("pair_specific_rules") or {},
        "estimated_trade_count_guard": spec.get("estimated_trade_count_guard") or directive.get("estimated_trade_count_guard") or {},
    }


def _codegen_prompt_fingerprint(fields: dict[str, Any]) -> dict[str, Any]:
    spec = fields.get("mutation_spec") if isinstance(fields, dict) else {}
    directive = fields.get("final_strategy_directive") if isinstance(fields, dict) else {}
    return {
        "mutation_intent": normalized_mutation_intent(fields),
        "codegen_requirements": directive.get("codegen_requirements", []) if isinstance(directive, dict) else [],
        "risk_constraints": directive.get("risk_constraints", []) if isinstance(directive, dict) else [],
        "spec_hash_basis": spec if isinstance(spec, dict) else {},
    }

def build_fingerprints(fields: dict[str, Any], compare_fields: list[str]) -> dict[str, Any]:
    expanded = dict(fields or {})
    expanded["normalized_mutation_intent"] = normalized_mutation_intent(expanded)
    expanded["codegen_prompt_fingerprint"] = _codegen_prompt_fingerprint(expanded)
    return {name: fingerprint_value(expanded.get(name, "")) for name in compare_fields}


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _write_registry(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")


def review_prompts(
    *,
    run_dir: Path,
    registry_path: Path,
    fields: dict[str, Any],
    config: dict[str, Any] | None,
    run_id: str,
    version: str,
    retry_index: int = 0,
    save: bool = True,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_PROMPT_SIMILARITY_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    compare_fields = [str(x) for x in cfg.get("compare_fields", []) if str(x)]
    threshold = float(cfg.get("threshold", 0.95))
    fps = build_fingerprints(fields, compare_fields)
    registry = _load_registry(registry_path)
    best = {"score": 0.0, "field": "", "similar_to_run": "", "similar_to_version": ""}
    for item in registry:
        if str(item.get("run_id", "")) == str(run_id) and str(item.get("version", "")) == str(version):
            continue
        prev = item.get("fingerprints", {}) if isinstance(item, dict) else {}
        for field in compare_fields:
            cur = fps.get(field, {})
            old = prev.get(field, {}) if isinstance(prev, dict) else {}
            if cur.get("hash") and cur.get("hash") == old.get("hash"):
                score = 1.0
            else:
                score = jaccard(set(cur.get("tokens", [])), set(old.get("tokens", [])))
            if score > best["score"]:
                best = {"score": round(score, 4), "field": field, "similar_to_run": str(item.get("run_id", "")), "similar_to_version": str(item.get("version", ""))}
    advisor_similarity = 0.0
    codegen_similarity = 0.0
    for item in registry:
        if str(item.get("run_id", "")) == str(run_id) and str(item.get("version", "")) == str(version):
            continue
        prev = item.get("fingerprints", {}) if isinstance(item, dict) else {}
        for field, target in [("normalized_mutation_intent", "advisor"), ("codegen_prompt_fingerprint", "codegen")]:
            cur = fps.get(field, {})
            old = prev.get(field, {}) if isinstance(prev, dict) else {}
            score = 1.0 if cur.get("hash") == old.get("hash") and cur.get("hash") else jaccard(set(cur.get("tokens", [])), set(old.get("tokens", [])))
            if target == "advisor":
                advisor_similarity = max(advisor_similarity, score)
            else:
                codegen_similarity = max(codegen_similarity, score)
    prompt_duplicate = max(advisor_similarity, codegen_similarity) >= threshold
    duplicate = prompt_duplicate or best["score"] >= threshold
    decision_duplicate = prompt_duplicate
    decision = "retry" if decision_duplicate and bool(cfg.get("retry_on_duplicate", True)) and retry_index < int(cfg.get("max_retries", 2)) else ("skip" if decision_duplicate else "continue")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "version": version,
        "retry_index": retry_index,
        "threshold": threshold,
        "advisor_prompt_similarity": round(advisor_similarity, 4),
        "codegen_prompt_similarity": round(codegen_similarity, 4),
        "similar_to_run": best["similar_to_run"],
        "similar_to_version": best["similar_to_version"],
        "similar_field": best["field"],
        "max_similarity": best["score"],
        "is_duplicate": duplicate,
        "prompt_duplicate": prompt_duplicate,
        "decision": decision,
        "fingerprints": fps,
    }
    if save:
        (run_dir / "prompt_fingerprints.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        registry.append({k: report[k] for k in ["generated_at", "run_id", "version", "retry_index", "threshold", "advisor_prompt_similarity", "codegen_prompt_similarity", "similar_to_run", "similar_to_version", "similar_field", "max_similarity", "is_duplicate", "prompt_duplicate", "decision", "fingerprints"]})
        _write_registry(registry_path, registry)
    return report
