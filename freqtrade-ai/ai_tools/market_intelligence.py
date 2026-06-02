# -*- coding: utf-8 -*-
"""Market intelligence collection for offline strategy-design prompts.

This module deliberately separates train-window intelligence from validation/holdout
usage.  The generated intelligence is an offline design aid only; it must never be
used as a candle-level live signal inside generated Freqtrade strategies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request

from dotenv import load_dotenv
from openai import OpenAI

from provider_config import (
    OPENAI_COMPATIBLE_TYPES,
    auto_provider_pools_enabled,
    load_provider_config,
    looks_like_placeholder_secret,
    provider_pool_names_for_env,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_MARKET_INTELLIGENCE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "search_provider_pool": ["PERPLEXITY_SEARCH", "OPENAI_WEB_SEARCH", "TAVILY_SEARCH"],
    "max_search_results": 12,
    "lookback_days": 30,
    "use_train_window_only_for_prompt": True,
    "allow_validation_intel_for_postmortem_only": True,
    "save_raw_sources": True,
    "timeout_seconds": 120,
    "cache_enabled": True,
    "cache_dir": "user_data/ai_memory/market_intel_cache",
    "force_refresh": False,
    "cache_ttl_days": 3650,
    "run_once_per_run": True,
    "reuse_same_timerange": True,
    "prompt_schema_version": "market_intel_v2",
}

_RUN_LEVEL_MARKET_INTEL_CACHE: dict[str, dict[str, Any]] = {}

SEARCH_TOPICS = [
    "BTC ETH crypto market trend",
    "FOMC CPI interest rate expectation crypto market",
    "USD DXY dollar risk sentiment crypto",
    "Nasdaq S&P 500 risk sentiment crypto",
    "geopolitical risks markets crypto",
    "crypto regulation market risk",
    "Bitcoin ETF flows Ethereum ETF flows",
    "stablecoin liquidity crypto market",
    "crypto liquidation funding rate leverage risk",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_config(goal: dict[str, Any] | None) -> dict[str, Any]:
    raw = (goal or {}).get("market_intelligence", {}) if isinstance(goal, dict) else {}
    cfg = dict(DEFAULT_MARKET_INTELLIGENCE_CONFIG)
    if isinstance(raw, dict):
        cfg.update(raw)
    env_enabled = os.getenv("MARKET_INTELLIGENCE_ENABLED")
    if env_enabled is not None:
        cfg["enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
    auto_market_pool = provider_pool_names_for_env("MARKET_SEARCH_PROVIDER_POOL") if auto_provider_pools_enabled() else []
    if auto_market_pool:
        cfg["search_provider_pool"] = auto_market_pool
    elif os.getenv("MARKET_INTELLIGENCE_PROVIDER_POOL"):
        cfg["search_provider_pool"] = [p.strip() for p in os.getenv("MARKET_INTELLIGENCE_PROVIDER_POOL", "").split(",") if p.strip()]
    elif os.getenv("MARKET_SEARCH_PROVIDER_POOL"):
        cfg["search_provider_pool"] = [p.strip() for p in os.getenv("MARKET_SEARCH_PROVIDER_POOL", "").split(",") if p.strip()]
    pool = [str(p).strip() for p in cfg.get("search_provider_pool", []) if str(p).strip()]
    if _openrouter_sonar_pro_configured() and "OPENROUTER_SONAR_PRO" not in [p.upper() for p in pool]:
        pool.append("OPENROUTER_SONAR_PRO")
        cfg["search_provider_pool"] = pool
    if os.getenv("MARKET_INTELLIGENCE_MAX_SEARCH_RESULTS"):
        cfg["max_search_results"] = int(os.getenv("MARKET_INTELLIGENCE_MAX_SEARCH_RESULTS", "12") or 12)
    if os.getenv("MARKET_INTELLIGENCE_TIMEOUT_SECONDS"):
        cfg["timeout_seconds"] = int(os.getenv("MARKET_INTELLIGENCE_TIMEOUT_SECONDS", "120") or 120)
    env_map = {
        "MARKET_INTEL_CACHE_ENABLED": ("cache_enabled", "bool"),
        "MARKET_INTEL_CACHE_DIR": ("cache_dir", "str"),
        "MARKET_INTEL_FORCE_REFRESH": ("force_refresh", "bool"),
        "MARKET_INTEL_CACHE_TTL_DAYS": ("cache_ttl_days", "int"),
        "MARKET_INTEL_RUN_ONCE_PER_RUN": ("run_once_per_run", "bool"),
        "MARKET_INTEL_REUSE_SAME_TIMERANGE": ("reuse_same_timerange", "bool"),
    }
    for env_name, (cfg_key, kind) in env_map.items():
        raw_env = os.getenv(env_name)
        if raw_env is None:
            continue
        if kind == "bool":
            cfg[cfg_key] = raw_env.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif kind == "int":
            cfg[cfg_key] = int(raw_env or cfg.get(cfg_key) or 0)
        else:
            cfg[cfg_key] = raw_env
    return cfg



def _openrouter_sonar_pro_configured() -> bool:
    prefix = "AI_PROVIDER_OPENROUTER_SONAR_PRO"
    return bool((os.getenv(f"{prefix}_MODEL") or "").strip() and ((os.getenv(f"{prefix}_API_KEY") or "").strip() or (os.getenv(f"{prefix}_API_KEY_ENV") or "").strip()))

def _pair_symbols(active_pairs: list[str]) -> list[str]:
    symbols: list[str] = []
    for pair in active_pairs:
        base = str(pair).split("/")[0].strip().upper()
        if base and base not in symbols:
            symbols.append(base)
    return symbols


def _failure_summary(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return "no live search providers configured"
    parts = []
    for item in failures:
        parts.append(
            f"{item.get('provider')} model={item.get('model') or '<missing>'} "
            f"base_url={item.get('base_url') or '<missing>'} "
            f"error={item.get('exception_type')}: {item.get('error_summary')}"
        )
    return " | ".join(parts)[:4000]


def _fallback_sources(query: str, train_timerange: str, failures: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    digest = hashlib.sha256(f"{query}|{train_timerange}".encode("utf-8")).hexdigest()[:12]
    failure_text = _failure_summary(failures or [])
    print(f"market_intelligence offline_fallback query={query!r}; provider_failures={failure_text}")
    return [{
        "provider": "offline_fallback",
        "model": "",
        "query": query,
        "title": f"Offline market-intel placeholder for {query}",
        "url": f"offline://market-intel/{digest}",
        "published_at": "",
        "content": "",
        "sources": [],
        "provider_failures": failures or [],
        "snippet": (
            "All configured live search providers failed. Failure reasons: "
            f"{failure_text}. Treat this as a conservative offline placeholder: "
            "prefer robust risk filters, avoid news-specific rules, and do not use validation/holdout news in prompts."
        ),
    }]


def _duckduckgo_search(query: str, timeout: int, max_results: int) -> list[dict[str, Any]]:
    """Best-effort public search fallback; returns empty on any network/API issue."""
    url = "https://api.duckduckgo.com/?" + parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    try:
        with request.urlopen(url, timeout=max(3, min(timeout, 20))) as resp:  # noqa: S310 - user-configured public API fallback
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for item in payload.get("RelatedTopics", []) or []:
        if isinstance(item, dict) and item.get("FirstURL"):
            out.append({
                "provider": "OPENAI_WEB_SEARCH" if not os.getenv("TAVILY_API_KEY") else "public_search_fallback",
                "query": query,
                "title": str(item.get("Text") or query)[:160],
                "url": str(item.get("FirstURL") or ""),
                "published_at": "",
                "snippet": str(item.get("Text") or "")[:500],
            })
        for sub in item.get("Topics", []) if isinstance(item, dict) else []:
            if isinstance(sub, dict) and sub.get("FirstURL"):
                out.append({"provider": "public_search_fallback", "query": query, "title": str(sub.get("Text") or query)[:160], "url": str(sub.get("FirstURL") or ""), "published_at": "", "snippet": str(sub.get("Text") or "")[:500]})
        if len(out) >= max_results:
            break
    return out[:max_results]


def _http_body_summary(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.text
            except Exception:  # noqa: BLE001 - best-effort error logging only.
                body = None
    if isinstance(body, (dict, list)):
        text = json.dumps(body, ensure_ascii=False)
    else:
        text = str(body or "")
    return text.replace("\n", " ")[:1000]


def _provider_failure(provider: dict[str, Any], exc: Exception, *, prefix: str = "") -> dict[str, Any]:
    failure = {
        "provider": provider.get("id") or provider.get("name") or "",
        "model": provider.get("model") or "",
        "base_url": provider.get("base_url") or "",
        "exception_type": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None),
        "http_body_summary": _http_body_summary(exc),
        "error_summary": (prefix + str(exc)).replace("\n", " ")[:1000],
    }
    print(
        "market_intelligence live_search_failed "
        f"provider={failure['provider']} model={failure['model']} base_url={failure['base_url']} "
        f"exception={failure['exception_type']} http_status={failure['http_status']} "
        f"http_body={failure['http_body_summary']} error={failure['error_summary']}"
    )
    return failure


def _extract_response_sources(message: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    annotations = getattr(message, "annotations", None) or []
    for ann in annotations:
        url_citation = getattr(ann, "url_citation", None)
        if url_citation is None and isinstance(ann, dict):
            url_citation = ann.get("url_citation")
        if url_citation:
            if isinstance(url_citation, dict):
                sources.append({"title": url_citation.get("title", ""), "url": url_citation.get("url", "")})
            else:
                sources.append({"title": getattr(url_citation, "title", ""), "url": getattr(url_citation, "url", "")})
    return sources


def _search_prompt(query: str, train_timerange: str) -> str:
    return (
        "Search the web for current crypto/macroeconomic information relevant to offline Freqtrade strategy design. "
        "Return concise bullet points with source names/URLs when available. "
        "Do not provide trading signals. "
        f"Training timerange context: {train_timerange}. Query: {query}"
    )


def _openai_compatible_search(query: str, cfg: dict[str, Any], train_timerange: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    pool = provider_pool_names_for_env("MARKET_SEARCH_PROVIDER_POOL")
    if not pool:
        pool = [str(p).strip() for p in cfg.get("search_provider_pool", []) if str(p).strip()]
    if _openrouter_sonar_pro_configured() and "OPENROUTER_SONAR_PRO" not in [p.upper() for p in pool]:
        pool.append("OPENROUTER_SONAR_PRO")
    print(f"market_intelligence provider_pool={pool}")
    for provider_name in pool:
        provider = load_provider_config(provider_name, default_timeout=int(cfg.get("timeout_seconds", 120)))
        provider_id = str(provider.get("id") or provider_name)
        model = str(provider.get("model") or "")
        base_url = str(provider.get("base_url") or "")
        provider_type = str(provider.get("type") or "openai_compatible").lower()
        if provider_type not in OPENAI_COMPATIBLE_TYPES:
            failures.append(_provider_failure(provider, RuntimeError(f"unsupported provider TYPE={provider_type}")))
            continue
        if not model or not provider.get("api_key") or looks_like_placeholder_secret(str(provider.get("api_key") or "")):
            failures.append(_provider_failure(provider, RuntimeError("missing API_KEY or MODEL")))
            continue
        try:
            print(f"market_intelligence live_search_call provider={provider_id} model={model} base_url={base_url}")
            client = OpenAI(api_key=str(provider["api_key"]), base_url=base_url or None, timeout=float(provider.get("timeout") or cfg.get("timeout_seconds", 120)))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a web-search market intelligence assistant. Cite source URLs when available."},
                    {"role": "user", "content": _search_prompt(query, train_timerange)},
                ],
                temperature=0.2,
            )
            message = response.choices[0].message
            content = str(getattr(message, "content", "") or "").strip()
            sources = _extract_response_sources(message)
            if not content and not sources:
                raise RuntimeError("empty live search response")
            print(f"market_intelligence live_search_success provider={provider_id} model={model} base_url={base_url} content_chars={len(content)} sources={len(sources)}")
            return ([{
                "provider": provider_id,
                "model": model,
                "base_url": base_url,
                "query": query,
                "title": f"Live market intelligence: {query}",
                "url": sources[0].get("url", "") if sources else "",
                "published_at": "",
                "content": content,
                "sources": sources,
                "snippet": content[:800],
            }], failures)
        except Exception as exc:  # noqa: BLE001 - try next configured provider and audit reason.
            failures.append(_provider_failure(provider, exc))
    return [], failures


def _search(query: str, cfg: dict[str, Any], train_timerange: str) -> list[dict[str, Any]]:
    live_results, failures = _openai_compatible_search(query, cfg, train_timerange)
    if live_results:
        return live_results
    if bool(cfg.get("allow_public_search_fallback", False)):
        results = _duckduckgo_search(query, int(cfg.get("timeout_seconds", 120)), int(cfg.get("max_search_results", 12)))
        if results:
            return results
    return _fallback_sources(query, train_timerange, failures)


def _classify_regimes(sources: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = " ".join(str(s.get("title", "")) + " " + str(s.get("snippet", "")) for s in sources).lower()
    risk_off_hits = len(re.findall(r"risk off|recession|hawkish|war|sanction|crackdown|liquidation|selloff|outflow|inflation", text))
    risk_on_hits = len(re.findall(r"risk on|rally|dovish|inflow|liquidity|easing|soft landing|etf inflow", text))
    if risk_off_hits > risk_on_hits + 1:
        risk = "risk_off"
    elif risk_on_hits > risk_off_hits + 1:
        risk = "risk_on"
    else:
        risk = "neutral"
    trend = "volatile" if re.search(r"volatile|liquidation|funding|leverage", text) else ("up" if risk == "risk_on" else ("down" if risk == "risk_off" else "range"))
    liquidity = "tight" if re.search(r"tight|outflow|hawkish|dollar", text) else ("loose" if re.search(r"inflow|easing|liquidity", text) else "neutral")
    leverage = "high" if re.search(r"liquidation|funding|leverage", text) else "medium"
    evidence = [s for s in sources[:6]]
    return (
        {"risk_on_off": risk, "confidence": 0.55 if sources else 0.0, "evidence": evidence},
        {"trend": trend, "liquidity": liquidity, "leverage_risk": leverage, "evidence": evidence},
    )



def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _cache_dir(cfg: dict[str, Any]) -> Path:
    raw = Path(str(cfg.get("cache_dir") or "user_data/ai_memory/market_intel_cache"))
    return raw if raw.is_absolute() else ROOT_DIR / raw


def _provider_cache_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    pool = provider_pool_names_for_env("MARKET_SEARCH_PROVIDER_POOL") or [str(p).strip() for p in cfg.get("search_provider_pool", []) if str(p).strip()]
    if _openrouter_sonar_pro_configured() and "OPENROUTER_SONAR_PRO" not in [p.upper() for p in pool]:
        pool.append("OPENROUTER_SONAR_PRO")
    identities = []
    for name in pool:
        provider = load_provider_config(name, default_timeout=int(cfg.get("timeout_seconds", 120)))
        identities.append({"provider_id": str(provider.get("id") or name), "model": str(provider.get("model") or "")})
    return {"pool": identities, "primary_provider_id": (identities[0].get("provider_id") if identities else ""), "primary_model": (identities[0].get("model") if identities else "")}


def _market_intel_cache_key(cfg: dict[str, Any], queries: list[str], train_timerange: str, active_pairs: list[str]) -> tuple[str, dict[str, Any]]:
    provider_identity = _provider_cache_identity(cfg)
    metadata = {
        "provider_id": provider_identity.get("primary_provider_id", ""),
        "model": provider_identity.get("primary_model", ""),
        "provider_pool": provider_identity.get("pool", []),
        "normalized_query": [_normalize_query(q) for q in queries],
        "train_timerange": train_timerange,
        "active_pairs": sorted(active_pairs),
        "lookback_days": int(cfg.get("lookback_days", 30) or 30),
        "market_intel_prompt_schema_version": str(cfg.get("prompt_schema_version") or "market_intel_v2"),
    }
    digest = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, metadata


def _read_market_cache(cache_path: Path, ttl_days: int) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(str(payload.get("metadata", {}).get("cached_at", "")).replace("Z", "+00:00"))
        if (datetime.now(timezone.utc) - created).days > ttl_days:
            return None
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_market_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)

def collect_market_intelligence(
    *,
    run_dir: Path,
    train_timerange: str,
    validation_timeranges: list[str] | None,
    active_pairs: list[str],
    timeframe: str,
    failure_type: str,
    pair_attribution: dict[str, Any] | None,
    last_run_summary: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_MARKET_INTELLIGENCE_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    out_dir = run_dir / "market_intel"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = [f"{topic} {train_timerange}" for topic in SEARCH_TOPICS]
    for symbol in _pair_symbols(active_pairs):
        if symbol in {"BTC", "OP", "SOL", "BNB", "DOGE", "ETH"}:
            queries.append(f"{symbol} crypto news market {train_timerange}")
    cache_key, cache_metadata = _market_intel_cache_key(cfg, queries, train_timerange, active_pairs)
    run_cache_key = cache_key if bool(cfg.get("reuse_same_timerange", True)) else f"{cache_key}:{run_dir}"
    cache_path = _cache_dir(cfg) / f"{cache_key}.json"
    if bool(cfg.get("run_once_per_run", True)) and run_cache_key in _RUN_LEVEL_MARKET_INTEL_CACHE and not bool(cfg.get("force_refresh", False)):
        print(f"market_intelligence cache_hit scope=run key={cache_key}")
        print(f"market_intelligence cache_path={cache_path}")
        print("market_intelligence skip_live_search=true")
        cached = _RUN_LEVEL_MARKET_INTEL_CACHE[run_cache_key]
        raw_sources = list(cached.get("raw_sources", []))
        intel = dict(cached.get("market_intel", {}))
        (out_dir / "raw_sources.json").write_text(json.dumps(raw_sources, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "market_intel.json").write_text(json.dumps(intel, ensure_ascii=False, indent=2), encoding="utf-8")
        return intel
    if bool(cfg.get("cache_enabled", True)) and not bool(cfg.get("force_refresh", False)):
        cached_payload = _read_market_cache(cache_path, int(cfg.get("cache_ttl_days", 3650) or 3650))
        if cached_payload:
            print(f"market_intelligence cache_hit scope=disk key={cache_key} path={cache_path}")
            print(f"market_intelligence cache_path={cache_path}")
            print("market_intelligence skip_live_search=true")
            raw_sources = list(cached_payload.get("raw_sources", []))
            intel = dict(cached_payload.get("market_intel", {}))
            _RUN_LEVEL_MARKET_INTEL_CACHE[run_cache_key] = {"raw_sources": raw_sources, "market_intel": intel}
            (out_dir / "raw_sources.json").write_text(json.dumps(raw_sources, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "market_intel.json").write_text(json.dumps(intel, ensure_ascii=False, indent=2), encoding="utf-8")
            return intel
    print(f"market_intelligence cache_miss key={cache_key}")
    print("market_intelligence skip_live_search=false")
    max_total = max(1, int(cfg.get("max_search_results", 12)))
    raw_sources: list[dict[str, Any]] = []
    deadline = time.time() + max(5, int(cfg.get("timeout_seconds", 120)))
    for q in queries:
        if len(raw_sources) >= max_total or time.time() > deadline:
            break
        raw_sources.extend(_search(q, cfg, train_timerange))
    raw_sources = raw_sources[:max_total]
    macro, crypto = _classify_regimes(raw_sources)
    pair_notes = {pair: [] for pair in active_pairs}
    for pair in active_pairs:
        base = pair.split("/")[0].upper()
        notes = [s for s in raw_sources if base.lower() in (str(s.get("title", "")) + " " + str(s.get("snippet", ""))).lower()]
        pair_notes[pair] = notes[:3]

    intel = {
        "generated_at": _utc_now(),
        "train_timerange": train_timerange,
        "validation_timeranges_postmortem_only": validation_timeranges or [],
        "active_pairs": active_pairs,
        "timeframe": timeframe,
        "failure_type": failure_type,
        "macro_regime": macro,
        "crypto_regime": crypto,
        "pair_notes": pair_notes,
        "strategy_implications": [
            "BTC/OP should use stricter entry threshold when stoploss exposure is high",
            "avoid choppy market and unstable leverage/funding regimes",
            "do not increase trade frequency just to satisfy trade-count targets",
            "convert news context into backtestable regime/volatility/liquidity proxies only",
        ],
        "inputs_snapshot": {
            "pair_attribution_summary": (pair_attribution or {}).get("summary", {}) if isinstance(pair_attribution, dict) else {},
            "last_run_failure": (last_run_summary or {}).get("failure_reason") or (last_run_summary or {}).get("invalid_reason") if isinstance(last_run_summary, dict) else "",
        },
        "do_not_use_for_live_signal": True,
        "lookahead_warning": "This intel is for offline strategy design only and must not be used as candle-level live signal.",
        "data_isolation": {
            "train_intel_used_for_prompt": True,
            "validation_intel_used_for_prompt": False,
            "holdout_intel_used_for_prompt": False,
            "validation_intel_policy": "postmortem_only_abstract_metrics",
        },
    }
    if bool(cfg.get("save_raw_sources", True)):
        (out_dir / "raw_sources.json").write_text(json.dumps(raw_sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "market_intel.json").write_text(json.dumps(intel, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Market Intelligence", "", f"- generated_at: {intel['generated_at']}", f"- train_timerange: {train_timerange}", f"- active_pairs: {', '.join(active_pairs)}", "", "## Macro Regime", json.dumps(macro, ensure_ascii=False, indent=2), "", "## Crypto Regime", json.dumps(crypto, ensure_ascii=False, indent=2), "", "## Strategy Implications"]
    md.extend(f"- {x}" for x in intel["strategy_implications"])
    md.extend(["", "## Raw Sources"])
    md.extend(f"- [{s.get('title')}]({s.get('url')}) — {s.get('snippet')}" for s in raw_sources)
    (out_dir / "market_intel.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cache_payload = {"metadata": {**cache_metadata, "cache_key": cache_key, "cached_at": _utc_now()}, "raw_response": raw_sources, "raw_sources": raw_sources, "sources": raw_sources, "summary": {"macro_regime": macro, "crypto_regime": crypto}, "market_intel": intel}
    _RUN_LEVEL_MARKET_INTEL_CACHE[run_cache_key] = {"raw_sources": raw_sources, "market_intel": intel}
    if bool(cfg.get("cache_enabled", True)):
        _write_market_cache(cache_path, cache_payload)
        print(f"market_intelligence cache_write key={cache_key} path={cache_path}")
    return intel


def summarize_for_prompt(intel: dict[str, Any] | None, max_chars: int = 5000) -> str:
    if not isinstance(intel, dict) or not intel:
        return ""
    safe = {
        "train_timerange": intel.get("train_timerange"),
        "active_pairs": intel.get("active_pairs"),
        "macro_regime": intel.get("macro_regime"),
        "crypto_regime": intel.get("crypto_regime"),
        "pair_notes": intel.get("pair_notes"),
        "strategy_implications": intel.get("strategy_implications"),
        "do_not_use_for_live_signal": True,
        "lookahead_warning": intel.get("lookahead_warning"),
    }
    return json.dumps(safe, ensure_ascii=False, sort_keys=True)[:max_chars]


def _goal_pairs(goal: dict[str, Any]) -> list[str]:
    pairs = goal.get("runtime_active_pairs") or goal.get("pairs") or goal.get("active_pairs") or []
    return [str(p) for p in pairs if str(p).strip()] or ["BTC/USDT", "OP/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Market intelligence live-search tester")
    parser.add_argument("--test-search", action="store_true", help="Only test live search and write market_intel artifacts; do not run backtests")
    parser.add_argument("--test-cache", action="store_true", help="Test market intelligence local cache by collecting twice and expecting the second call to hit cache")
    parser.add_argument("--goal", default=str(ROOT_DIR / "ai_tools" / "optimization_goal.json"), help="Path to optimization_goal.json")
    parser.add_argument("--run-dir", default="", help="Optional output run directory")
    args = parser.parse_args(argv)
    if not (args.test_search or args.test_cache):
        parser.error("Use --test-search or --test-cache")
    goal_path = Path(args.goal)
    goal = json.loads(goal_path.read_text(encoding="utf-8")) if goal_path.exists() else {}
    cfg = merge_config(goal)
    cfg["enabled"] = True
    test_root = ROOT_DIR / "user_data" / "backtest_results"
    test_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix="market_intel_test_", dir=str(test_root)))
    train_timerange = str((goal.get("train_period") or {}).get("timerange") or goal.get("timerange") or "")
    validation_timeranges = [str(v.get("timerange")) for v in goal.get("validation_periods", []) if isinstance(v, dict) and v.get("timerange")]
    intel = collect_market_intelligence(
        run_dir=run_dir,
        train_timerange=train_timerange,
        validation_timeranges=validation_timeranges,
        active_pairs=_goal_pairs(goal),
        timeframe=str(goal.get("timeframe") or "5m"),
        failure_type="test_search",
        pair_attribution={},
        last_run_summary={},
        config=cfg,
    )
    if args.test_cache:
        intel2 = collect_market_intelligence(
            run_dir=run_dir,
            train_timerange=train_timerange,
            validation_timeranges=validation_timeranges,
            active_pairs=_goal_pairs(goal),
            timeframe=str(goal.get("timeframe") or "5m"),
            failure_type="test_cache_second_call",
            pair_attribution={},
            last_run_summary={},
            config=cfg,
        )
        if summarize_for_prompt(intel) != summarize_for_prompt(intel2):
            print("market_intelligence test_cache mismatch")
            return 3
    print(f"market_intelligence test_search output_dir={run_dir / 'market_intel'}")
    providers = sorted({str(s.get("provider")) for s in (json.loads((run_dir / "market_intel" / "raw_sources.json").read_text(encoding="utf-8")) if (run_dir / "market_intel" / "raw_sources.json").exists() else [])})
    print(f"market_intelligence test_search providers={providers}")
    if args.test_cache:
        print("market_intelligence test_cache ok")
        return 0
    return 0 if any(p != "offline_fallback" for p in providers) else 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
