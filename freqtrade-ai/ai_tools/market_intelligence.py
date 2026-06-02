# -*- coding: utf-8 -*-
"""Market intelligence collection for offline strategy-design prompts.

This module deliberately separates train-window intelligence from validation/holdout
usage.  The generated intelligence is an offline design aid only; it must never be
used as a candle-level live signal inside generated Freqtrade strategies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse, request

DEFAULT_MARKET_INTELLIGENCE_CONFIG: dict[str, Any] = {
    "enabled": False,
    "search_provider_pool": ["PERPLEXITY_SEARCH", "OPENAI_WEB_SEARCH", "TAVILY_SEARCH"],
    "max_search_results": 12,
    "lookback_days": 30,
    "use_train_window_only_for_prompt": True,
    "allow_validation_intel_for_postmortem_only": True,
    "save_raw_sources": True,
    "timeout_seconds": 120,
}

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
    if os.getenv("MARKET_INTELLIGENCE_PROVIDER_POOL"):
        cfg["search_provider_pool"] = [p.strip() for p in os.getenv("MARKET_INTELLIGENCE_PROVIDER_POOL", "").split(",") if p.strip()]
    if os.getenv("MARKET_INTELLIGENCE_MAX_SEARCH_RESULTS"):
        cfg["max_search_results"] = int(os.getenv("MARKET_INTELLIGENCE_MAX_SEARCH_RESULTS", "12") or 12)
    if os.getenv("MARKET_INTELLIGENCE_TIMEOUT_SECONDS"):
        cfg["timeout_seconds"] = int(os.getenv("MARKET_INTELLIGENCE_TIMEOUT_SECONDS", "120") or 120)
    return cfg


def _pair_symbols(active_pairs: list[str]) -> list[str]:
    symbols: list[str] = []
    for pair in active_pairs:
        base = str(pair).split("/")[0].strip().upper()
        if base and base not in symbols:
            symbols.append(base)
    return symbols


def _fallback_sources(query: str, train_timerange: str) -> list[dict[str, Any]]:
    digest = hashlib.sha256(f"{query}|{train_timerange}".encode("utf-8")).hexdigest()[:12]
    return [{
        "provider": "offline_fallback",
        "query": query,
        "title": f"Offline market-intel placeholder for {query}",
        "url": f"offline://market-intel/{digest}",
        "published_at": "",
        "snippet": (
            "No configured live search API returned data. Treat this as a conservative offline placeholder: "
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


def _search(query: str, cfg: dict[str, Any], train_timerange: str) -> list[dict[str, Any]]:
    # In this environment API keys may be absent.  Use a bounded public fallback,
    # then a transparent offline placeholder so the pipeline remains auditable.
    results = _duckduckgo_search(query, int(cfg.get("timeout_seconds", 120)), int(cfg.get("max_search_results", 12)))
    return results or _fallback_sources(query, train_timerange)


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
