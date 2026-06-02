# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

AI_TOOLS_PATH = Path(__file__).resolve().parents[1] / "ai_tools"
if str(AI_TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(AI_TOOLS_PATH))

import provider_config


def _set_provider(monkeypatch, provider_id: str, roles: str, priorities: dict[str, int]) -> None:
    prefix = f"AI_PROVIDER_{provider_id}"
    monkeypatch.setenv(f"{prefix}_ENABLED", "true")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv(f"{prefix}_API_KEY_ENV", "SHARED_PROVIDER_API_KEY")
    monkeypatch.setenv(f"{prefix}_MODEL", f"model-{provider_id.lower()}")
    monkeypatch.setenv(f"{prefix}_TYPE", "openai_compatible")
    monkeypatch.setenv(f"{prefix}_ROLE", roles)
    for suffix, priority in priorities.items():
        monkeypatch.setenv(f"{prefix}_PRIORITY_{suffix}", str(priority))


def test_auto_provider_pools_builds_all_roles_sorted_by_priority(monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUTO_BUILD_PROVIDER_POOLS", "true")
    monkeypatch.setenv("FORCE_PROVIDER_POOL_MANUAL", "false")
    monkeypatch.setenv("SHARED_PROVIDER_API_KEY", "sk-shared-secret")

    _set_provider(monkeypatch, "SEARCH_FAST", "market_search,committee", {"MARKET_SEARCH": 5, "COMMITTEE": 20})
    _set_provider(monkeypatch, "SEARCH_SLOW", "market_search", {"MARKET_SEARCH": 15})
    _set_provider(monkeypatch, "COMMITTEE_LEAD", "committee,chairman", {"COMMITTEE": 10, "CHAIRMAN": 5})
    _set_provider(monkeypatch, "ADVISOR_FAST", "advisor,codegen", {"ADVISOR": 1, "CODEGEN": 30})
    _set_provider(monkeypatch, "CODEGEN_FAST", "codegen,advisor", {"CODEGEN": 10, "ADVISOR": 30})

    pools = provider_config.build_auto_provider_pools()

    assert pools["MARKET_SEARCH_PROVIDER_POOL"] == ["search_fast", "search_slow"]
    assert pools["AI_COMMITTEE_ANALYST_PROVIDER_POOL"] == ["committee_lead", "search_fast"]
    assert pools["STRATEGY_ADVISOR_PROVIDER_POOL"] == ["advisor_fast", "codegen_fast"]
    assert pools["STRATEGY_CODEGEN_PROVIDER_POOL"] == ["codegen_fast", "advisor_fast"]
    assert pools["AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL"] == ["committee_lead"]

    provider_config.print_auto_provider_pool_log()
    log = capsys.readouterr().out
    assert "========== 自动组装 Provider Pool ==========" in log
    assert "MARKET_SEARCH_PROVIDER_POOL=search_fast,search_slow" in log
    assert "AI_COMMITTEE_ANALYST_PROVIDER_POOL=committee_lead,search_fast" in log
    assert "STRATEGY_ADVISOR_PROVIDER_POOL=advisor_fast,codegen_fast" in log
    assert "STRATEGY_CODEGEN_PROVIDER_POOL=codegen_fast,advisor_fast" in log
    assert "AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL=committee_lead" in log
    assert "sk-shared-secret" not in log

    discovered = provider_config.discover_enabled_providers()
    assert {item["api_key_source"] for item in discovered} == {"SHARED_PROVIDER_API_KEY"}
    assert {item["api_key"] for item in discovered} == {"sk-shared-secret"}


def test_force_provider_pool_manual_disables_auto_pool_log(monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUTO_BUILD_PROVIDER_POOLS", "true")
    monkeypatch.setenv("FORCE_PROVIDER_POOL_MANUAL", "true")
    monkeypatch.setenv("STRATEGY_ADVISOR_PROVIDER_POOL", "manual_advisor")
    _set_provider(monkeypatch, "AUTO_ADVISOR", "advisor", {"ADVISOR": 1})

    assert provider_config.provider_pool_names_for_env("STRATEGY_ADVISOR_PROVIDER_POOL") == ["manual_advisor"]

    provider_config.print_auto_provider_pool_log()
    assert capsys.readouterr().out == ""
