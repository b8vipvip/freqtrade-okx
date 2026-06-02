# -*- coding: utf-8 -*-
"""Provider discovery and role-based provider pool assembly."""

from __future__ import annotations

import os
import re
from typing import Any

AUTO_BUILD_PROVIDER_POOLS_ENV = "AUTO_BUILD_PROVIDER_POOLS"
FORCE_PROVIDER_POOL_MANUAL_ENV = "FORCE_PROVIDER_POOL_MANUAL"

ROLE_TO_POOL_ENV: dict[str, str] = {
    "market_search": "MARKET_SEARCH_PROVIDER_POOL",
    "committee": "AI_COMMITTEE_ANALYST_PROVIDER_POOL",
    "advisor": "STRATEGY_ADVISOR_PROVIDER_POOL",
    "codegen": "STRATEGY_CODEGEN_PROVIDER_POOL",
    "chairman": "AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL",
}

ROLE_TO_PRIORITY_SUFFIX: dict[str, str] = {
    "market_search": "MARKET_SEARCH",
    "committee": "COMMITTEE",
    "advisor": "ADVISOR",
    "codegen": "CODEGEN",
    "chairman": "CHAIRMAN",
}

POOL_ENV_TO_ROLE: dict[str, str] = {pool_env: role for role, pool_env in ROLE_TO_POOL_ENV.items()}
POOL_ENV_TO_ROLE["AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER"] = "chairman"

OPENAI_COMPATIBLE_TYPES = {
    "openai_compatible",
    "openai_compatible_online",
    "openai_compatible_search",
}

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def auto_provider_pools_enabled() -> bool:
    return env_flag(AUTO_BUILD_PROVIDER_POOLS_ENV, False) and not env_flag(FORCE_PROVIDER_POOL_MANUAL_ENV, False)


def provider_env_prefix(provider_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider_name.strip()).strip("_").upper()
    return f"AI_PROVIDER_{normalized}"


def parse_csv(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def looks_like_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return not lowered or lowered in {"你的key", "your_key", "your-api-key", "你的deepseek_key", "你的glm_key"} or lowered.startswith("你的")


def _provider_id_from_enabled_env(env_name: str) -> str | None:
    match = re.fullmatch(r"AI_PROVIDER_(.+)_ENABLED", env_name)
    if not match:
        return None
    return match.group(1)


def _provider_display_name(provider_id: str) -> str:
    return provider_id.strip().lower()


def _provider_api_key(prefix: str) -> tuple[str, str]:
    direct_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
    key_env_name = (os.getenv(f"{prefix}_API_KEY_ENV") or "").strip()
    if direct_key and not looks_like_placeholder_secret(direct_key):
        return direct_key, f"{prefix}_API_KEY"
    if key_env_name:
        return (os.getenv(key_env_name) or "").strip(), key_env_name
    if direct_key:
        return direct_key, f"{prefix}_API_KEY"
    return "", ""


def _provider_roles(prefix: str) -> list[str]:
    roles = []
    for role in parse_csv(os.getenv(f"{prefix}_ROLE")):
        normalized = role.strip().lower()
        if normalized in ROLE_TO_POOL_ENV and normalized not in roles:
            roles.append(normalized)
    return roles


def _priority_for_role(prefix: str, role: str) -> int:
    suffix = ROLE_TO_PRIORITY_SUFFIX.get(role, role.upper())
    raw = (os.getenv(f"{prefix}_PRIORITY_{suffix}") or "").strip()
    if not raw:
        return 999
    try:
        return int(raw)
    except ValueError:
        return 999


def discover_enabled_providers() -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    for env_name in sorted(os.environ):
        provider_id = _provider_id_from_enabled_env(env_name)
        if not provider_id:
            continue
        prefix = f"AI_PROVIDER_{provider_id}"
        if not env_flag(f"{prefix}_ENABLED", False):
            continue
        api_key, api_key_source = _provider_api_key(prefix)
        provider_type = (os.getenv(f"{prefix}_TYPE") or "openai_compatible").strip().lower()
        providers.append({
            "id": provider_id,
            "name": _provider_display_name(provider_id),
            "prefix": prefix,
            "enabled": True,
            "base_url": (os.getenv(f"{prefix}_BASE_URL") or "").strip(),
            "api_key": api_key,
            "api_key_source": api_key_source,
            "model": (os.getenv(f"{prefix}_MODEL") or "").strip(),
            "type": provider_type,
            "roles": _provider_roles(prefix),
            "capabilities": parse_csv(os.getenv(f"{prefix}_CAPABILITIES")),
            "temperature": (os.getenv(f"{prefix}_TEMPERATURE") or "").strip(),
            "max_tokens": (os.getenv(f"{prefix}_MAX_TOKENS") or "").strip(),
            "timeout": (os.getenv(f"{prefix}_TIMEOUT") or "").strip(),
        })
    return providers




def load_provider_config(provider_name: str, *, default_timeout: int | str = 120) -> dict[str, Any]:
    """Load one AI_PROVIDER_<ID> configuration from environment.

    This is the shared provider loader used by live-search and model pools.  It
    supports direct API keys and API_KEY_ENV indirection, and keeps the original
    provider id for audit logs.
    """
    provider_id = re.sub(r"[^A-Za-z0-9]+", "_", provider_name.strip()).strip("_").upper()
    prefix = f"AI_PROVIDER_{provider_id}"
    api_key, api_key_source = _provider_api_key(prefix)
    timeout_raw = (os.getenv(f"{prefix}_TIMEOUT") or str(default_timeout) or "120").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = float(default_timeout or 120)
    return {
        "id": provider_id,
        "name": provider_name.strip(),
        "prefix": prefix,
        "type": (os.getenv(f"{prefix}_TYPE") or "openai_compatible").strip().lower(),
        "base_url": (os.getenv(f"{prefix}_BASE_URL") or "").strip(),
        "api_key": api_key,
        "api_key_source": api_key_source,
        "model": (os.getenv(f"{prefix}_MODEL") or "").strip(),
        "timeout": timeout,
        "timeout_source": f"{prefix}_TIMEOUT" if os.getenv(f"{prefix}_TIMEOUT") else "default",
    }

def build_auto_provider_pool_names(role: str) -> list[str]:
    candidates: list[tuple[int, str, str]] = []
    for provider in discover_enabled_providers():
        if role not in provider.get("roles", []):
            continue
        priority = _priority_for_role(str(provider.get("prefix") or ""), role)
        candidates.append((priority, str(provider.get("name") or ""), str(provider.get("name") or "")))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, _, name in candidates if name]


def build_auto_provider_pools() -> dict[str, list[str]]:
    return {pool_env: build_auto_provider_pool_names(role) for role, pool_env in ROLE_TO_POOL_ENV.items()}


def provider_pool_names_for_env(provider_pool_env: str) -> list[str]:
    if auto_provider_pools_enabled() and provider_pool_env in POOL_ENV_TO_ROLE:
        return build_auto_provider_pool_names(POOL_ENV_TO_ROLE[provider_pool_env])
    return parse_csv(os.getenv(provider_pool_env))


def print_auto_provider_pool_log() -> None:
    if not auto_provider_pools_enabled():
        return
    pools = build_auto_provider_pools()
    print("\n========== 自动组装 Provider Pool ==========")
    for pool_env in (
        "MARKET_SEARCH_PROVIDER_POOL",
        "AI_COMMITTEE_ANALYST_PROVIDER_POOL",
        "STRATEGY_ADVISOR_PROVIDER_POOL",
        "STRATEGY_CODEGEN_PROVIDER_POOL",
        "AI_COMMITTEE_FINAL_CHAIRMAN_PROVIDER_POOL",
    ):
        print(f"{pool_env}={','.join(pools.get(pool_env, []))}")
