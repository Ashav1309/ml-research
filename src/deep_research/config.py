"""Model and runtime configuration.

Supports both **OpenAI** and **Anthropic**. You only need one set of keys.

Provider selection:
    1. If env var LLM_PROVIDER is set to "openai" or "anthropic", that wins.
    2. Otherwise auto-detect: whichever of OPENAI_API_KEY / ANTHROPIC_API_KEY
       is present. If both are present, OpenAI is preferred.
    3. If neither key is set, model_for() raises a clear error.

The chosen provider drives the default model for each agent role. You can
still override any single role via `<ROLE>_MODEL` (e.g. WRITER_MODEL=gpt-4.1
even when LLM_PROVIDER=anthropic — provider is inferred from the model name
prefix).

Env var convention:  <ROLE>_MODEL
    e.g. CLASSIFIER_MODEL=gpt-4.1-mini
         WRITER_MODEL=claude-opus-4-7
"""

from __future__ import annotations

import os
import sys

from langchain_core.language_models import BaseChatModel


# Tier split per provider:
#   high tier — heavy reasoning, planning, writing
#   low tier  — classification, compression, light edits

OPENAI_DEFAULTS: dict[str, str] = {
    "classifier":     "deepseek-v4.1-flash-1017",
    "meta":           "deepseek-v4.1-flash-1017",
    "clarifier":      "deepseek-v4.1-flash-1017",
    "shallow":        "deepseek-v4.1-flash-1017",
    "scout":          "deepseek-v4.1-flash-1017",
    "architect":      "deepseek-v4-pro-0813",
    "specialist":     "deepseek-v4-pro-0813",
    "synthesis":      "deepseek-v4-pro-0813",
    "gap_analyzer":   "glm-5.2",
    "citation_audit": "glm-5.2",
    "writer":         "deepseek-v4-pro-0813",
    "refiner":        "glm-5.2",
}

ANTHROPIC_DEFAULTS: dict[str, str] = {
    "classifier":     "claude-haiku-4-5-20251001",
    "meta":           "claude-haiku-4-5-20251001",
    "clarifier":      "claude-haiku-4-5-20251001",
    "shallow":        "claude-haiku-4-5-20251001",
    "scout":          "claude-sonnet-4-6",
    "architect":      "claude-sonnet-4-6",
    "specialist":     "claude-sonnet-4-6",
    "synthesis":      "claude-haiku-4-5-20251001",
    "gap_analyzer":   "claude-sonnet-4-6",
    "citation_audit": "claude-sonnet-4-6",
    "writer":         "claude-opus-4-7",
    "refiner":        "claude-haiku-4-5-20251001",
}


# Backward-compatible aliases for old env var names.
# Example: SHALLOW_RESEARCHER_MODEL is accepted for the "shallow" role.
ROLE_ENV_ALIASES: dict[str, str] = {
    "shallow": "SHALLOW_RESEARCHER",
}


def _detect_provider() -> str:
    """Resolve which provider to use. See module docstring for the rules."""
    explicit = os.getenv("LLM_PROVIDER", "").lower().strip()
    if explicit in ("openai", "anthropic"):
        return explicit
    if explicit:
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'anthropic' (got {explicit!r})"
        )

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_openai:
        return "openai"  # also wins when both are present
    if has_anthropic:
        return "anthropic"

    raise ValueError(
        "No LLM provider key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "(or set LLM_PROVIDER explicitly)."
    )


def _defaults_for(provider: str) -> dict[str, str]:
    return OPENAI_DEFAULTS if provider == "openai" else ANTHROPIC_DEFAULTS


def model_for(role: str, *, temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model for the given agent role.

    Resolution order:
        1. env var <ROLE>_MODEL (provider inferred from the model-name prefix)
        2. legacy alias env var (e.g. SHALLOW_RESEARCHER_MODEL)
        3. default for the auto-detected / explicit LLM_PROVIDER
    """
    primary_key = f"{role.upper()}_MODEL"
    alias_role = ROLE_ENV_ALIASES.get(role)
    alias_key = f"{alias_role}_MODEL" if alias_role else None

    name = os.getenv(primary_key)
    if not name and alias_key:
        name = os.getenv(alias_key)

    if not name:
        provider = _detect_provider()
        default_name = _defaults_for(provider).get(role)
        if not default_name:
            raise ValueError(f"No model configured for role '{role}' (set {primary_key})")
        print(
            f"[config] role={role} env {primary_key} not set, "
            f"using default {default_name}",
            file=sys.stderr,
        )
        name = default_name

    if name.startswith("claude"):
        # Lazy import so a project running purely on OpenAI keys doesn't need
        # the langchain-anthropic package importable at module load time.
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name, temperature=temperature)

    # Any non-claude model is assumed to be OpenAI-compatible and served by
    # AITunnel (or another OpenAI-compatible endpoint). This covers gpt-*,
    # o1/o3, deepseek-*, glm-*, qwen-*, etc.
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.aitunnel.ru/v1/"),
    )


# ─── Runtime limits ──────────────────────────────────────────────────────────

# Max number of architect→specialist→synthesis→gap-analyzer iterations
MAX_DEEP_ITERATIONS: int = int(os.getenv("DEEP_RESEARCH_MAX_ITER", "2"))

# Max searches the shallow researcher may run before being forced to answer
SHALLOW_MAX_SEARCHES: int = int(os.getenv("SHALLOW_MAX_SEARCHES", "3"))

# Max ReAct steps a single specialist may take. Each search OR extract is one
# tool call (≈ 2 graph steps each). Default 18 leaves room for ~4 searches + ~3 extracts.
SPECIALIST_RECURSION_LIMIT: int = int(os.getenv("SPECIALIST_RECURSION_LIMIT", "18"))
