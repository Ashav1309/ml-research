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

from langchain_core.language_models import BaseChatModel


# Tier split per provider:
#   high tier — heavy reasoning, planning, writing
#   low tier  — classification, compression, light edits

OPENAI_DEFAULTS: dict[str, str] = {
    "classifier":     "gpt-4.1-mini",
    "meta":           "gpt-4.1-mini",
    "clarifier":      "gpt-4.1-mini",
    "shallow":        "gpt-4.1-mini",
    "scout":          "gpt-4.1",
    "architect":      "gpt-4.1",
    "specialist":     "gpt-4.1",
    "synthesis":      "gpt-4.1-mini",
    "gap_analyzer":   "gpt-4.1",
    "citation_audit": "gpt-4.1",
    "writer":         "gpt-4.1",
    "refiner":        "gpt-4.1-mini",
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
        2. default for the auto-detected / explicit LLM_PROVIDER
    """
    env_key = f"{role.upper()}_MODEL"
    name = os.getenv(env_key)
    if not name:
        provider = _detect_provider()
        name = _defaults_for(provider).get(role)
    if not name:
        raise ValueError(f"No model configured for role '{role}' (set {env_key})")

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
