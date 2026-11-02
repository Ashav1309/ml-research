"""Search tools used by the researcher agents.

Two AITunnel tools are exposed:

- `make_search_tool()` — returns a synthesized answer with cited sources
  (URLs + snippets/summaries). Used by every agent that searches the web.

- `make_extract_tool()` — fetches and summarizes the content of given URLs.
  This is a compatibility shim, not a real page extractor.

Both share the same `OPENAI_API_KEY`. Wrapping is intentionally thin —
agents handle filtering and citation extraction themselves.

The list of allowed academic domains can be overridden with the
`ACADEMIC_DOMAINS` environment variable (comma-separated). If unset,
`DEFAULT_ACADEMIC_DOMAINS` is used.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.tools import tool
from openai import OpenAI


DEFAULT_ACADEMIC_DOMAINS: list[str] = [
    "arxiv.org",
    "scholar.google.com",
    "semanticscholar.org",
    "base-search.net",
    "core.ac.uk",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "biorxiv.org",
    "medrxiv.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "openreview.net",
    "nature.com",
    "science.org",
    "springeropen.com",
    "sciencedirect.com",
    "ssrn.com",
    "eric.ed.gov",
]


def _require_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing — set it in .env or your shell")


def _get_client() -> OpenAI:
    """Return an OpenAI client configured from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.aitunnel.ru/v1/")
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_allowed_domains() -> list[str]:
    """Return the list of allowed academic domains.

    Reads the `ACADEMIC_DOMAINS` environment variable (comma-separated).
    If the variable is unset or empty, returns `DEFAULT_ACADEMIC_DOMAINS`.
    """
    raw = os.getenv("ACADEMIC_DOMAINS", "")
    if not raw.strip():
        return DEFAULT_ACADEMIC_DOMAINS

    domains = [d.strip() for d in raw.split(",") if d.strip()]
    return domains or DEFAULT_ACADEMIC_DOMAINS


def make_search_tool(max_results: int = 5, model: str | None = None):
    """Build an AITunnel web_search tool.

    The tool returns a synthesized answer with cited sources (URLs +
    snippets/summaries). `max_results` is used only in the system prompt to
    guide the model toward roughly that many distinct sources.
    """
    _require_key()
    client = _get_client()
    chosen_model = model or os.getenv("SCOUT_MODEL", "deepseek-v4-flash-0731")

    SYSTEM = (
        "You are a web search assistant. Use the provided web_search tool to "
        "find information. Cite sources with URLs. Aim for roughly "
        f"{max_results} distinct sources. Return a concise answer with "
        "inline citations and a list of sources at the end."
    )

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web and return a synthesized answer with cited sources."""
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": query},
            ],
            tools=[
                {
                    "type": "aitunnel:web_search",
                    "parameters": {
                        "engine": "exa",
                        "max_results": max_results,
                        "allowed_domains": _get_allowed_domains(),
                    },
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return web_search


def make_extract_tool(model: str | None = None):
    """Build an AITunnel web_extract tool (compatibility shim).

    Takes one or more URLs (already discovered via search) and returns a
    synthesized summary of their content, quoting verbatim where relevant.
    """
    _require_key()
    client = _get_client()
    chosen_model = model or os.getenv("SCOUT_MODEL", "deepseek-v4-flash-0731")

    SYSTEM = (
        "You are a web extraction assistant. Use the provided web_search tool "
        "to fetch and summarize the content of the given URLs. Quote verbatim "
        "where relevant. Return a concise summary with citations."
    )

    @tool("web_extract")
    def web_extract(urls: str) -> str:
        """Fetch and summarize the content of the given URLs."""
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": urls},
            ],
            tools=[
                {
                    "type": "aitunnel:web_search",
                    "parameters": {
                        "engine": "exa",
                        "max_results": 5,
                        "allowed_domains": _get_allowed_domains(),
                    },
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return web_extract


def make_structured_search_tool(max_results: int = 5, model: str | None = None):
    """Build an AITunnel web_search tool that returns structured JSON.

    The tool returns a list of dicts, each with url, title, snippet.
    """
    _require_key()
    client = _get_client()
    chosen_model = model or os.getenv("SCOUT_MODEL", "deepseek-v4-flash-0731")

    SYSTEM = (
        "You are a web search assistant. Search the web and return ONLY a JSON "
        "object with this exact shape: "
        '{"sources": [{"url": "...", "title": "...", "snippet": "..."}]}. '
        "Include 3-5 sources. No prose, no markdown fences, only JSON."
    )

    @tool("web_search_structured")
    def web_search_structured(query: str) -> list[dict[str, str]]:
        """Search the web and return structured sources."""
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": query},
            ],
            tools=[
                {
                    "type": "aitunnel:web_search",
                    "parameters": {
                        "engine": "exa",
                        "max_results": max_results,
                        "allowed_domains": _get_allowed_domains(),
                        "return_citations": True,
                    },
                }
            ],
        )

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return []

        # Try direct JSON parse
        try:
            data = json.loads(content)
            sources = data.get("sources", [])
            if isinstance(sources, list):
                return [
                    {
                        "url": str(s.get("url", "")),
                        "title": str(s.get("title", "")),
                        "snippet": str(s.get("snippet", "")),
                    }
                    for s in sources
                    if isinstance(s, dict)
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: extract first {...} block with regex
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                sources = data.get("sources", [])
                if isinstance(sources, list):
                    return [
                        {
                            "url": str(s.get("url", "")),
                            "title": str(s.get("title", "")),
                            "snippet": str(s.get("snippet", "")),
                        }
                        for s in sources
                        if isinstance(s, dict)
                    ]
            except json.JSONDecodeError:
                pass

        return []

    return web_search_structured
