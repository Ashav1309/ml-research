"""Search tools used by the researcher agents.

Two AITunnel tools are exposed:

- `make_search_tool()` — returns a synthesized answer with cited sources
  (URLs + snippets/summaries). Used by every agent that searches the web.

- `make_extract_tool()` — fetches and summarizes the content of given URLs.
  This is a compatibility shim, not a real page extractor.

Both share the same `OPENAI_API_KEY`. Wrapping is intentionally thin —
agents handle filtering and citation extraction themselves.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool
from openai import OpenAI


def _require_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing — set it in .env or your shell")


def _get_client() -> OpenAI:
    """Return an OpenAI client configured from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.aitunnel.ru/v1/")
    return OpenAI(api_key=api_key, base_url=base_url)


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
            tools=[{"type": "aitunnel:web_search"}],
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
            tools=[{"type": "aitunnel:web_search"}],
        )
        return (resp.choices[0].message.content or "").strip()

    return web_extract
