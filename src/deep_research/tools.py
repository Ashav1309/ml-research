"""Search tools used by the researcher agents.

Two Tavily tools are exposed:

- `make_search_tool()` — returns search hits (title, url, short content snippet).
  Cheap, good for discovery. Used by every agent that searches the web.

- `make_extract_tool()` — fetches the full clean text of specific URLs. Used
  by specialists after they've picked promising sources, so they can quote
  verbatim passages instead of relying on snippet text.

Both share the same `TAVILY_API_KEY`. Wrapping is intentionally thin —
agents handle filtering and citation extraction themselves.
"""

from __future__ import annotations

import os

from langchain_tavily import TavilyExtract, TavilySearch


def _require_key() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY missing — set it in .env or your shell")


def make_search_tool(max_results: int = 5):
    """Build a Tavily search tool.

    `search_depth='advanced'` enables deeper crawling for complex queries;
    'basic' is faster and usually sufficient for shallow lookups.
    """
    _require_key()
    return TavilySearch(
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False,  # specialists call Extract on top picks instead
    )


def make_extract_tool():
    """Build a Tavily extract tool.

    Takes one or more URLs (already discovered via search) and returns the
    full cleaned text + a short summary. Lets specialists cite verbatim
    passages rather than search snippets.
`extract_depth='advanced'` handles JS-rendered pages but is ~2x slower
    and slightly more expensive; switch to 'basic' if you don't need it."""
    _require_key()
    return TavilyExtract(extract_depth="advanced")
