"""Shallow research path.

`shallow_researcher_node` runs a small ReAct loop with the Tavily tool and
produces a concise answer + citations. It self-judges sufficiency: if it could
not answer confidently, it sets `shallow_sufficient=False` so the graph escalates
to the deep path (scout → architect → ...).

`shallow_finalize_node` formats the audited shallow answer as the final report.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from deep_research.config import SHALLOW_MAX_SEARCHES, model_for
from deep_research.prompts import SHALLOW_RESEARCHER_SYSTEM
from deep_research.state import Citation, ResearchState
from deep_research.tools import make_search_tool, make_structured_search_tool

_shallow_agent = None


def _get_shallow_agent():
    """Lazy singleton — the ReAct agent is reusable across invocations."""
    global _shallow_agent
    if _shallow_agent is None:
        _shallow_agent = create_react_agent(
            model=model_for("shallow"),
            tools=[
                make_search_tool(max_results=SHALLOW_MAX_SEARCHES),
                make_structured_search_tool(max_results=SHALLOW_MAX_SEARCHES),
            ],
            prompt=SHALLOW_RESEARCHER_SYSTEM,
        )
    return _shallow_agent


_URL_RE = re.compile(r"https?://[^\s\)\]]+")


def _extract_citations_from_text(text: str) -> list[Citation]:
    """Pull naked URLs out of the agent's final answer as citations.

    The shallow researcher is instructed to inline citations as URLs, so we
    extract them with a permissive regex rather than asking for structured JSON
    (cheaper, fewer brittle JSON parses)."""
    urls = list(dict.fromkeys(_URL_RE.findall(text)))  # dedupe, preserve order
    return [Citation(url=url) for url in urls]


def shallow_researcher_node(state: ResearchState) -> dict[str, Any]:
    """Run a small ReAct loop. Set shallow_sufficient flag for the router."""
    agent = _get_shallow_agent()
    result = agent.invoke({"messages": [HumanMessage(content=state["question"])]})
    answer = result["messages"][-1].content

    sufficient = "INSUFFICIENT" not in answer.upper()
    citations = _extract_citations_from_text(answer)

    # Store structured search results for source invariant enforcement
    last_search_results = state.get("_last_search_results", [])
    search_urls = {c["url"] for c in last_search_results if isinstance(c, dict) and "url" in c}

    for c in citations:
        if c.url not in search_urls:
            state.setdefault("warnings", []).append(
                f"citation url not from web_search: {c.url}"
            )

    return {
        "shallow_answer": answer,
        "shallow_citations": citations,
        "shallow_sufficient": sufficient,
        "_last_search_results": last_search_results,
    }


def shallow_finalize_node(state: ResearchState) -> dict[str, Any]:
    """Promote the (audited) shallow answer to final_report.

    The citation_audit node has already run and may have flagged issues. We
    surface them as a short note at the end of the report rather than rerunning
    the writer for a shallow query."""
    answer = state.get("shallow_answer", "(no shallow answer produced)")
    audit = state.get("citation_audit")
    issues = audit.flagged_issues if audit else []

    report = answer.strip()
    if issues:
        report += "\n\n---\n**Citation audit flags:**\n" + "\n".join(
            f"- {issue}" for issue in issues
        )
    return {"final_report": report}
