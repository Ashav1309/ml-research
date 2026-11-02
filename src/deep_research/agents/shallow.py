"""Shallow research path.

`shallow_researcher_node` runs a small ReAct loop with the Tavily tool and
produces a concise answer + citations. It self-judges sufficiency: if it could
not answer confidently, it sets `shallow_sufficient=False` so the graph escalates
to the deep path (scout → architect → ...).

`shallow_finalize_node` formats the audited shallow answer as the final report.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
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


def _collect_structured_search_results(messages: list[Any]) -> list[dict[str, str]]:
    """Extract URLs from ToolMessage outputs of web_search_structured.

    The tool returns a JSON list of dicts with url/title/snippet. We parse
    each ToolMessage whose name is 'web_search_structured' and collect all
    URLs. Returns a list of dicts (each with url/title/snippet) for later
    source-invariant checks.
    """
    collected: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "web_search_structured":
            try:
                data = json.loads(msg.content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "url" in item:
                            collected.append(
                                {
                                    "url": str(item.get("url", "")),
                                    "title": str(item.get("title", "")),
                                    "snippet": str(item.get("snippet", "")),
                                }
                            )
            except (json.JSONDecodeError, TypeError):
                # Ignore malformed tool output; the source invariant will
                # still flag any citation that doesn't match a known URL.
                continue
    return collected


def shallow_researcher_node(state: ResearchState) -> dict[str, Any]:
    """Run a small ReAct loop. Set shallow_sufficient flag for the router."""
    agent = _get_shallow_agent()
    result = agent.invoke({"messages": [HumanMessage(content=state["question"])]})
    answer = result["messages"][-1].content

    sufficient = "INSUFFICIENT" not in answer.upper()
    citations = _extract_citations_from_text(answer)

    # Collect structured search results from the agent's tool messages
    structured_results = _collect_structured_search_results(result["messages"])

    # Merge with any existing results (do not overwrite across iterations)
    existing = state.get("_last_search_results", [])
    merged = list(existing)  # shallow copy
    merged.extend(structured_results)

    # Deduplicate by URL while preserving order
    seen_urls = set()
    deduped = []
    for item in merged:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(item)

    # Save back to state
    state["_last_search_results"] = deduped

    # Enforce source invariant: every citation URL must appear in the
    # structured search results.
    search_urls = {c["url"] for c in deduped if isinstance(c, dict) and "url" in c}

    for c in citations:
        if c.url not in search_urls:
            state.setdefault("warnings", []).append(
                f"citation url not from web_search: {c.url}"
            )

    return {
        "shallow_answer": answer,
        "shallow_citations": citations,
        "shallow_sufficient": sufficient,
        "_last_search_results": deduped,
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
