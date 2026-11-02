"""Citation manager / source audit.

Runs after synthesis is complete (deep path) OR after the shallow researcher
(shallow path). Produces a CitationAudit containing:
  - deduplicated source registry (S1, S2, ...)
  - claim → source-id mappings
  - flagged issues (weak/unsupported/mismatched citations)

The writer is instructed to cite ONLY ids from this registry.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from deep_research.config import model_for
from deep_research.prompts import CITATION_AUDIT_SYSTEM
from deep_research.state import (
    Citation,
    CitationAudit,
    ResearchState,
    SourceRecord,
)


def _gather_citations(state: ResearchState) -> list[Citation]:
    """Collect every citation from either path (deep specialist notes or shallow)."""
    out: list[Citation] = []
    for note in state.get("specialist_notes", []) or []:
        out.extend(note.citations)
    out.extend(state.get("shallow_citations", []) or [])
    return out


def _dedupe_to_registry(cites: list[Citation]) -> list[SourceRecord]:
    """Group citations by URL, build a stable S1, S2, ... registry."""
    seen: dict[str, SourceRecord] = {}
    for i, c in enumerate(cites):
        if not c.url or c.url in seen:
            continue
        seen[c.url] = SourceRecord(id=f"S{len(seen) + 1}", url=c.url, title=c.title)
    return list(seen.values())


def _sanitize_json(raw: str) -> dict | None:
    """Try to extract a valid JSON object from a possibly malformed string."""
    if not raw:
        return None

    # Strip markdown fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: regex for a complete {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def citation_audit_node(state: ResearchState) -> dict[str, Any]:
    """Build the audited source registry that the writer will be told to obey."""
    cites = _gather_citations(state)
    if not cites:
        # Empty audit so the writer/finalizer can still proceed (e.g. meta route
        # ends up here in edge cases — we want a coherent path).
        return {
            "citation_audit": CitationAudit(flagged_issues=["no citations gathered"]),
            "warnings": ["citation_audit: no citations to audit"],
        }

    # Use verified sources from _last_search_results if available
    verified_sources = state.get("_last_search_results", [])
    verified_by_url: dict[str, dict[str, str]] = {}
    for src in verified_sources:
        if isinstance(src, dict) and "url" in src:
            verified_by_url[src["url"]] = {
                "title": src.get("title", ""),
                "snippet": src.get("snippet", ""),
            }

    # Build registry using verified titles/snippets where possible
    registry: list[SourceRecord] = []
    seen_urls: set[str] = set()
    for c in cites:
        if not c.url or c.url in seen_urls:
            continue
        seen_urls.add(c.url)
        # Prefer verified title/snippet
        info = verified_by_url.get(c.url, {})
        title = info.get("title", c.title)
        snippet = info.get("snippet", c.snippet)
        # If no title/snippet, mark as unknown/low
        if not title and not snippet:
            source_type = "unknown"
            quality = "low"
        else:
            source_type = "unknown"  # will be refined by model later
            quality = "medium"
        registry.append(
            SourceRecord(
                id=f"S{len(registry) + 1}",
                url=c.url,
                title=title,
                source_type=source_type,
                quality=quality,
            )
        )

    # Ask the model to classify source quality and (best-effort) map claims to
    # source ids. We pre-build the registry so the model never invents source ids.
    notes_blob = "\n\n".join(
        f"### {n.task_id} ({n.specialist_role})\n"
        + "\n".join(f"- {f}" for f in n.facts)
        + (f"\nINTERPRETATION: {n.interpretation}" if n.interpretation else "")
        for n in state.get("specialist_notes", []) or []
    )
    if not notes_blob:
        notes_blob = state.get("shallow_answer", "") or "(no notes)"

    # Build registry blob with snippets from shallow_citations
    shallow_citations = state.get("shallow_citations", []) or []
    snippet_by_url: dict[str, str] = {}
    for sc in shallow_citations:
        if sc.url and sc.snippet:
            snippet_by_url[sc.url] = sc.snippet

    registry_blob_lines: list[str] = []
    for s in registry:
        snippet = snippet_by_url.get(s.url, "")
        # Use full snippet (no truncation) to avoid false SNIPPET_MISMATCH
        snippet_display = snippet if snippet else "(no snippet)"
        line = f"- {s.id}: {s.url}"
        if s.title:
            line += f" | title: {s.title}"
        line += f" | snippet: \"{snippet_display}\""
        registry_blob_lines.append(line)
    registry_blob = "\n".join(registry_blob_lines)

    model = model_for("citation_audit")
    structured_model = model.with_structured_output(CitationAudit)

    user = (
        f"Research question:\n{state['question']}\n\n"
        f"Pre-built source registry (use these ids exactly, do not invent new ones):\n"
        f"{registry_blob}\n\n"
        f"Evidence / specialist notes:\n{notes_blob}\n\n"
        "Audit: classify each source, map significant claims to source ids, flag "
        "any weak/unsupported/mismatched citations. Return ONLY ids from the "
        "registry above."
    )

    messages = [SystemMessage(content=CITATION_AUDIT_SYSTEM), HumanMessage(content=user)]

    # Attempt 1: structured output
    try:
        audited: CitationAudit = structured_model.invoke(messages)
    except ValidationError as ve:
        # Structured output failed validation. Try to recover raw content.
        raw_content = ""
        try:
            # Some LangChain wrappers expose .content on the response object
            raw_content = getattr(ve, "content", "")
        except Exception:
            raw_content = ""

        if not raw_content:
            # Fallback: call model without structured output
            try:
                raw_msg = model.invoke(messages)
                raw_content = raw_msg.content
            except Exception:
                raw_content = ""

        # Sanitize JSON
        parsed = _sanitize_json(raw_content)
        if parsed is not None:
            try:
                audited = CitationAudit.model_validate(parsed)
            except ValidationError:
                audited = None
        else:
            audited = None

        if audited is None:
            # Retry once with stricter prompt
            strict_system = (
                CITATION_AUDIT_SYSTEM
                + "\n\nReturn ONLY valid JSON matching this schema. Do not wrap in markdown. "
                "Do not add comments. Start with '{' and end with '}'. Output nothing else."
            )
            strict_messages = [
                SystemMessage(content=strict_system),
                HumanMessage(content=user),
            ]
            try:
                # Try structured output again with stricter prompt
                audited = structured_model.invoke(strict_messages)
            except ValidationError:
                audited = None
            except Exception:
                audited = None

            if audited is None:
                # Final fallback: raw registry with parse failure
                return {
                    "citation_audit": CitationAudit(
                        sources=registry,
                        flagged_issues=[
                            "audit parse failed (2 attempts): "
                            + str(ve)
                        ],
                    ),
                    "warnings": [
                        "citation_audit: parse failed after retry; using raw registry only"
                    ],
                }

    except Exception as e:  # noqa: BLE001
        # Non-validation error (e.g. API failure)
        return {
            "citation_audit": CitationAudit(
                sources=registry,
                flagged_issues=[f"audit parse failed: {e}"],
            ),
            "warnings": [f"citation_audit: parse failed ({e}); using raw registry only"],
        }

    # Defensive merge: if the model dropped any registry entries, restore them.
    audited_urls = {s.url for s in audited.sources}
    for s in registry:
        if s.url not in audited_urls:
            audited.sources.append(s)

    # And remove any source ids that aren't in the final registry from claim mappings.
    valid_ids = {s.id for s in audited.sources}
    for cm in audited.claim_source_mappings:
        cm.source_ids = [sid for sid in cm.source_ids if sid in valid_ids]

    # DEFECT B: flag sources that are in the registry but have no claim mappings
    mapped_ids = set()
    for cm in audited.claim_source_mappings:
        mapped_ids.update(cm.source_ids)

    for s in audited.sources:
        if s.id not in mapped_ids:
            audited.flagged_issues.append(
                f"UNUSED_SOURCE: {s.id} ({s.url}) is in the registry but not mapped "
                "to any claim. Do not cite it in the final report unless a claim is added."
            )

    return {"citation_audit": audited}
