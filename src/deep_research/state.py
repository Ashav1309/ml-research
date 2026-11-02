"""Pydantic models and LangGraph state for the deep research graph.

Two state objects:
- `ResearchState`: top-level graph state, threaded through every node.
- `SpecialistState`: per-branch state passed to one specialist worker via Send().

Reducers
--------
Several fields use `_add_lists` as their reducer — a None-safe concatenator that
LangGraph applies when parallel branches return updates to the same key. This is
how the architect's fan-out and the gap analyzer's follow-up fan-out both append
to `specialist_notes` without overwriting each other.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, computed_field


# ─── Reducers ────────────────────────────────────────────────────────────────


def _add_lists(left: list | None, right: list | None) -> list:
    """None-safe list concatenation reducer used by accumulating state fields."""
    return (left or []) + (right or [])


# ─── Building blocks ─────────────────────────────────────────────────────────


class Citation(BaseModel):
    url: str
    title: str = ""
    snippet: str = Field("", description="Verbatim snippet supporting the claim")


# ─── 1. Intent classification ────────────────────────────────────────────────

Route = Literal[
    "meta_response",
    "shallow_research",
    "deep_research",
    "needs_clarification",
]
ResearchDepth = Literal["minimal", "shallow", "moderate", "deep"]


class IntentClassification(BaseModel):
    route: Route
    reasoning: str
    clarification_question: str | None = None
    expected_output_type: str = Field(
        description="e.g. 'factual_answer', 'report', 'comparison', 'analysis', 'capability_response'"
    )
    research_depth: ResearchDepth


# ─── 2. Scout (research landscape) ───────────────────────────────────────────


class ScoutFinding(BaseModel):
    url: str
    title: str = ""
    summary: str = ""


class ResearchLandscape(BaseModel):
    key_entities: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    timeline_notes: list[str] = Field(default_factory=list)
    controversies: list[str] = Field(default_factory=list)
    likely_source_types: list[str] = Field(default_factory=list)
    likely_gaps: list[str] = Field(default_factory=list)
    findings: list[ScoutFinding] = Field(default_factory=list)


# ─── 3. Architect plan ───────────────────────────────────────────────────────

SpecialistRole = Literal[
    "evidence_gatherer",
    "mechanism_explorer",
    "comparator",
    "skeptic",
    "horizon_scanner",
    "domain_specialist",
]


class ResearchTask(BaseModel):
    id: str = Field(description="Stable short id, e.g. 't1'")
    specialist_role: SpecialistRole
    objective: str
    required_evidence: list[str] = Field(default_factory=list)
    output_format: str = "Structured specialist notes with citations"


class ArchitectPlan(BaseModel):
    research_brief: str = Field(description="Crisp restatement of what is being researched")
    report_outline: list[str] = Field(
        default_factory=list,
        description="Ordered section titles for the final report",
    )
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    source_strategy: str = ""
    quality_constraints: list[str] = Field(default_factory=list)
    specialist_roles_required: list[SpecialistRole] = Field(default_factory=list)


# ─── 4. Specialist notes ─────────────────────────────────────────────────────


class SpecialistNote(BaseModel):
    """One specialist's findings for one task. Keeps facts, interpretation and
    uncertainty visibly separated so the synthesis step can preserve them."""

    task_id: str
    specialist_role: SpecialistRole
    facts: list[str] = Field(
        default_factory=list,
        description="Atomic, citation-tagged facts (e.g. 'X reached $1.2B in 2024 [https://...]')",
    )
    interpretation: str = ""
    uncertainty: list[str] = Field(
        default_factory=list,
        description="Open questions, conflicting evidence, or low-confidence claims",
    )
    citations: list[Citation] = Field(default_factory=list)
    raw_text: str | None = Field(
        default=None,
        description="Set when structured parsing failed; preserves the model's raw output",
    )


# ─── 5. Synthesis brief ──────────────────────────────────────────────────────


class SynthesisBrief(BaseModel):
    key_findings: list[str] = Field(default_factory=list)
    evidence_by_section: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps each report-outline section title to its supporting bullets (each bullet should reference source URLs)",
    )
    contradictions: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    source_list: list[str] = Field(
        default_factory=list,
        description="Deduplicated source URLs referenced by any finding",
    )


# ─── 6. Gap analysis ─────────────────────────────────────────────────────────


class FollowUpTask(BaseModel):
    gap: str = Field(description="The specific gap this follow-up addresses")
    specialist_role: SpecialistRole
    objective: str


class GapAnalysis(BaseModel):
    sufficient: bool
    gaps: list[str] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
    missing_source_types: list[str] = Field(default_factory=list)
    max_iterations_reached: bool = False


# ─── 7. Citation audit ───────────────────────────────────────────────────────


class SourceRecord(BaseModel):
    id: str = Field(description="Stable short id used in the report, e.g. 'S1'")
    url: str
    title: str = ""
    source_type: str = Field(
        "unknown",
        description="primary | official | secondary | blog | unknown",
    )
    quality: Literal["high", "medium", "low"] = "medium"


class ClaimSourceMapping(BaseModel):
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class FlaggedClaim(BaseModel):
    verbatim_sentence: str = Field(
        description=(
            "EXACT sentence from shallow_answer, copied character-for-"
            "character. Must be a literal substring of shallow_answer. "
            "Do not paraphrase, do not fix grammar, do not strip markdown."
        )
    )
    verdict: Literal[
        "unverified", "single_source", "snippet_mismatch",
        "unsupported", "source_quality"
    ] = Field(description="Type of issue.")
    reason: str = Field(
        description="Short explanation (1-2 sentences) of why this claim is flagged."
    )


class CitationAudit(BaseModel):
    sources: list[SourceRecord] = Field(default_factory=list)
    claim_source_mappings: list[ClaimSourceMapping] = Field(default_factory=list)
    flagged_claims: list[FlaggedClaim] = Field(default_factory=list)

    @computed_field
    @property
    def flagged_issues(self) -> list[str]:
        return [
            f"{c.verdict.upper()}: {c.reason} "
            f"(sentence: \"{c.verbatim_sentence[:80]}...\")"
            for c in self.flagged_claims
        ]


# ─── 8. Final report ────────────────────────────────────────────────────────


class RefinedReport(BaseModel):
    report_markdown: str
    changes_summary: str = ""


# ─── Top-level graph state ──────────────────────────────────────────────────


class ResearchState(TypedDict, total=False):
    """The shared state object threaded through the entire graph."""

    # Input
    question: str
    force_path: Literal["shallow", "deep"]  # set by CLI --path to bypass classifier

    # Classifier output (drives top-level routing)
    intent: IntentClassification

    # Shallow research path
    shallow_answer: str
    shallow_citations: Annotated[list[Citation], _add_lists]
    shallow_sufficient: bool  # if False, route escalates to deep path

    # Deep research path
    scout_findings: ResearchLandscape
    plan: ArchitectPlan
    specialist_notes: Annotated[list[SpecialistNote], _add_lists]
    synthesis: SynthesisBrief
    gap_analysis: GapAnalysis
    iterations: int  # planning iterations (initial + follow-up cycles)
    follow_up_tasks: Annotated[list[FollowUpTask], _add_lists]

    # Audit + final output
    citation_audit: CitationAudit
    draft_report: str   # writer output, pre-refinement
    final_report: str   # refiner output — this is what the CLI prints

    # Debugging — every node may append warnings without overwriting peers'
    warnings: Annotated[list[str], _add_lists]


class SpecialistState(TypedDict):
    """Per-branch state passed to one specialist worker via Send()."""

    question: str
    research_brief: str
    task: ResearchTask
