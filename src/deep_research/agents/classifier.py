"""Classifier / meta / clarifier nodes.

These three nodes share a file because they are all top-of-graph: the classifier
decides the route, and meta + clarifier are leaf nodes for routes that don't
need research.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.config import model_for
from deep_research.prompts import CLARIFIER_SYSTEM, CLASSIFIER_SYSTEM, META_SYSTEM
from deep_research.state import IntentClassification, ResearchState


def classifier_node(state: ResearchState) -> dict[str, Any]:
    """Decide whether this question is meta / shallow / deep / needs clarification.

    Short-circuits (no LLM call) if `intent` is already in state — this is how
    `--path shallow|deep` bypasses the classifier."""
    # Early return when force_path is set (CLI --path)
    fp = state.get("force_path")
    if fp:
        return {
            "intent": IntentClassification(
                route=f"{fp}_research",
                reasoning=f"forced via --path {fp}",
                clarification_question=None,
                expected_output_type="factual_answer",
                research_depth=fp,
            )
        }

    if state.get("intent"):
        return {}
    model = model_for("classifier").with_structured_output(IntentClassification)
    intent: IntentClassification = model.invoke(
        [
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=f"User question:\n{state['question']}"),
        ]
    )
    return {"intent": intent}


def meta_node(state: ResearchState) -> dict[str, Any]:
    """Answer a meta question about the system (no research)."""
    model = model_for("meta", temperature=0.2)
    msg = model.invoke(
        [
            SystemMessage(content=META_SYSTEM),
            HumanMessage(content=state["question"]),
        ]
    )
    # Final output lives in `final_report` so the CLI prints it uniformly.
    return {"final_report": msg.content}


def clarifier_node(state: ResearchState) -> dict[str, Any]:
    """Ask the user a clarifying question instead of researching blindly."""
    model = model_for("clarifier", temperature=0.2)
    intent = state.get("intent")
    seed = (
        intent.clarification_question
        if intent and intent.clarification_question
        else "(no specific clarification was suggested)"
    )
    msg = model.invoke(
        [
            SystemMessage(content=CLARIFIER_SYSTEM),
            HumanMessage(
                content=(
                    f"User question:\n{state['question']}\n\n"
                    f"Classifier's suggested clarification:\n{seed}"
                )
            ),
        ]
    )
    return {"final_report": msg.content}
