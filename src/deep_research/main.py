"""CLI entry: `python -m deep_research.main "your question"` or `deep-research "..."`.

Streams progress as the graph runs and prints the `final_report` field on
completion. Supports early exit:

  --max-time SEC   soft wall-clock budget (checked between nodes)
  Ctrl-C           graceful interrupt at any time

On early exit, whatever evidence has been collected is fed through the rest of
the pipeline (audit → writer → refiner) so you still get a report.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import mlflow
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from deep_research.finalize import finalize_from_partial
from deep_research.graph import build_graph
from deep_research.state import IntentClassification

console = Console()

# State fields whose reducer is _add_lists in state.py — we mirror that here
# while accumulating updates manually as we stream.
ACCUMULATING_FIELDS = {
    "specialist_notes",
    "shallow_citations",
    "follow_up_tasks",
    "warnings",
}


def _merge_update(state: dict[str, Any], update: dict[str, Any] | None) -> None:
    """Mirror the graph's reducer behavior for streaming-mode 'updates'."""
    if not update:
        return
    for key, value in update.items():
        if key in ACCUMULATING_FIELDS:
            state[key] = (state.get(key) or []) + (value or [])
        else:
            state[key] = value


def _format_progress(node_name: str, update: dict | None) -> str | None:
    """One-line progress message for interesting updates, or None to skip.

    Update may be None when a node short-circuits with `return {}` (e.g. the
    classifier when --path preloads intent)."""
    if not update:
        return None
    if node_name == "classifier" and "intent" in update:
        intent = update["intent"]
        return f"intent: [bold]{intent.route}[/] (depth={intent.research_depth})"
    if node_name == "scout" and "scout_findings" in update:
        sf = update["scout_findings"]
        return f"scout: {len(sf.findings)} findings, {len(sf.key_entities)} entities"
    if node_name == "architect" and "plan" in update:
        plan = update["plan"]
        return (
            f"plan: {len(plan.research_tasks)} tasks across "
            f"{len(plan.specialist_roles_required)} roles"
        )
    if node_name == "specialist" and "specialist_notes" in update:
        notes = update["specialist_notes"]
        if notes:
            n = notes[0]
            return (
                f"specialist[{n.task_id}/{n.specialist_role}]: "
                f"{len(n.facts)} facts, {len(n.citations)} citations"
            )
    if node_name == "synthesis" and "synthesis" in update:
        b = update["synthesis"]
        return f"synthesis: {len(b.key_findings)} findings, {len(b.source_list)} sources"
    if node_name == "gap_analyzer" and "gap_analysis" in update:
        g = update["gap_analysis"]
        verdict = (
            "sufficient"
            if g.sufficient
            else f"insufficient ({len(g.follow_up_tasks)} follow-ups)"
        )
        return f"gap_analyzer: {verdict}"
    if node_name == "citation_audit" and "citation_audit" in update:
        a = update["citation_audit"]
        return f"citation_audit: {len(a.sources)} sources, {len(a.flagged_issues)} flags"
    return None


def cli() -> None:
    load_dotenv()  # picks up OPENAI_API_KEY / ANTHROPIC_API_KEY, TAVILY_API_KEY

    parser = argparse.ArgumentParser(description="Multi-agent deep research")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--out", help="Write report to this file (default: stdout)")
    parser.add_argument(
        "--max-time",
        type=int,
        metavar="SEC",
        help="Wall-clock budget in seconds. Checked between nodes; on exceed, "
             "the partial state is finalized into a best-effort report.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        metavar="N",
        help="Cap the architect→specialist→synthesis→gap loop iterations "
             "(default 2; overrides DEEP_RESEARCH_MAX_ITER for this run).",
    )
    parser.add_argument(
        "--path",
        choices=["shallow", "deep"],
        help="Force a research path, bypassing the classifier. "
             "shallow = quick search-only answer (no escalation); "
             "deep = full scout→architect→specialist→…→writer pipeline.",
    )
    args = parser.parse_args()

    # --max-iter must be set BEFORE build_graph because MAX_DEEP_ITERATIONS is
    # read at import time in config.py.
    if args.max_iter is not None:
        os.environ["DEEP_RESEARCH_MAX_ITER"] = str(args.max_iter)
        # Refresh the cached value so the routing function picks it up.
        from deep_research import config as _cfg
        _cfg.MAX_DEEP_ITERATIONS = args.max_iter

    graph = build_graph()
    state: dict[str, Any] = {"question": args.question}

    # --path forces a route by pre-populating intent + force_path. The classifier
    # node short-circuits when it sees an intent already in state, and the
    # shallow-path router honors force_path to suppress auto-escalation.
    if args.path == "shallow":
        state["force_path"] = "shallow"
        state["intent"] = IntentClassification(
            route="shallow_research",
            reasoning="forced via --path shallow",
            expected_output_type="factual_answer",
            research_depth="shallow",
        )
    elif args.path == "deep":
        state["force_path"] = "deep"
        state["intent"] = IntentClassification(
            route="deep_research",
            reasoning="forced via --path deep",
            expected_output_type="report",
            research_depth="deep",
        )

    console.print(f"[bold cyan]Researching:[/] {args.question}")
    if args.path:
        console.print(f"[dim]Forced path: {args.path} (classifier bypassed)[/]")
    if args.max_time:
        console.print(f"[dim]Time budget: {args.max_time}s. Ctrl-C to finalize early.[/]")
    if args.max_iter is not None:
        console.print(f"[dim]Max iterations: {args.max_iter}[/]")
    console.print()

    mlflow.set_experiment("deep-research")
    mlflow.langchain.autolog()

    with mlflow.start_run(run_name=f"deep-research:{(args.question or '')[:60]}"):
        mlflow.log_param("question", args.question or "")
        mlflow.log_param("path", args.path or "auto")
        mlflow.log_param("scout_model", os.getenv("SCOUT_MODEL", ""))
        mlflow.log_param("refiner_model", os.getenv("REFINER_MODEL", ""))
        mlflow.log_param("academic_domains", os.getenv("ACADEMIC_DOMAINS", "default"))

        start = time.monotonic()
        early_exit: str | None = None

        try:
            for event in graph.stream(dict(state), stream_mode="updates"):
                for node_name, update in event.items():
                    _merge_update(state, update)
                    msg = _format_progress(node_name, update)
                    console.print(
                        f"[dim]✓ {node_name}[/]" + (f" — {msg}" if msg else "")
                    )

                    if args.max_time and (time.monotonic() - start) > args.max_time:
                        early_exit = f"time budget ({args.max_time}s) exceeded"
                        break
                if early_exit:
                    break
        except KeyboardInterrupt:
            early_exit = "interrupted (Ctrl-C)"

        if early_exit:
            console.print(
                f"\n[yellow]Early exit:[/] {early_exit}. "
                "Finalizing from partial state...\n"
            )
            try:
                report = finalize_from_partial(state)
            except KeyboardInterrupt:
                # Second Ctrl-C during finalize — bail with what's in state.
                console.print("[red]Finalize also interrupted; dumping raw state notes.[/]")
                report = state.get("draft_report") or state.get("shallow_answer") or "(no report)"
        else:
            report = state.get("final_report", "(no report produced)")

        elapsed = time.monotonic() - start
        console.print(f"\n[dim]Total time: {elapsed:.1f}s[/]")

        if args.out:
            with open(args.out, "w") as f:
                f.write(report)
            console.print(f"\n[green]Report saved to {args.out}[/]")
        else:
            console.print("\n[bold green]── Report ──[/]\n")
            console.print(Markdown(report))

        mlflow.log_text(report, "report.md")


if __name__ == "__main__":
    cli()
