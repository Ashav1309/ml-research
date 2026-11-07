#!/usr/bin/env python3
"""Audit MLflow traces against EBT invariants; compare trace pairs; run regression.

Usage:
    python scripts/evidence_ledger.py audit --db mlflow.db [--limit 50] [--json]
    python scripts/evidence_ledger.py compare TRACE_A TRACE_B --db mlflow.db
    python scripts/evidence_ledger.py run --query "..." --path shallow --baseline tests/baselines/transformer.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ALLOWED_VERDICTS = {
    "unverified", "single_source", "snippet_mismatch", "unsupported",
    "source_quality", "outdated_source", "non_canonical",
}


# ─── trace I/O ─────────────────────────────────────────────────────────
def fetch_trace(db_path: str, trace_id: str) -> dict:
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{Path(db_path).resolve()}"}
    r = subprocess.run(
        ["mlflow", "traces", "get", "--trace-id", trace_id],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mlflow traces get failed for {trace_id}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def list_trace_ids(db_path: str, limit: int) -> list[str]:
    import sqlite3
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT request_id FROM trace_info ORDER BY timestamp_ms DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def get_trace_state(trace: dict) -> dict:
    root = next((s for s in trace["data"]["spans"] if s["parent_span_id"] is None), None)
    if not root:
        return {}
    raw = root.get("attributes", {}).get("mlflow.spanOutputs")
    return json.loads(raw) if raw else {}


def get_spans(trace: dict, name: str) -> list[dict]:
    return [s for s in trace["data"]["spans"] if s["name"] == name]


def span_io(span: dict) -> tuple[dict | None, dict | None]:
    a = span.get("attributes", {})
    ins = json.loads(a["mlflow.spanInputs"]) if "mlflow.spanInputs" in a else None
    outs = json.loads(a["mlflow.spanOutputs"]) if "mlflow.spanOutputs" in a else None
    return ins, outs


# ─── invariant checks ─────────────────────────────────────────────────
def _normalize(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def check_verbatim_contract(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    answer = _normalize(state.get("shallow_answer") or "")
    if not answer:
        return v
    audit = state.get("citation_audit") or {}
    for i, fc in enumerate(audit.get("flagged_claims") or []):
        vs = fc.get("verbatim_sentence") or ""
        if vs and _normalize(vs) not in answer:
            v.append(f"flagged_claims[{i}].verbatim_sentence NOT a literal substring of shallow_answer: {vs[:80]!r}")
    return v


def check_cache_hit_consistency(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    hit = state.get("cache_hit") is True
    span_names = {s["name"] for s in trace["data"]["spans"]}
    if not hit:
        return v
    # checks that don't require instrumentation
    if "web_search_structured" in span_names:
        v.append("cache_hit=True but web_search_structured span present (should have been skipped)")
    if state.get("_last_search_results"):
        v.append(f"cache_hit=True but _last_search_results has {len(state['_last_search_results'])} entries")
    # instrumented checks
    lookups = get_spans(trace, "qdrant_lookup")
    if lookups:
        _, outs = span_io(lookups[0])
        n = (outs or {}).get("n_hits", 0)
        if n < 2:
            v.append(f"cache_hit=True but qdrant_lookup.n_hits={n} < 2")
    return v


def check_score_threshold(trace: dict) -> list[str]:
    v = []
    for s in get_spans(trace, "qdrant_lookup"):
        ins, outs = span_io(s)
        if not ins or not outs:
            continue
        min_score = ins.get("min_score", 0.0)
        for sc in outs.get("scores") or []:
            if sc < min_score:
                v.append(f"qdrant_lookup: returned score {sc} < min_score {min_score} (should be filtered)")
    return v


def check_store_consistency(trace: dict) -> list[str]:
    v = []
    for s in get_spans(trace, "qdrant_store"):
        ins, outs = span_io(s)
        if not ins or not outs:
            continue
        if "error" in outs:
            v.append(f"qdrant_store errored: {outs['error']}")
        if outs.get("n_points", 0) > ins.get("n_sources", 0):
            v.append(f"qdrant_store.n_points={outs['n_points']} > n_sources={ins['n_sources']}")
    return v


def check_citations_populated(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    for i, c in enumerate(state.get("shallow_citations") or []):
        if not c.get("url"):
            v.append(f"shallow_citations[{i}] missing url")
        if not c.get("title"):
            v.append(f"shallow_citations[{i}] missing title: {c.get('url')}")
        if not c.get("snippet"):
            v.append(f"shallow_citations[{i}] missing snippet: {c.get('url')}")
    return v


def check_report_unverified_count(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    report = state.get("final_report") or ""
    m = re.search(r"contains (\d+) claim\(s\) marked", report)
    if m:
        declared = int(m.group(1))
        body = report.split("\n", 1)[1] if "\n" in report else ""
        actual = body.count("[unverified]")
        if declared != actual:
            v.append(f"final_report header declares {declared} [unverified] but body has {actual}")
    return v


def check_verdicts_whitelist(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    audit = state.get("citation_audit") or {}
    for i, fc in enumerate(audit.get("flagged_claims") or []):
        verdict = fc.get("verdict")
        if verdict not in ALLOWED_VERDICTS:
            v.append(f"flagged_claims[{i}].verdict={verdict!r} not in allowed set")
    return v


def check_verification_warnings(trace: dict) -> list[str]:
    v = []
    state = get_trace_state(trace)
    vmeta = state.get("_verification_meta") or {}
    warnings_text = "\n".join(state.get("warnings") or [])
    for url, meta in vmeta.items():
        if (meta or {}).get("verification") == "error" and url not in warnings_text:
            v.append(f"source_verification: error for {url} but no corresponding warning")
    return v


CHECKS = [
    ("verbatim_contract", check_verbatim_contract),
    ("cache_hit_consistency", check_cache_hit_consistency),
    ("score_threshold", check_score_threshold),
    ("store_consistency", check_store_consistency),
    ("citations_populated", check_citations_populated),
    ("report_unverified_count", check_report_unverified_count),
    ("verdicts_whitelist", check_verdicts_whitelist),
    ("verification_warnings", check_verification_warnings),
]


def audit_trace(trace: dict) -> list[str]:
    violations = []
    for name, fn in CHECKS:
        try:
            for msg in fn(trace):
                violations.append(f"[{name}] {msg}")
        except Exception as e:
            violations.append(f"[{name}] CHECK CRASHED: {type(e).__name__}: {e}")
    return violations


# ─── commands ─────────────────────────────────────────────────────────
def cmd_audit(args) -> int:
    trace_ids = [args.trace_id] if args.trace_id else list_trace_ids(args.db, args.limit)
    with_violations = []
    for tid in trace_ids:
        try:
            trace = fetch_trace(args.db, tid)
        except Exception as e:
            with_violations.append({"trace_id": tid, "error": str(e)})
            continue
        vs = audit_trace(trace)
        if vs:
            with_violations.append({"trace_id": tid, "violations": vs})
    report = {
        "audited": len(trace_ids),
        "total_violations": sum(len(t.get("violations") or [t.get("error")]) for t in with_violations),
        "traces_with_violations": with_violations,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Audited {report['audited']} traces")
        print(f"Violations: {report['total_violations']}")
        for t in with_violations:
            print(f"\n  {t['trace_id']}")
            if "error" in t:
                print(f"    ERROR: {t['error']}")
            for msg in t.get("violations") or []:
                print(f"    {msg}")
    return 0 if report["total_violations"] == 0 else 1


def cmd_compare(args) -> int:
    a = get_trace_state(fetch_trace(args.db, args.trace_a))
    b = get_trace_state(fetch_trace(args.db, args.trace_b))
    ta = fetch_trace(args.db, args.trace_a)
    tb = fetch_trace(args.db, args.trace_b)

    def span_field(trace, span_name, field):
        spans = get_spans(trace, span_name)
        if not spans:
            return "<no span>"
        _, outs = span_io(spans[0])
        return (outs or {}).get(field, "<missing>")

    rows = {
        "cache_hit":                (a.get("cache_hit"), b.get("cache_hit")),
        "last_search_results_len":  (len(a.get("_last_search_results") or []), len(b.get("_last_search_results") or [])),
        "shallow_citations":        (len(a.get("shallow_citations") or []), len(b.get("shallow_citations") or [])),
        "flagged_claims":           (len((a.get("citation_audit") or {}).get("flagged_claims") or []),
                                     len((b.get("citation_audit") or {}).get("flagged_claims") or [])),
        "qdrant_lookup.n_hits":     (span_field(ta, "qdrant_lookup", "n_hits"), span_field(tb, "qdrant_lookup", "n_hits")),
        "qdrant_lookup.top_score":  (span_field(ta, "qdrant_lookup", "top_score"), span_field(tb, "qdrant_lookup", "top_score")),
        "qdrant_store.n_points":    (span_field(ta, "qdrant_store", "n_points"), span_field(tb, "qdrant_store", "n_points")),
    }
    print(f"{'metric':30s}  {'A':>20s}  {'B':>20s}  {'same?':>6s}")
    for k, (va, vb) in rows.items():
        print(f"{k:30s}  {str(va):>20s}  {str(vb):>20s}  {'yes' if va == vb else 'NO':>6s}")
    return 0


def cmd_run(args) -> int:
    # runs pipeline once on empty collection, then again — verifies miss→hit
    tracking = f"sqlite:///{Path(args.db).resolve()}"
    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking}

    print(">> clearing research_sources_v2")
    subprocess.run([
        sys.executable, "-c",
        "from qdrant_client import QdrantClient; "
        "QdrantClient(url='http://localhost:6333').delete_collection('research_sources_v2')",
    ], check=True)

    print(">> run 1 (cache miss)")
    subprocess.run(["deep-research", args.query, "--path", args.path, "--out", "/tmp/ledger_miss.md"],
                   check=True, env=env)

    print(">> run 2 (cache hit)")
    subprocess.run(["deep-research", args.query, "--path", args.path, "--out", "/tmp/ledger_hit.md"],
                   check=True, env=env)

    print(">> auditing both traces")
    miss_id = list_trace_ids(args.db, 2)[1]
    hit_id = list_trace_ids(args.db, 2)[0]
    miss_v = audit_trace(fetch_trace(args.db, miss_id))
    hit_v = audit_trace(fetch_trace(args.db, hit_id))

    print(f"miss {miss_id}: {'OK' if not miss_v else 'FAIL'}")
    for m in miss_v:
        print(f"  {m}")
    print(f"hit  {hit_id}: {'OK' if not hit_v else 'FAIL'}")
    for m in hit_v:
        print(f"  {m}")

    ok = not miss_v and not hit_v
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        hit_state = get_trace_state(fetch_trace(args.db, hit_id))
        hit_spans = fetch_trace(args.db, hit_id)
        lookups = get_spans(hit_spans, "qdrant_lookup")
        _, louts = span_io(lookups[0]) if lookups else (None, {})
        checks = [
            ("cache_hit", hit_state.get("cache_hit"), baseline["expect"]["cache_hit"]),
            ("n_hits >= min", (louts or {}).get("n_hits", 0), baseline["expect"]["min_n_hits"]),
            ("top_score >= min", (louts or {}).get("top_score", 0), baseline["expect"]["min_top_score"]),
        ]
        for name, got, want in checks:
            ok_run = got == want if name == "cache_hit" else got >= want
            print(f"  baseline {name}: got={got}, want={want} -> {'OK' if ok_run else 'FAIL'}")
            ok = ok and ok_run
    return 0 if ok else 1


def main() -> int:
    # Общие опции — определяются в отдельном парсере,
    # чтобы их можно было ставить и до, и после субкоманды.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default="mlflow.db")

    p = argparse.ArgumentParser(parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", parents=[common])
    a.add_argument("--trace-id")
    a.add_argument("--limit", type=int, default=50)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("compare", parents=[common])
    c.add_argument("trace_a")
    c.add_argument("trace_b")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("run", parents=[common])
    r.add_argument("--query", required=True)
    r.add_argument("--path", default="shallow", choices=["shallow", "deep"])
    r.add_argument("--baseline")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
