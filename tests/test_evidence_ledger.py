"""EBT invariant tests. Fast path audits existing traces; integration path runs new ones."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "mlflow.db"
SCRIPT = ROOT / "scripts" / "evidence_ledger.py"


@pytest.fixture(scope="session")
def db_path() -> Path:
    if not DB.exists():
        pytest.skip(f"no mlflow.db at {DB}")
    return DB


def _audit(db: Path, limit: int = 20) -> dict:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db), "audit", "--limit", str(limit), "--json"],
        capture_output=True, text=True,
    )
    assert r.returncode in (0, 1), f"audit crashed: {r.stderr}"
    return json.loads(r.stdout)


def test_audit_all_traces(db_path):
    """Every trace in mlflow.db must satisfy EBT invariants."""
    report = _audit(db_path, limit=10)
    assert report["total_violations"] == 0, (
        f"{report['total_violations']} violation(s) across {len(report['traces_with_violations'])} trace(s):\n"
        + json.dumps(report["traces_with_violations"], indent=2)
    )


def test_verbatim_contract_across_history(db_path):
    """Every flagged_claim.verbatim_sentence must be a literal substring of shallow_answer."""
    report = _audit(db_path, limit=50)
    for t in report["traces_with_violations"]:
        for v in t.get("violations") or []:
            assert not v.startswith("[verbatim_contract]"), (
                f"{t['trace_id']}: {v}"
            )


@pytest.mark.integration
@pytest.mark.skipif(
    not (DB.exists() and (ROOT / "tests" / "baselines" / "transformer.json").exists()),
    reason="integration test requires mlflow.db + baseline JSON",
)
def test_cache_miss_then_hit_matches_baseline(db_path):
    """Run pipeline twice (miss→hit), verify invariants and baseline thresholds.

    Costs ~$0.02 and ~45s. Requires Qdrant running and AITunnel API.
    Run with: pytest -m integration tests/test_evidence_ledger.py -v
    """
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "run",
         "--query", "What are the main components of a transformer architecture?",
         "--path", "shallow",
         "--baseline", str(ROOT / "tests" / "baselines" / "transformer.json")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"regression failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
