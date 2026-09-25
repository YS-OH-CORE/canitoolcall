"""Committed results snapshots (results/YYYY-MM-DD/) are real runs; keep their triage honest.

For every snapshot with a ``triage.jsonl``:

* every results file loads and validates against the results schema;
* every failing case of every engine is assigned to exactly one finding;
* every fixture a finding lists really failed on that engine in the snapshot;
* referenced direct-repro scripts exist.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from canitoolcall.fixtures import repo_root
from canitoolcall.matrix import load_results
from canitoolcall.results import Status

ROOT = repo_root() or Path(__file__).resolve().parents[2]
SNAPSHOTS = sorted(p.parent for p in (ROOT / "results").glob("*/triage.jsonl"))
CLASSIFICATIONS = {
    "engine_bug",
    "engine_bug_known_upstream",
    "engine_bug_candidate",
    "truncation_policy",
    "strict_grammar_variant",
    "not_generated_by_model",
    "not_applicable_stop_applied",
    "intended_engine_behaviour",
    "uncertain_history_render",
    "contested_fixture",
    "untriaged_discrepancy",
}


@pytest.mark.skipif(not SNAPSHOTS, reason="no committed results snapshot with a triage.jsonl")
@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=[p.name for p in SNAPSHOTS])
def test_triage_matches_snapshot(snapshot: Path) -> None:
    runs = {r.engine.name: r for r in load_results(snapshot)}
    failing = {(name, c.fixture_id) for name, run in runs.items() for c in run.cases if c.status is Status.FAIL}
    findings = [json.loads(line) for line in (snapshot / "triage.jsonl").read_text(encoding="utf-8").splitlines()]
    assigned: collections.Counter[tuple[str, str]] = collections.Counter()
    for f in findings:
        assert f["classification"] in CLASSIFICATIONS, f["id"]
        assert f["engine"] in runs, f["id"]
        assert f["engine_version"] == runs[f["engine"]].engine.version, f["id"]
        assert f["fixtures"], f["id"]
        assert f["repro"].startswith(f"uv run canitoolcall run --engine {f['engine']} --id "), f["id"]
        if f.get("direct_repro"):
            assert (snapshot / f["direct_repro"]).is_file(), f["direct_repro"]
        for fid in f["fixtures"]:
            assert (f["engine"], fid) in failing, f"{f['id']}: {fid} did not fail on {f['engine']}"
            assigned[(f["engine"], fid)] += 1
    assert set(assigned) == failing, sorted(failing - set(assigned))[:10]
    assert max(assigned.values()) == 1, [k for k, v in assigned.items() if v > 1][:10]
