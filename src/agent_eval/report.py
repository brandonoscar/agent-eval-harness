"""What a red gate says, and where it says it.

The headline is the line people read: the last line ``agent-eval run`` prints
and the first line of the job summary. :func:`gate_headline` leads with the
verdict and the worst case, and counts only what was judged as "run": a
skipped case or an outage is listed apart, never counted as checked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_eval.case import Case
from agent_eval.passk import PassKResult
from agent_eval.runner import SuiteResult

_MAX_ROWS = 25


def _ids(ids: list[str]) -> str:
    return f"{', '.join(ids[:3])}{'…' if len(ids) > 3 else ''}"


def gate_headline(suite: SuiteResult) -> str:
    if suite.passed:
        parts = [f"{len(suite.measured)} case(s) run"]
        if suite.skipped:
            parts.append(f"{len(suite.skipped)} skipped")
        outages = suite.by_kind("infrastructure")
        if outages:
            parts.append(f"{len(outages)} outage(s) not gating")
        if suite.quarantined:
            parts.append(f"{len(suite.quarantined)} quarantined quality miss(es)")
        return f"EVAL GATE PASSED: {', '.join(parts)}"
    parts = []
    for kind in ("safety", "infrastructure", "quality"):
        ids = [r.case_id for r in suite.by_kind(kind) if r.case_id in suite.gating]
        if ids:
            parts.append(f"{len(ids)} {kind} ({_ids(ids)})")
    if suite.skip_floor_breached:
        parts.append(f"{len(suite.skipped)} skipped, over the limit")
    return f"EVAL GATE FAILED: {'; '.join(parts)}"


def _not_gating(kind: str | None) -> str:
    if kind == "infrastructure":
        return "no (outages not gating)"
    return "no (quarantined)"


def summary_markdown(suite: SuiteResult) -> str:
    lines = [f"### {gate_headline(suite)}", ""]
    failed = [r for r in suite.results if not r.passed]
    if failed:
        lines += ["| Case | Kind | Gates | Why |", "|---|---|---|---|"]
        for r in failed[:_MAX_ROWS]:
            why = "; ".join(a.reason or a.name for a in r.failed_assertions) or (r.error or "")
            why = why.replace("|", "\\|").replace("\n", " ")[:300]
            gates = "yes" if r.case_id in suite.gating else _not_gating(r.failure_kind)
            lines.append(f"| `{r.case_id}` | {r.failure_kind} | {gates} | {why} |")
        if len(failed) > _MAX_ROWS:
            lines.append(f"\n…and {len(failed) - _MAX_ROWS} more.")
    if suite.parked:
        lines += ["", "Parked (never run): " + ", ".join(f"`{cid}`" for cid, _ in suite.parked)]
    return "\n".join(lines) + "\n"


def to_json(suite: SuiteResult) -> dict[str, Any]:
    return {
        "version": 1,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": suite.passed,
        "headline": gate_headline(suite),
        "gating": suite.gating,
        "quarantined": suite.quarantined,
        "skip_floor_breached": suite.skip_floor_breached,
        "parked": [{"case": cid, "reason": why} for cid, why in suite.parked],
        "results": [r.as_dict() for r in suite.results],
    }


def write_json(suite: SuiteResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_json(suite), indent=2) + "\n", encoding="utf-8")


# ── pass^k ─────────────────────────────────────────────────────────────────


def passk_passed(results: list[PassKResult]) -> bool:
    return all(r.safe for r in results)


def passk_headline(k: int, results: list[PassKResult]) -> str:
    """Safe means safe on every one of the k runs; unmeasured is not safe."""
    if passk_passed(results):
        return f"PASS^{k} PASSED: {len(results)} case(s) safe on {k} of {k} runs"
    unsafe = [r.case_id for r in results if r.safety_failures]
    unmeasured = [r.case_id for r in results if not r.safety_failures and not r.safe]
    parts = []
    if unsafe:
        parts.append(f"{len(unsafe)} unsafe ({_ids(unsafe)})")
    if unmeasured:
        parts.append(f"{len(unmeasured)} unmeasured ({_ids(unmeasured)})")
    return f"PASS^{k} FAILED: {'; '.join(parts)}"


def passk_summary_markdown(k: int, results: list[PassKResult], parked: list[Case]) -> str:
    lines = [f"### {passk_headline(k, results)}", ""]
    if results:
        lines += ["| Case | Result |", "|---|---|"]
        for r in results[:_MAX_ROWS]:
            summary = r.summary().replace("|", "\\|")
            lines.append(f"| `{r.case_id}` | {summary} |")
        if len(results) > _MAX_ROWS:
            lines.append(f"\n…and {len(results) - _MAX_ROWS} more.")
    if parked:
        lines += ["", "Parked (never run): " + ", ".join(f"`{c.id}`" for c in parked)]
    return "\n".join(lines) + "\n"


def passk_to_json(k: int, results: list[PassKResult], parked: list[Case]) -> dict[str, Any]:
    """pass^k results. Not a run file: ``agent-eval track`` refuses it."""
    return {
        "version": 1,
        "mode": "pass_k",
        "k": k,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passed": passk_passed(results),
        "headline": passk_headline(k, results),
        "parked": [{"case": c.id, "reason": c.parked} for c in parked],
        "cases": [
            {
                "case": r.case_id,
                "safe": r.safe,
                "summary": r.summary(),
                "runs": [run.as_dict() for run in r.runs],
            }
            for r in results
        ],
    }


__all__ = [
    "gate_headline",
    "passk_headline",
    "passk_passed",
    "passk_summary_markdown",
    "passk_to_json",
    "summary_markdown",
    "to_json",
    "write_json",
]
