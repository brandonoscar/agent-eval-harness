"""The nightly trend signal: which cases are failing, flaky, unreached or
drifting, across several runs of the suite.

A single run answers "is this push safe to merge". It cannot see a case that
passes two nights in three, a model update that slowly changed how the agent
words its answers, or the tail of the suite a timed-out run never reached.
This module reads the JSON results of N runs (``agent-eval run --json``) and
reports, worst first:

* ``safety``: failed on safety in any run;
* ``failing``: failed (any kind) in any run;
* ``unreached``: in the corpus but in no run's results, or skipped or hit by
  an outage in every run. A case nobody judged is not a passing case; without
  this the report describes however much of the suite happened to run;
* ``flaky``: needed a quality re-roll in at least ``flake_share`` of its runs,
  trending toward a red gate on someone's unrelated PR;
* ``drift``: its replies moved away from the stored baseline wording
  (average similarity below ``drift_threshold``). Accept the new wording with
  ``--update-baseline`` once a person has read it;
* ``unmeasured``: replies with no baseline reply to compare against, so their
  drift could not be measured (one finding, listing the cases).

It gates nothing; it is for the issue a person reads in the morning.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FLAKE_SHARE = 0.4
DRIFT_THRESHOLD = 0.6
_SEVERITY = ("safety", "failing", "unreached", "flaky", "drift", "unmeasured")
#: The ``case`` of the finding that lists cases with no baseline reply.
BASELINE = "(baseline)"


@dataclass
class CaseStats:
    runs: int = 0
    fails: int = 0
    safety_fails: int = 0
    infra_fails: int = 0
    flakes: int = 0
    skips: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)

    @property
    def measured(self) -> int:
        """Runs that judged the case: not skipped, not an outage."""
        return self.runs - self.skips - self.infra_fails


@dataclass(frozen=True)
class Finding:
    case: str
    kind: str
    detail: str


def load_run(path: str | Path) -> dict[str, dict[str, Any]]:
    """One run's results, by case id. Raises ``ValueError`` for a file that is
    not ``agent-eval run --json`` output (a pass^k file, say): reading it as an
    empty run would report every case unreached for the wrong reason."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: can't read it ({exc})") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not all(
        isinstance(row, dict) and isinstance(row.get("case"), str) for row in results
    ):
        raise ValueError(f"{path}: not the results of `agent-eval run --json`")
    return {row["case"]: row for row in results}


def aggregate(runs: Iterable[dict[str, dict[str, Any]]]) -> dict[str, CaseStats]:
    stats: dict[str, CaseStats] = {}
    for run in runs:
        for case_id, row in run.items():
            s = stats.setdefault(case_id, CaseStats())
            s.runs += 1
            status = row.get("status")
            if status == "skip":
                s.skips += 1
                if row.get("skip_reason"):
                    s.skip_reasons.append(str(row["skip_reason"]))
                continue
            if status == "fail":
                s.fails += 1
                if row.get("kind") == "safety":
                    s.safety_fails += 1
                elif row.get("kind") == "infrastructure":
                    s.infra_fails += 1
            elif row.get("rerolls"):
                s.flakes += 1
            if row.get("reply"):
                s.replies.append(str(row["reply"]))
    return stats


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def similarity(a: str, b: str) -> float:
    """0.0 to 1.0 wording similarity (difflib ratio over normalized text)."""
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_findings(
    stats: dict[str, CaseStats],
    *,
    expected_ids: Iterable[str] = (),
    baseline: dict[str, str] | None = None,
    flake_share: float = FLAKE_SHARE,
    drift_threshold: float = DRIFT_THRESHOLD,
) -> list[Finding]:
    """Findings, worst first. ``baseline=None`` means drift isn't tracked; a
    baseline (even an empty one) means every case with replies is compared, and
    those with no baseline reply are reported as unmeasured."""
    findings: list[Finding] = []
    for case_id in sorted(set(expected_ids) - set(stats)):
        findings.append(
            Finding(case_id, "unreached", "in no run's results: unmeasured, not passing")
        )
    no_baseline: list[str] = []
    for case_id, s in sorted(stats.items()):
        if s.safety_fails:
            findings.append(
                Finding(case_id, "safety", f"safety failure in {s.safety_fails} of {s.runs} runs")
            )
        elif s.fails and s.fails > s.infra_fails:
            findings.append(Finding(case_id, "failing", f"failed {s.fails} of {s.runs} runs"))
        elif not s.measured:
            findings.append(Finding(case_id, "unreached", _never_judged(s)))
        judged = s.runs - s.skips
        if judged and s.flakes / judged >= flake_share:
            findings.append(
                Finding(
                    case_id,
                    "flaky",
                    f"needed a re-roll in {s.flakes} of {judged} runs "
                    f"(threshold {flake_share:.0%})",
                )
            )
        if baseline is None or not s.replies:
            continue
        reference = baseline.get(case_id)
        if not reference:
            no_baseline.append(case_id)
            continue
        mean = sum(similarity(reference, r) for r in s.replies) / len(s.replies)
        if mean < drift_threshold:
            latest = " ".join(s.replies[-1].split())[:160]
            findings.append(
                Finding(
                    case_id,
                    "drift",
                    f"wording similarity {mean:.2f} to the baseline "
                    f"(threshold {drift_threshold:.2f}); latest: {latest}",
                )
            )
    if no_baseline:
        shown = ", ".join(no_baseline[:10]) + (", ..." if len(no_baseline) > 10 else "")
        findings.append(
            Finding(
                BASELINE,
                "unmeasured",
                f"no baseline reply for {len(no_baseline)} case(s), so their wording drift "
                f"is unmeasured: {shown}. Read their replies, then store them with "
                "--update-baseline.",
            )
        )
    return sorted(findings, key=lambda f: (_SEVERITY.index(f.kind), f.case))


def _never_judged(s: CaseStats) -> str:
    parts = []
    if s.skips:
        reasons = sorted(set(s.skip_reasons))
        why = f" ({'; '.join(reasons)})" if reasons else ""
        parts.append(f"skipped in {'all ' if s.skips == s.runs else ''}{s.skips} runs{why}")
    if s.infra_fails:
        parts.append(f"an outage in {s.infra_fails} run(s)")
    return f"{', '.join(parts)}: never judged, so unmeasured, not passing"


def new_baseline(
    runs: list[dict[str, dict[str, Any]]], old: dict[str, str] | None = None
) -> dict[str, str]:
    """The latest passing reply per case, over the old baseline."""
    baseline = dict(old or {})
    for run in runs:
        for case_id, row in run.items():
            if row.get("status") == "pass" and row.get("reply"):
                baseline[case_id] = str(row["reply"])
    return dict(sorted(baseline.items()))


def render_markdown(
    findings: list[Finding],
    stats: dict[str, CaseStats],
    n_runs: int,
    *,
    drift_checked: bool = True,
) -> str:
    """The report. "No findings" is written only when something was measured;
    a report of nothing measured says so instead."""
    lines = [f"## Eval tracker: {n_runs} run(s), {len(stats)} case(s) measured", ""]
    if not findings:
        if not any(s.measured for s in stats.values()):
            lines.append("Nothing measured: no run holds a result that was judged.")
        elif drift_checked:
            lines.append("No findings: nothing failed, flaked, drifted or went unreached.")
        else:
            lines.append(
                "No findings: nothing failed, flaked or went unreached. "
                "Wording drift was not checked (no --baseline)."
            )
        return "\n".join(lines) + "\n"
    lines += ["| Case | Finding | Detail |", "|---|---|---|"]
    for f in findings:
        detail = f.detail.replace("|", "\\|")
        lines.append(f"| `{f.case}` | **{f.kind}** | {detail} |")
    if not drift_checked:
        lines += ["", "Wording drift was not checked (no --baseline)."]
    return "\n".join(lines) + "\n"


#: Finding kinds, worst first (``agent-eval track --fail-on`` takes these).
FINDING_KINDS = _SEVERITY

__all__ = [
    "BASELINE",
    "DRIFT_THRESHOLD",
    "FINDING_KINDS",
    "FLAKE_SHARE",
    "CaseStats",
    "Finding",
    "aggregate",
    "find_findings",
    "load_run",
    "new_baseline",
    "render_markdown",
    "similarity",
]
