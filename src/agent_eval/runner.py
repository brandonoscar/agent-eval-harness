"""Run cases against your agent and decide whether the merge may proceed.

Your target is an async function that runs the agent for one case and
returns a :class:`~agent_eval.transcript.Transcript`:

    async def target(case: Case) -> Transcript: ...

The rules, each one there because the opposite went wrong:

* **Safety failures are never re-rolled.** A forbidden action that happens one
  run in three is a forbidden action.
* **Quality failures are re-rolled** (``quality_retries``, default 2) and the
  first pass wins: model wording varies run to run and one miss should not
  block an unrelated change.
* **Model-driven cases are quarantined from quality gating by default.**
  Their wording misses print and are counted by the nightly tracker, but only
  their safety failures gate. A crash still gates: it is not wording.
* **Transient outages are retried** with backoff; terminal ones (spend cap,
  bad key) are not, because retrying them is guaranteed waste. An outage
  still fails the gate by default: a run that did not happen is not evidence
  of safety.
* **Too many skips fail the gate.** A suite that skipped its cases for a
  missing key would otherwise go green having checked nothing.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_eval.assertions import AssertionResult, check
from agent_eval.case import Case
from agent_eval.judge import Judge
from agent_eval.result import DEFAULT_MARKERS, CaseResult, OutageMarkers
from agent_eval.transcript import Transcript

Target = Callable[[Case], Awaitable[Transcript]]


@dataclass(frozen=True)
class RunPolicy:
    quality_retries: int = 2
    infra_retries: int = 2
    infra_backoff_s: tuple[float, ...] = (2.0, 6.0)
    #: Re-roll quarantined cases too. Off by default: their quality failures
    #: do not gate, so a re-roll only spends model calls. Turn it on for the
    #: nightly tracker, which counts re-rolls as flakes.
    reroll_quarantined: bool = False
    #: Fail the gate when more than this many cases skip.
    max_skips: int = 3
    #: An outage fails the gate.
    infra_gates: bool = True
    markers: OutageMarkers = DEFAULT_MARKERS
    #: Cases run at once.
    concurrency: int = 4


@dataclass
class SuiteResult:
    results: list[CaseResult] = field(default_factory=list)
    parked: list[tuple[str, str]] = field(default_factory=list)
    #: Case ids whose failure blocks the merge.
    gating: list[str] = field(default_factory=list)
    #: Quarantined quality failures: printed and counted, not gating.
    quarantined: list[str] = field(default_factory=list)
    skip_floor_breached: bool = False

    @property
    def passed(self) -> bool:
        return not self.gating and not self.skip_floor_breached

    def by_kind(self, kind: str) -> list[CaseResult]:
        return [r for r in self.results if r.failure_kind == kind]

    @property
    def skipped(self) -> list[CaseResult]:
        return [r for r in self.results if r.skipped]

    @property
    def measured(self) -> list[CaseResult]:
        """Cases that ran and were judged: not skipped, not an outage."""
        return [r for r in self.results if not r.skipped and r.failure_kind != "infrastructure"]


def _missing_env(case: Case) -> list[str]:
    return [name for name in case.requires_env if not os.environ.get(name)]


async def run_case(
    case: Case,
    target: Target,
    *,
    judge: Judge | None = None,
    judge_gates: bool = False,
    markers: OutageMarkers = DEFAULT_MARKERS,
) -> CaseResult:
    """One attempt at one case."""
    missing = _missing_env(case)
    if missing:
        return CaseResult(
            case.id, skipped=True, skip_reason=f"missing env {missing}", markers=markers
        )
    if case.uses_judge and judge is None:
        return CaseResult(
            case.id, skipped=True, skip_reason="needs a judge: pass --judge", markers=markers
        )

    started = time.monotonic()
    try:
        transcript = await target(case)
    except Exception as exc:
        return CaseResult(
            case.id,
            error=f"{type(exc).__name__}: {exc}"[:1000],
            latency_ms=round((time.monotonic() - started) * 1000, 1),
            markers=markers,
        )
    latency = round((time.monotonic() - started) * 1000, 1)

    try:
        assertions = check(transcript, case.expected, is_outage=markers.names_outage)
    except Exception as exc:  # a malformed assertion value: the case is wrong
        return CaseResult(case.id, error=f"bad assertion: {exc}", markers=markers)
    error = transcript.error
    if judge is not None:
        try:
            assertions.extend(await _judge(case, transcript, judge, judge_gates))
        except Exception as exc:
            # The judge's own provider failing (a 429, an overload) is an
            # outage of this run; a judge bug is an error that gates. Either
            # way the assertions already checked still count.
            judge_error = f"judge error: {type(exc).__name__}: {exc}"[:1000]
            error = f"{error}; {judge_error}" if error else judge_error
    return CaseResult(
        case.id,
        assertions=assertions,
        error=error,
        warnings=list(transcript.warnings),
        reply=transcript.reply,
        usage=dict(transcript.usage),
        latency_ms=latency,
        markers=markers,
    )


async def _judge(
    case: Case, transcript: Transcript, judge: Judge, judge_gates: bool
) -> list[AssertionResult]:
    out = []
    for key in ("judge", "judge_safety"):
        rubric = case.expected.get(key)
        if rubric is None:
            continue
        verdict = await judge(str(rubric), transcript)
        # An unvalidated judge is advisory: its "unsafe" is a quality signal.
        critical = key == "judge_safety" and judge_gates
        reason = None
        if not verdict.passed:
            reason = verdict.reason or "the judge failed the reply"
            if key == "judge_safety" and not judge_gates:
                reason += " (advisory: the judge is not validated to gate)"
        out.append(AssertionResult(f"{key}", verdict.passed, reason, critical=critical))
    return out


async def run_case_with_retries(
    case: Case,
    target: Target,
    policy: RunPolicy | None = None,
    *,
    judge: Judge | None = None,
    judge_gates: bool = False,
) -> CaseResult:
    """A case under the retry rules: transient outages retried with backoff
    (on a quality re-roll too), quality failures re-rolled, safety failures and
    skips returned as they are."""
    policy = policy or RunPolicy()
    infra_retries = 0

    async def attempt() -> CaseResult:
        nonlocal infra_retries
        result = await run_case(
            case, target, judge=judge, judge_gates=judge_gates, markers=policy.markers
        )
        while result.retryable_infrastructure and infra_retries < policy.infra_retries:
            backoff = policy.infra_backoff_s
            await asyncio.sleep(backoff[min(infra_retries, len(backoff) - 1)] if backoff else 0)
            result = await run_case(
                case, target, judge=judge, judge_gates=judge_gates, markers=policy.markers
            )
            infra_retries += 1
        return result

    result = await attempt()
    rerolls = 0
    if policy.reroll_quarantined or not case.quarantined:
        while (
            not result.passed
            and result.failure_kind == "quality"
            and rerolls < policy.quality_retries
        ):
            result = await attempt()
            rerolls += 1
    result.rerolls = rerolls
    result.infra_retries = infra_retries
    return result


def _gates(case: Case, result: CaseResult, policy: RunPolicy) -> bool:
    kind = result.failure_kind
    if kind is None:
        return False
    if kind == "safety":
        return True
    if kind == "infrastructure":
        return policy.infra_gates
    # Quality: a quarantined case's wording miss does not gate; a crash does.
    return not (case.quarantined and result.error is None)


def line_for(result: CaseResult) -> str:
    """The one-line record the tracker parses. Keep the format stable."""
    if result.skipped:
        return f"[SKIP] {result.case_id} ({result.skip_reason})"
    if result.passed:
        if result.rerolls:
            return f"[PASS] {result.case_id} (after {result.rerolls} re-roll(s): quality flake)"
        return f"[PASS] {result.case_id}"
    reasons = "; ".join(f"{a.name}: {a.reason}" for a in result.failed_assertions if a.reason) or (
        result.error or "failed"
    )
    return f"[FAIL] {result.case_id} (kind={result.failure_kind}) {reasons}"


async def run_suite(
    cases: Sequence[Case],
    target: Target,
    policy: RunPolicy | None = None,
    *,
    judge: Judge | None = None,
    judge_gates: bool = False,
    on_result: Callable[[str], Any] | None = print,
) -> SuiteResult:
    """Run every case (parked ones are listed, never run) and decide the gate."""
    policy = policy or RunPolicy()
    suite = SuiteResult()
    runnable = []
    for case in cases:
        if case.parked:
            suite.parked.append((case.id, str(case.parked)))
            if on_result is not None:
                on_result(f"[PARKED] {case.id} ({case.parked})")
        else:
            runnable.append(case)

    semaphore = asyncio.Semaphore(max(1, policy.concurrency))

    async def one(case: Case) -> CaseResult:
        async with semaphore:
            try:
                result = await run_case_with_retries(
                    case, target, policy, judge=judge, judge_gates=judge_gates
                )
            except Exception as exc:
                # Whatever broke, it broke this case: the rest of the suite
                # still runs and still reports.
                result = CaseResult(
                    case.id,
                    error=f"harness error: {type(exc).__name__}: {exc}"[:1000],
                    markers=policy.markers,
                )
        if on_result is not None:
            on_result(line_for(result))
        return result

    results = await asyncio.gather(*(one(c) for c in runnable))
    for case, result in zip(runnable, results, strict=True):
        suite.results.append(result)
        if _gates(case, result, policy):
            suite.gating.append(case.id)
        elif result.failure_kind == "quality":
            suite.quarantined.append(case.id)
    suite.skip_floor_breached = sum(r.skipped for r in results) > policy.max_skips
    return suite


__all__ = [
    "RunPolicy",
    "SuiteResult",
    "Target",
    "line_for",
    "run_case",
    "run_case_with_retries",
    "run_suite",
]
