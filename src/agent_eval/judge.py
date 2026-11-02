"""LLM-as-judge, gated on proof that the judge agrees with people.

A judge's verdict may gate a merge only after it was checked against at least
``MIN_TRACES`` hand-labelled examples with a true-positive rate AND a
true-negative rate of at least ``MIN_RATE``, each measured on its own, so a
judge cannot pass by always giving the same answer. Until then its verdicts
are advisory: ``judge_safety`` failures are reported as quality, not safety.

The judge itself is yours: any async function ``(rubric, transcript) ->
JudgeVerdict``. This module validates it and records the result in a small
JSON report that ``agent-eval run --judge-report`` reads. That report is
trusted for its counts only: whether the judge gates is worked out again from
them, never read from a stored flag, and the report must name the judge it
measured, so one judge's report can't let another one gate.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_eval.transcript import Transcript

MIN_TRACES = 50
MIN_RATE = 0.90


@dataclass(frozen=True)
class JudgeVerdict:
    #: True when the transcript meets the rubric.
    passed: bool
    reason: str = ""


Judge = Callable[[str, Transcript], Awaitable[JudgeVerdict]]


@dataclass(frozen=True)
class LabeledTrace:
    """A hand-labelled example: does ``reply`` meet ``rubric``?"""

    id: str
    rubric: str
    reply: str
    label: bool


class JudgeReportError(ValueError):
    """A judge report or trace file that can't be used: unreadable, malformed,
    or written for a different judge."""


def load_labeled_traces(path: str | Path) -> list[LabeledTrace]:
    """JSON lines of ``{"id", "rubric", "reply", "label"}``. A row missing a
    field, or a label that isn't true or false, raises: a malformed label set
    fails loudly."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise JudgeReportError(f"{path}: can't read the traces ({exc})") from None
    traces = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            label = row["label"]
            if not isinstance(label, bool):
                raise TypeError("label must be true or false")
            traces.append(
                LabeledTrace(
                    id=str(row["id"]),
                    rubric=str(row["rubric"]),
                    reply=str(row["reply"]),
                    label=label,
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise JudgeReportError(f"{path}:{number}: not a labelled trace ({exc!r})") from None
    return traces


@dataclass(frozen=True)
class JudgeReport:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    #: Disagreements, so a person can read why the judge missed.
    misses: tuple[str, ...] = ()

    @property
    def tpr(self) -> float | None:
        positives = self.tp + self.fn
        return self.tp / positives if positives else None

    @property
    def tnr(self) -> float | None:
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else None

    @property
    def gating(self) -> bool:
        """Validated: enough traces, and both rates clear the bar. A class with
        no examples has no rate, and an unmeasured rate never clears it."""
        return (
            self.n >= MIN_TRACES
            and self.tpr is not None
            and self.tnr is not None
            and self.tpr >= MIN_RATE
            and self.tnr >= MIN_RATE
        )

    def verdict(self) -> str:
        if self.gating:
            return "gating"
        why = []
        if self.n < MIN_TRACES:
            why.append(f"{self.n} traces, need {MIN_TRACES}")
        for name, rate in (("TPR", self.tpr), ("TNR", self.tnr)):
            if rate is None:
                why.append(f"{name} unmeasured (no examples of that class)")
            elif rate < MIN_RATE:
                why.append(f"{name} {rate:.0%} < {MIN_RATE:.0%}")
        return "advisory: " + "; ".join(why)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(tpr=self.tpr, tnr=self.tnr, gating=self.gating, verdict=self.verdict())
        return out


async def validate_judge(traces: Iterable[LabeledTrace], judge: Judge) -> JudgeReport:
    tp = fp = tn = fn = 0
    misses = []
    n = 0
    for trace in traces:
        n += 1
        verdict = await judge(trace.rubric, Transcript(reply=trace.reply))
        if verdict.passed and trace.label:
            tp += 1
        elif verdict.passed and not trace.label:
            fp += 1
            misses.append(f"{trace.id}: judge passed a reply people failed")
        elif not verdict.passed and not trace.label:
            tn += 1
        else:
            fn += 1
            misses.append(f"{trace.id}: judge failed a reply people passed")
    return JudgeReport(n=n, tp=tp, fp=fp, tn=tn, fn=fn, misses=tuple(misses))


def read_report(path: str | Path, *, judge: str | None = None) -> bool:
    """Whether a saved report shows the judge may gate.

    The verdict is worked out again from the report's counts; a stored
    ``gating`` field is ignored, so editing it changes nothing. With ``judge``
    (the ``module:function`` about to be used), the report must name that same
    judge: a report for another judge raises :class:`JudgeReportError`.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JudgeReportError(f"{path}: can't read the judge report ({exc})") from None
    try:
        counts = {name: data[name] for name in ("n", "tp", "fp", "tn", "fn")}
    except (KeyError, TypeError):
        raise JudgeReportError(f"{path}: not a validate-judge report") from None
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()):
        raise JudgeReportError(
            f"{path}: not a validate-judge report (counts must be whole numbers)"
        )
    report = JudgeReport(**counts)
    if report.n != report.tp + report.fp + report.tn + report.fn:
        raise JudgeReportError(f"{path}: the counts don't add up to n={report.n}")
    if judge is not None:
        named = data.get("judge")
        if named is None:
            raise JudgeReportError(
                f"{path} names no judge, so it can't vouch for {judge!r}: "
                "run validate-judge --out again"
            )
        if named != judge:
            raise JudgeReportError(
                f"{path} validated {named!r}, not {judge!r}: validate this judge "
                "with validate-judge --out"
            )
    return report.gating


__all__ = [
    "MIN_RATE",
    "MIN_TRACES",
    "Judge",
    "JudgeReport",
    "JudgeReportError",
    "JudgeVerdict",
    "LabeledTrace",
    "load_labeled_traces",
    "read_report",
    "validate_judge",
]
