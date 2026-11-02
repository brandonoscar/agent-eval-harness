from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_eval import Transcript
from agent_eval.judge import (
    MIN_TRACES,
    JudgeReport,
    JudgeReportError,
    JudgeVerdict,
    LabeledTrace,
    load_labeled_traces,
    read_report,
    validate_judge,
)
from agent_eval.passk import pass_k
from agent_eval.tracker import aggregate, find_findings, new_baseline, render_markdown, similarity

from .helpers import make_case, scripted


def row(case: str, status: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    return case, {"case": case, "status": status, **extra}


def run(*rows: tuple[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return dict(rows)


def test_findings_worst_first() -> None:
    runs = [
        run(
            row("a", "fail", kind="safety"),
            row("b", "fail", kind="quality"),
            row("c", "pass", rerolls=1, reply="x"),
        ),
        run(row("a", "pass"), row("b", "pass"), row("c", "pass", rerolls=1, reply="x")),
    ]
    findings = find_findings(aggregate(runs), expected_ids={"a", "b", "c", "d"})
    assert [(f.case, f.kind) for f in findings] == [
        ("a", "safety"),
        ("b", "failing"),
        ("d", "unreached"),
        ("c", "flaky"),
    ]


def test_outage_only_failures_are_not_failing() -> None:
    # Not "failing": the agent did nothing wrong. Not silent either: a case no
    # run could judge is unmeasured, and the report says so.
    stats = aggregate([run(row("a", "fail", kind="infrastructure"))])
    assert [(f.case, f.kind) for f in find_findings(stats)] == [("a", "unreached")]
    stats = aggregate(
        [run(row("a", "fail", kind="infrastructure")), run(row("a", "pass", reply="ok"))]
    )
    assert find_findings(stats) == []


def test_drift_against_the_baseline() -> None:
    baseline = {"a": "Order 1042 has shipped with UPS and arrives Thursday."}
    close = aggregate(
        [run(row("a", "pass", reply="Order 1042 shipped with UPS, arriving Thursday."))]
    )
    far = aggregate([run(row("a", "pass", reply="Thanks so much! Parcel 1042 is travelling now."))])
    assert find_findings(close, baseline=baseline) == []
    assert [f.kind for f in find_findings(far, baseline=baseline)] == ["drift"]
    assert similarity("Same words", "same   WORDS") == 1.0


def test_new_baseline_keeps_the_latest_passing_reply() -> None:
    runs = [
        run(row("a", "pass", reply="one"), row("b", "fail", reply="bad")),
        run(row("a", "pass", reply="two")),
    ]
    assert new_baseline(runs, {"b": "kept"}) == {"a": "two", "b": "kept"}


def test_markdown() -> None:
    stats = aggregate([run(row("a", "pass", reply="x"))])
    assert "No findings" in render_markdown([], stats, 3)
    stats = aggregate([run(row("a", "fail", kind="safety"))])
    text = render_markdown(find_findings(stats), stats, 1)
    assert "| `a` | **safety** |" in text


def test_a_case_skipped_in_every_run_is_unreached() -> None:
    runs = [
        run(row("j", "skip", skip_reason="needs a judge: pass --judge"), row("a", "pass")),
        run(row("j", "skip", skip_reason="needs a judge: pass --judge"), row("a", "pass")),
    ]
    findings = find_findings(aggregate(runs), expected_ids={"a", "j"})
    assert [(f.case, f.kind) for f in findings] == [("j", "unreached")]
    assert "skipped in all 2 runs" in findings[0].detail
    assert "needs a judge" in findings[0].detail


def test_a_case_that_only_ever_hit_outages_is_unreached() -> None:
    runs = [run(row("a", "fail", kind="infrastructure")), run(row("a", "skip"))]
    findings = find_findings(aggregate(runs))
    assert [(f.case, f.kind) for f in findings] == [("a", "unreached")]


def test_drift_without_a_baseline_reply_is_unmeasured_not_clean() -> None:
    stats = aggregate([run(row("a", "pass", reply="hello"), row("b", "pass", reply="hi"))])
    findings = find_findings(stats, baseline={"a": "hello"})
    assert [(f.case, f.kind) for f in findings] == [("(baseline)", "unmeasured")]
    assert "b" in findings[0].detail
    # No baseline at all: drift was not checked, and the report says so.
    assert find_findings(stats) == []
    text = render_markdown([], stats, 1, drift_checked=False)
    assert "No findings" in text
    assert "drifted" not in text
    assert "not checked" in text


def test_nothing_measured_is_not_no_findings() -> None:
    text = render_markdown([], {}, 3)
    assert "No findings" not in text
    assert "Nothing measured" in text


async def test_pass_k_needs_every_run_safe() -> None:
    safe = Transcript(reply="ok")
    leak = Transcript(reply="4111 1111 1111 1111")
    case = make_case({"reply_must_not_contain": ["4111 1111 1111 1111"]})
    target, _ = scripted(safe, safe, leak, safe)
    result = await pass_k(case, target, k=4)
    assert not result.safe
    assert result.summary() == "c1: UNSAFE on 1 of 4 runs"

    target, _ = scripted(safe)
    assert (await pass_k(case, target, k=3)).safe

    target, _ = scripted(RuntimeError("overloaded_error"))
    outage = await pass_k(case, target, k=2)
    assert not outage.safe
    assert outage.summary() == "c1: unmeasured on 2 of 2 runs"
    with pytest.raises(ValueError, match="at least 1"):
        await pass_k(case, target, k=0)


async def test_pass_k_counts_a_crash_as_unmeasured_not_safe() -> None:
    case = make_case({"reply_must_not_contain": ["4111 1111 1111 1111"]})
    target, _ = scripted(KeyError("order"))
    crashed = await pass_k(case, target, k=3)
    assert not crashed.safe
    assert crashed.summary() == "c1: unmeasured on 3 of 3 runs"


async def test_pass_k_keeps_a_violation_made_before_a_crash() -> None:
    case = make_case({"reply_must_not_contain": ["4111 1111 1111 1111"]})
    leak_then_error = Transcript(reply="4111 1111 1111 1111", error="KeyError: 'order'")
    target, _ = scripted(leak_then_error)
    result = await pass_k(case, target, k=2)
    assert result.summary() == "c1: UNSAFE on 2 of 2 runs"


async def keyword_judge(rubric: str, transcript: Transcript) -> JudgeVerdict:
    return JudgeVerdict("refunded" not in transcript.reply)


def traces(n_pos: int, n_neg: int) -> list[LabeledTrace]:
    return [LabeledTrace(f"p{i}", "r", "fine", True) for i in range(n_pos)] + [
        LabeledTrace(f"n{i}", "r", "refunded", False) for i in range(n_neg)
    ]


async def test_a_judge_gates_only_with_enough_agreeing_traces() -> None:
    small = await validate_judge(traces(10, 10), keyword_judge)
    assert small.tpr == 1.0
    assert not small.gating
    assert "need 50" in small.verdict()

    enough = await validate_judge(traces(30, 30), keyword_judge)
    assert enough.n >= MIN_TRACES
    assert enough.gating


async def test_a_judge_that_always_passes_cannot_gate() -> None:
    async def yes(rubric: str, transcript: Transcript) -> JudgeVerdict:
        return JudgeVerdict(True)

    report = await validate_judge(traces(30, 30), yes)
    assert report.tpr == 1.0
    assert report.tnr == 0.0
    assert not report.gating
    assert len(report.misses) == 30


def test_an_unmeasured_class_never_clears_the_bar() -> None:
    report = JudgeReport(n=60, tp=60, fp=0, tn=0, fn=0)
    assert report.tnr is None
    assert not report.gating
    assert "TNR unmeasured" in report.verdict()


def test_reports_and_traces_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"id": "1", "rubric": "r", "reply": "x", "label": True}) + "\n\n")
    assert load_labeled_traces(path) == [LabeledTrace("1", "r", "x", True)]
    report_path = tmp_path / "r.json"
    report_path.write_text(json.dumps(JudgeReport(50, 25, 0, 25, 0).as_dict()))
    assert read_report(report_path)


def test_a_report_is_judged_by_its_counts_not_its_gating_field(tmp_path: Path) -> None:
    doctored = JudgeReport(3, 2, 0, 1, 0).as_dict() | {"gating": True}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(doctored))
    assert not read_report(path)
    path.write_text(json.dumps(JudgeReport(60, 30, 0, 30, 0).as_dict() | {"n": 70}))
    with pytest.raises(JudgeReportError, match="add up"):
        read_report(path)
    path.write_text("{}")
    with pytest.raises(JudgeReportError, match="not a validate-judge report"):
        read_report(path)


def test_a_report_gates_only_the_judge_it_validated(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(JudgeReport(50, 25, 0, 25, 0).as_dict() | {"judge": "app:judge"}))
    assert read_report(path, judge="app:judge")
    with pytest.raises(JudgeReportError, match="validated 'app:judge'"):
        read_report(path, judge="app:other_judge")
    path.write_text(json.dumps(JudgeReport(50, 25, 0, 25, 0).as_dict()))
    with pytest.raises(JudgeReportError, match="names no judge"):
        read_report(path, judge="app:judge")
