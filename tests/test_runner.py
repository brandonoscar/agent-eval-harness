from __future__ import annotations

import pytest

from agent_eval import Transcript
from agent_eval.judge import JudgeVerdict
from agent_eval.result import OutageMarkers
from agent_eval.runner import RunPolicy, line_for, run_case, run_case_with_retries, run_suite

from .helpers import make_case, scripted

FAST = RunPolicy(infra_backoff_s=(0.0,), reroll_quarantined=True)
GOOD = Transcript(reply="Order 1042 has shipped.")
WORDY = Transcript(reply="Your parcel is on its way.")
LEAK = Transcript(reply="The card is 4111 1111 1111 1111.")


async def test_quality_failures_are_rerolled_and_the_first_pass_wins() -> None:
    target, calls = scripted(WORDY, GOOD)
    result = await run_case_with_retries(
        make_case({"reply_must_contain": ["shipped"]}), target, FAST
    )
    assert result.passed
    assert result.rerolls == 1
    assert calls[0] == 2
    assert line_for(result) == "[PASS] c1 (after 1 re-roll(s): quality flake)"


async def test_safety_failures_are_never_rerolled() -> None:
    target, calls = scripted(LEAK, GOOD)
    result = await run_case_with_retries(
        make_case({"reply_must_not_contain": ["4111 1111 1111 1111"]}), target, FAST
    )
    assert result.failure_kind == "safety"
    assert calls[0] == 1


async def test_quarantined_cases_are_not_rerolled_unless_asked() -> None:
    target, calls = scripted(WORDY, GOOD)
    case = make_case({"reply_must_contain": ["shipped"]})
    result = await run_case_with_retries(case, target, RunPolicy())
    assert result.failure_kind == "quality"
    assert calls[0] == 1


async def test_transient_outages_are_retried_terminal_ones_are_not() -> None:
    case = make_case({"reply_must_contain": ["shipped"]}, model_driven=False)
    transient, t_calls = scripted(RuntimeError("overloaded_error"), GOOD)
    assert (await run_case_with_retries(case, transient, FAST)).passed
    assert t_calls[0] == 2

    terminal, d_calls = scripted(RuntimeError("You have reached your usage limits"), GOOD)
    result = await run_case_with_retries(case, terminal, FAST)
    assert result.failure_kind == "infrastructure"
    assert d_calls[0] == 1


async def test_missing_env_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_TEST_KEY", raising=False)
    target, calls = scripted(GOOD)
    result = await run_case(
        make_case({"reply_must_contain": ["x"]}, requires_env=("EVAL_TEST_KEY",)), target
    )
    assert result.skipped
    assert "EVAL_TEST_KEY" in str(result.skip_reason)
    assert calls[0] == 0


async def test_a_judge_case_without_a_judge_skips_rather_than_passes() -> None:
    target, _ = scripted(GOOD)
    result = await run_case(make_case({"judge_safety": "no promises"}), target)
    assert result.skipped


async def test_an_unvalidated_judge_is_advisory() -> None:
    async def strict(rubric: str, transcript: Transcript) -> JudgeVerdict:
        return JudgeVerdict(False, "promised a refund")

    target, _ = scripted(GOOD)
    case = make_case({"judge_safety": "no promises"})
    advisory = await run_case(case, target, judge=strict)
    gating = await run_case(case, target, judge=strict, judge_gates=True)
    assert advisory.failure_kind == "quality"
    assert "advisory" in str(advisory.failed_assertions[0].reason)
    assert gating.failure_kind == "safety"


async def test_a_malformed_assertion_value_is_an_error() -> None:
    target, _ = scripted(GOOD)
    result = await run_case(make_case({"reply_must_contain": [1]}), target)
    assert result.error is not None
    assert result.error.startswith("bad assertion")


@pytest.mark.parametrize(
    "expected",
    [
        {"reply_must_not_match_regex": ["(4111"]},
        {"approval_card_count_max": None},
        {"tool_call_count_max": {"x": "0"}},
    ],
)
async def test_a_bad_value_in_a_case_built_in_code_fails_that_case_only(
    expected: dict[str, object],
) -> None:
    # parse_case rejects these at load; a Case built in code skips parse_case,
    # so the run must still turn them into an error, not a crash of the suite.
    bad = make_case(expected, case_id="bad")
    fine = make_case({"reply_must_contain": ["shipped"]}, case_id="fine")
    target, _ = scripted(GOOD)
    suite = await run_suite([bad, fine], target, RunPolicy(), on_result=None)
    results = {r.case_id: r for r in suite.results}
    assert str(results["bad"].error).startswith("bad assertion")
    assert results["fine"].passed
    assert suite.gating == ["bad"]


async def test_a_judge_that_raises_fails_its_case_not_the_suite() -> None:
    async def broken(rubric: str, transcript: Transcript) -> JudgeVerdict:
        raise KeyError("verdict")

    cases = [
        make_case({"judge": "polite"}, case_id="judged"),
        make_case({"reply_must_contain": ["shipped"]}, case_id="plain"),
    ]
    target, _ = scripted(GOOD)
    suite = await run_suite(cases, target, FAST, judge=broken, on_result=None)
    results = {r.case_id: r for r in suite.results}
    assert "judge" in str(results["judged"].error)
    assert results["judged"].failure_kind == "quality"
    assert results["plain"].passed
    assert suite.gating == ["judged"]


async def test_a_judge_outage_is_infrastructure_and_retried() -> None:
    calls = [0]

    async def busy_then_fine(rubric: str, transcript: Transcript) -> JudgeVerdict:
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")
        return JudgeVerdict(True)

    target, _ = scripted(GOOD)
    case = make_case({"judge": "polite"}, model_driven=False)
    result = await run_case_with_retries(case, target, FAST, judge=busy_then_fine)
    assert result.passed
    assert result.infra_retries == 1


async def test_a_judge_error_keeps_the_violation_it_did_not_judge() -> None:
    async def broken(rubric: str, transcript: Transcript) -> JudgeVerdict:
        raise RuntimeError("overloaded_error")

    target, _ = scripted(LEAK)
    case = make_case({"judge": "polite", "reply_must_not_contain": ["4111 1111 1111 1111"]})
    result = await run_case(case, target, judge=broken)
    assert result.failure_kind == "safety"


async def test_an_outage_during_a_quality_reroll_is_retried() -> None:
    case = make_case({"reply_must_contain": ["shipped"]}, model_driven=False)
    target, calls = scripted(WORDY, RuntimeError("overloaded_error"), GOOD)
    result = await run_case_with_retries(case, target, FAST)
    assert result.passed
    assert calls[0] == 3
    assert result.rerolls == 1
    assert result.infra_retries == 1


async def test_the_suite_gate() -> None:
    cases = [
        make_case({"reply_must_contain": ["shipped"]}, case_id="quality_quarantined"),
        make_case(
            {"reply_must_contain": ["shipped"]}, case_id="quality_gating", single_run_gate=True
        ),
        make_case({"reply_must_not_contain": ["4111"]}, case_id="safety"),
        make_case({"reply_must_contain": ["x"]}, case_id="parked", parked="switched off"),
    ]

    async def target(case: object) -> Transcript:
        return WORDY if "quality" in getattr(case, "id", "") else LEAK

    lines: list[str] = []
    suite = await run_suite(cases, target, RunPolicy(quality_retries=0), on_result=lines.append)
    assert sorted(suite.gating) == ["quality_gating", "safety"]
    assert suite.quarantined == ["quality_quarantined"]
    assert suite.parked == [("parked", "switched off")]
    assert "[PARKED] parked (switched off)" in lines
    assert not suite.passed


async def test_a_crash_gates_even_when_quarantined() -> None:
    target, _ = scripted(RuntimeError("KeyError in my agent"))
    suite = await run_suite(
        [make_case({"reply_must_contain": ["x"]})], target, RunPolicy(), on_result=None
    )
    assert suite.gating == ["c1"]


async def test_too_many_skips_fail_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_TEST_KEY", raising=False)
    cases = [
        make_case({"reply_must_contain": ["x"]}, case_id=f"c{i}", requires_env=("EVAL_TEST_KEY",))
        for i in range(3)
    ]
    target, _ = scripted(GOOD)
    within = await run_suite(cases, target, RunPolicy(max_skips=3), on_result=None)
    over = await run_suite(cases, target, RunPolicy(max_skips=2), on_result=None)
    assert within.passed
    assert not over.passed
    assert over.skip_floor_breached


async def test_outages_gate_unless_told_not_to() -> None:
    target, _ = scripted(RuntimeError("You have reached your usage limits"))
    case = make_case({"reply_must_contain": ["x"]})
    gating = await run_suite([case], target, RunPolicy(), on_result=None)
    lenient = await run_suite([case], target, RunPolicy(infra_gates=False), on_result=None)
    assert gating.gating == ["c1"]
    assert lenient.passed


async def test_custom_outage_markers() -> None:
    markers = OutageMarkers(terminal=("model is asleep",), transient=())
    target, _ = scripted(RuntimeError("the model is asleep"))
    result = await run_case(make_case({"reply_must_contain": ["x"]}), target, markers=markers)
    assert result.failure_kind == "infrastructure"


def test_line_formats() -> None:
    from agent_eval.result import CaseResult

    assert line_for(CaseResult("a")) == "[PASS] a"
    assert line_for(CaseResult("a", skipped=True, skip_reason="env")) == "[SKIP] a (env)"
    assert line_for(CaseResult("a", error="boom")).startswith("[FAIL] a (kind=quality) boom")
