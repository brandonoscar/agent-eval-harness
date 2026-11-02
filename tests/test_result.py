from __future__ import annotations

import pytest

from agent_eval.assertions import AssertionResult
from agent_eval.result import CaseResult


def row(name: str, *, critical: bool = True, infra_signal: bool = False) -> AssertionResult:
    return AssertionResult(name, False, "x", critical=critical, infra_signal=infra_signal)


def test_kinds() -> None:
    assert CaseResult("a").failure_kind is None
    assert CaseResult("a", skipped=True, skip_reason="env").failure_kind is None
    assert CaseResult("a", assertions=[row("reply_must_not_contain=x")]).failure_kind == "safety"
    assert (
        CaseResult("a", assertions=[row("reply_must_contain=x", critical=False)]).failure_kind
        == "quality"
    )
    assert CaseResult("a", error="ValueError: bug").failure_kind == "quality"


def test_an_outage_is_infrastructure_not_safety() -> None:
    result = CaseResult(
        "a",
        error="RateLimitError: rate_limit_error",
        assertions=[row("reply_must_contain=x", critical=False)],
    )
    assert result.failure_kind == "infrastructure"
    assert result.retryable_infrastructure


def test_an_outage_never_launders_a_real_violation() -> None:
    # The agent called a forbidden tool, THEN the provider failed.
    result = CaseResult(
        "a",
        error="APIConnectionError: connection reset",
        assertions=[row("tool_calls_must_not_include='refund_order'")],
    )
    assert result.failure_kind == "safety"


def test_the_outage_reporting_itself_is_not_safety() -> None:
    result = CaseResult(
        "a",
        warnings=["llm_unavailable"],
        assertions=[row("warnings_must_not_include='llm_unavailable'", infra_signal=True)],
    )
    assert result.failure_kind == "infrastructure"


def test_terminal_outages_are_not_retried_even_when_transient_is_named_too() -> None:
    result = CaseResult(
        "a", error="rate_limit_error after retries; then: You have reached your usage limits"
    )
    assert result.failure_kind == "infrastructure"
    assert not result.retryable_infrastructure


def test_a_validated_judge_safety_failure_is_never_laundered_by_an_outage() -> None:
    # judge_safety has no "must_not" in its name, but a gating judge's failure
    # is a critical, observed violation like any prohibition.
    result = CaseResult(
        "a",
        warnings=["tool_timeout"],
        assertions=[row("judge_safety")],
    )
    assert result.failure_kind == "safety"
    assert not result.retryable_infrastructure


@pytest.mark.parametrize(
    "error",
    [
        "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached', "
        "'code': 'rate_limit_exceeded'}}",
        "HTTPStatusError: Client error '429 Too Many Requests' for url 'https://api.example.com'",
        "OverloadedError: Error code: 529",
    ],
)
def test_rate_limits_and_overloads_are_transient_outages(error: str) -> None:
    result = CaseResult("a", error=error)
    assert result.failure_kind == "infrastructure"
    assert result.retryable_infrastructure


def test_an_openai_quota_429_is_terminal() -> None:
    result = CaseResult(
        "a", error="RateLimitError: Error code: 429 - {'error': {'code': 'insufficient_quota'}}"
    )
    assert result.failure_kind == "infrastructure"
    assert not result.retryable_infrastructure


def test_a_tool_name_cannot_pass_for_an_outage() -> None:
    # Outage text is read from the error and warnings only, never from
    # assertion reasons, so this forbidden call stays safety.
    result = CaseResult("a", assertions=[row("tool_calls_must_not_include='retry_on_timeout'")])
    assert result.failure_kind == "safety"


def test_as_dict_is_what_the_tracker_reads() -> None:
    data = CaseResult(
        "a", assertions=[row("reply_must_contain=x", critical=False)], reply="hi", rerolls=1
    ).as_dict()
    assert data["status"] == "fail"
    assert data["kind"] == "quality"
    assert data["reply"] == "hi"
    assert data["rerolls"] == 1
    assert data["failed"] == [{"name": "reply_must_contain=x", "reason": "x", "critical": False}]
