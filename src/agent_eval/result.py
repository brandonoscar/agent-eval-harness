"""A case's verdict, and the one word that decides what happens next.

``failure_kind`` has three values and each means something different:

* ``safety``: a critical assertion failed (a forbidden action, an over-cap, a
  leak). Blocks the merge. Never re-rolled, never averaged across runs.
* ``quality``: only expectations failed (wording, routing taste). Re-rolled;
  non-gating on quarantined cases; counted by the nightly tracker.
* ``infrastructure``: the provider never answered, so the run judged nothing.

``safety`` is the loudest word the harness has. Spending it on "the API was
down" teaches readers to discount the one label that must never be
discounted, so an outage reads as infrastructure. The rule is asymmetric,
though: an outage never launders a real violation. A forbidden tool call made
before the provider failed is still a forbidden call, so any failed critical
assertion (a prohibition, a cap, a validated judge's safety verdict) keeps the
case at safety unless the violation is the outage reporting itself (a
forbidden ``llm_unavailable`` warning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent_eval.assertions import AssertionResult

FailureKind = Literal["safety", "quality", "infrastructure"]

#: Provider errors where a retry is guaranteed waste: the spend cap names the
#: day access returns, and a wrong key stays wrong.
DEFAULT_TERMINAL_MARKERS: tuple[str, ...] = (
    "usage limits",
    "credit balance",
    "quota exceeded",
    "insufficient_quota",
    "authenticationerror",
    "invalid_api_key",
    "invalid x-api-key",
    "permission_denied",
    "llm_quota_exhausted",
)

#: Provider errors that are a moment, not a state: the same call usually works
#: seconds later. A 429 is a burst limit, not the spend cap, so it lives here
#: (an OpenAI 429 that names ``insufficient_quota`` is the spend cap, and
#: terminal wins).
DEFAULT_TRANSIENT_MARKERS: tuple[str, ...] = (
    "rate_limit_error",
    "ratelimiterror",
    "rate_limit_exceeded",
    "too many requests",
    "overloaded_error",
    "overloadederror",
    "service_unavailable",
    "service unavailable",
    "bad gateway",
    "apiconnectionerror",
    "internalservererror",
    "timeout",
    "llm_unavailable",
)


@dataclass(frozen=True)
class OutageMarkers:
    """Lower-case substrings that mean the provider, not the agent, failed.
    Matched against the run's error and warnings only, never against
    assertion text, so a tool named ``retry_on_timeout`` cannot pass for an
    outage."""

    terminal: tuple[str, ...] = DEFAULT_TERMINAL_MARKERS
    transient: tuple[str, ...] = DEFAULT_TRANSIENT_MARKERS

    def extended(
        self, *, terminal: tuple[str, ...] = (), transient: tuple[str, ...] = ()
    ) -> OutageMarkers:
        """These markers plus more (matched case-insensitively), for a provider
        whose errors the defaults don't name."""
        return OutageMarkers(
            terminal=self.terminal + tuple(m.lower() for m in terminal if m.strip()),
            transient=self.transient + tuple(m.lower() for m in transient if m.strip()),
        )

    def names_outage(self, text: str | None) -> bool:
        return self.names_terminal(text) or self.names_transient(text)

    def names_terminal(self, text: str | None) -> bool:
        return bool(text) and any(m in str(text).lower() for m in self.terminal)

    def names_transient(self, text: str | None) -> bool:
        return bool(text) and any(m in str(text).lower() for m in self.transient)


DEFAULT_MARKERS = OutageMarkers()


@dataclass
class CaseResult:
    case_id: str
    assertions: list[AssertionResult] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    #: The reply the user saw, kept for the drift tracker.
    reply: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    #: Quality re-rolls and provider retries this result took.
    rerolls: int = 0
    infra_retries: int = 0
    markers: OutageMarkers = DEFAULT_MARKERS

    @property
    def passed(self) -> bool:
        if self.skipped:
            return True
        return self.error is None and all(a.passed for a in self.assertions)

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        return [a for a in self.assertions if not a.passed]

    def _outage_texts(self) -> list[str | None]:
        return [self.error, *self.warnings]

    @property
    def infrastructure_failure(self) -> bool:
        """The provider never answered, so this run judged nothing.

        Any critical failure keeps the run at safety: a forbidden call, an
        over-cap or a validated judge's verdict was OBSERVED, and an outage
        afterwards doesn't undo it. The one exception is a forbidden warning
        that is itself the outage marker."""
        if self.passed:
            return False
        if not any(self.markers.names_outage(t) for t in self._outage_texts()):
            return False
        return not any(a.critical and not a.infra_signal for a in self.failed_assertions)

    @property
    def retryable_infrastructure(self) -> bool:
        """A blip rather than a refusal. Terminal wins: a run that names both (a
        rate-limit retry that finally hit the spend cap) is not retried."""
        if not self.infrastructure_failure:
            return False
        texts = self._outage_texts()
        if any(self.markers.names_terminal(t) for t in texts):
            return False
        return any(self.markers.names_transient(t) for t in texts)

    @property
    def failure_kind(self) -> FailureKind | None:
        if self.passed:
            return None
        if self.infrastructure_failure:
            return "infrastructure"
        if any(a.critical for a in self.failed_assertions):
            return "safety"
        # A runner error with no critical failure: an incomplete run made no
        # forbidden move we can see, and a deterministic error fails every
        # attempt anyway.
        return "quality"

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_id,
            "status": "skip" if self.skipped else ("pass" if self.passed else "fail"),
            "kind": self.failure_kind,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "rerolls": self.rerolls,
            "infra_retries": self.infra_retries,
            "reply": self.reply,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "failed": [
                {"name": a.name, "reason": a.reason, "critical": a.critical}
                for a in self.failed_assertions
            ],
        }


__all__ = [
    "DEFAULT_MARKERS",
    "DEFAULT_TERMINAL_MARKERS",
    "DEFAULT_TRANSIENT_MARKERS",
    "CaseResult",
    "FailureKind",
    "OutageMarkers",
]
