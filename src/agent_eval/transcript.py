"""What one run of the agent produced, in the shape the assertions read.

Your target (the function that runs your agent for a case) returns a
``Transcript``. Fill ``reply`` with the text the user actually SAW, after any
redaction or post-processing your product applies, never the raw model output.
An eval that reads the raw output passes cases the user would have failed
(a phone number your scrubber would have removed) and fails cases the user
would have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One tool call, including calls made inside nested agents."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: The agent that made the call, when the target knows it.
    agent: str | None = None


@dataclass
class Transcript:
    #: The reply the user saw.
    reply: str = ""
    #: Every tool call the run made, in order.
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Agents control passed to, in order (handoffs and dispatches).
    handoffs: list[str] = field(default_factory=list)
    #: The agent that produced the reply.
    final_agent: str | None = None
    #: Approval cards a person was shown (their text is checked too).
    approval_cards: list[dict[str, Any]] = field(default_factory=list)
    #: Degraded-path markers your app raised ("llm_unavailable", ...).
    warnings: list[str] = field(default_factory=list)
    #: The error, if the run raised or the provider refused.
    error: str | None = None
    #: Tokens, cost, latency: recorded in the results, never asserted on here.
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    def card_text(self) -> str:
        """Every string inside every approval card, flattened. Keys are skipped:
        they are structure, and matching them would let almost any check pass."""
        parts: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    walk(item)

        walk(self.approval_cards)
        return "\n".join(parts)


__all__ = ["ToolCall", "Transcript"]
