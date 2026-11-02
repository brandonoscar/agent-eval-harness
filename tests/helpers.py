from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent_eval import Case, Transcript
from agent_eval.runner import Target


def make_case(
    expected: dict[str, Any],
    *,
    case_id: str = "c1",
    message: str = "hi",
    **fields: Any,
) -> Case:
    return Case(id=case_id, input={"message": message}, expected=expected, **fields)


def scripted(*transcripts: Transcript | Exception) -> tuple[Target, list[int]]:
    """A target that answers attempt by attempt; the last answer repeats."""
    calls = [0]

    async def target(case: Case) -> Transcript:
        index = min(calls[0], len(transcripts) - 1)
        calls[0] += 1
        answer = transcripts[index]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return target, calls


def always(build: Callable[[Case], Transcript]) -> Target:
    async def target(case: Case) -> Transcript:
        return build(case)

    return target


def ids(cases: Sequence[Case]) -> list[str]:
    return [c.id for c in cases]
