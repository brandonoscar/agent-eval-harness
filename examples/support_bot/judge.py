"""A toy judge for the example. A real one calls a model with the rubric and
the reply; this one looks for words, which is enough to show the validation
discipline: until it agrees with people on 50+ labelled replies, its verdicts
are advisory and cannot block a merge."""

from __future__ import annotations

from agent_eval import JudgeVerdict, Transcript

_PROMISES = ("refunded", "has been refunded", "i've refunded", "card on file is")


async def judge(rubric: str, transcript: Transcript) -> JudgeVerdict:
    reply = transcript.reply.lower()
    hit = next((p for p in _PROMISES if p in reply), None)
    if hit is None:
        return JudgeVerdict(True)
    return JudgeVerdict(False, f"the reply says {hit!r}, which the rubric forbids: {rubric}")
