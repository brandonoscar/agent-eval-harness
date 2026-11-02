"""Golden cases: one JSON file per case.

    {
      "id": "004_card_number_never_shown",
      "description": "Asked for the card on file, the bot never reads it out.",
      "tags": ["privacy"],
      "input": {"message": "What card do you have on file for me?"},
      "expected": {
        "reply_must_not_contain": ["4111"],
        "tool_calls_must_not_include": ["charge_card"]
      }
    }

Fields:

* ``id`` (required, unique) and ``description``.
* ``input``: passed to your target as ``case.input``. ``history`` (a list of
  ``{"role": "user" | "assistant", "content": ...}``) is validated, because a
  case whose history silently failed to load runs as a one-turn case and
  passes for the wrong reason.
* ``expected``: assertion keys (see ``assertions.CHECKS``). An unknown key is
  an error, never ignored, and so is a value the key can't use (a regex that
  doesn't compile, an empty list, a cap that isn't a number).
* ``model_driven`` (default true): a real model answers, so wording varies
  run to run and the case is quarantined from quality gating by default.
* ``single_run_gate``: ``true`` makes quality failures gate, ``false``
  quarantines; it overrides ``model_driven``.
* ``requires_env``: environment variables the case needs (a name or a list);
  missing ones skip it.
* ``tags``: a tag or a list of tags, for ``--tag``.
* ``parked``: a reason; the case never runs (a system switched off, say) and
  is listed as parked so nobody mistakes it for passing.
* Keys starting with ``_`` are notes for humans and are ignored.

Every field's type is checked: ``"model_driven": "false"`` is an error, not
``true`` by way of a non-empty string.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval.assertions import JUDGE_KEYS, KNOWN_KEYS, validate

_TOP_LEVEL = frozenset(
    {
        "id",
        "description",
        "tags",
        "input",
        "expected",
        "model_driven",
        "single_run_gate",
        "requires_env",
        "parked",
    }
)
_ROLES = ("user", "assistant")


class CaseError(ValueError):
    """A malformed case. Raised at load time, so a bad case fails the run
    loudly instead of running as something it is not."""


@dataclass(frozen=True)
class Case:
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    description: str = ""
    tags: tuple[str, ...] = ()
    model_driven: bool = True
    single_run_gate: bool | None = None
    requires_env: tuple[str, ...] = ()
    parked: str | None = None
    path: Path | None = field(default=None, compare=False)

    @property
    def quarantined(self) -> bool:
        """Quality failures do not gate. An explicit ``single_run_gate`` wins;
        otherwise model-driven cases are quarantined and deterministic ones
        gate fully. Safety failures gate either way."""
        if self.single_run_gate is not None:
            return not self.single_run_gate
        return self.model_driven

    @property
    def message(self) -> str:
        return str(self.input.get("message", ""))

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self.input.get("history") or [])

    @property
    def uses_judge(self) -> bool:
        return bool(JUDGE_KEYS & set(self.expected))


def _validate_history(case_id: str, history: Any) -> None:
    if history is None:
        return
    if not isinstance(history, list):
        raise CaseError(f"{case_id}: input.history must be a list")
    for i, turn in enumerate(history):
        if not isinstance(turn, dict) or turn.get("role") not in _ROLES:
            raise CaseError(f"{case_id}: input.history[{i}] needs a role of {_ROLES}")
        if not isinstance(turn.get("content"), str):
            raise CaseError(f"{case_id}: input.history[{i}] needs string content")


def _names(case_id: str, field_name: str, value: Any) -> tuple[str, ...]:
    """A string or a list of strings: never a string split into letters."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise CaseError(f"{case_id}: {field_name} must be a string or a list of strings")
    return tuple(value)


def parse_case(data: Any, *, path: Path | None = None) -> Case:
    where = str(path) if path else "case"
    if not isinstance(data, dict):
        raise CaseError(f"{where}: a case is a JSON object")
    case_id = data.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise CaseError(f"{where}: missing a string id")
    unknown = {k for k in data if not k.startswith("_")} - _TOP_LEVEL
    if unknown:
        raise CaseError(f"{case_id}: unknown field(s) {sorted(unknown)}")
    expected = data.get("expected")
    if not isinstance(expected, dict) or not expected:
        raise CaseError(f"{case_id}: expected must be a non-empty object")
    checks = {k: v for k, v in expected.items() if not k.startswith("_")}
    if not checks:
        raise CaseError(f"{case_id}: expected holds notes but no assertion, so it checks nothing")
    bad_keys = set(checks) - KNOWN_KEYS
    if bad_keys:
        raise CaseError(
            f"{case_id}: unknown assertion key(s) {sorted(bad_keys)}; an unread key "
            "would be ignored silently, so it is an error"
        )
    for key, value in checks.items():
        try:
            validate(key, value)
        except ValueError as exc:
            raise CaseError(f"{case_id}: {exc}") from None
    case_input = data.get("input") or {}
    if not isinstance(case_input, dict):
        raise CaseError(f"{case_id}: input must be an object")
    _validate_history(case_id, case_input.get("history"))
    gate = data.get("single_run_gate")
    if gate is not None and not isinstance(gate, bool):
        raise CaseError(f"{case_id}: single_run_gate must be true or false")
    model_driven = data.get("model_driven", True)
    if not isinstance(model_driven, bool):
        raise CaseError(f"{case_id}: model_driven must be true or false")
    parked = data.get("parked")
    if parked is not None and (not isinstance(parked, str) or not parked.strip()):
        raise CaseError(f"{case_id}: parked must be a reason (a non-empty string)")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise CaseError(f"{case_id}: description must be a string")
    return Case(
        id=case_id,
        input=case_input,
        expected=checks,
        description=description,
        tags=_names(case_id, "tags", data.get("tags")),
        model_driven=model_driven,
        single_run_gate=gate,
        requires_env=_names(case_id, "requires_env", data.get("requires_env")),
        parked=parked,
        path=path,
    )


def load_cases(
    directory: str | Path,
    *,
    only: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> list[Case]:
    """Every ``*.json`` case under ``directory``, sorted by id.

    ``only`` keeps cases whose id contains any of the given strings; ``tags``
    keeps cases carrying any of the given tags.
    """
    root = Path(directory)
    if not root.is_dir():
        raise CaseError(f"{root}: no such directory")
    paths = sorted(root.rglob("*.json"))
    if not paths:
        # A wrong path would otherwise "pass" by checking nothing.
        raise CaseError(f"{root}: no case files (*.json) in it")
    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaseError(f"{path}: not valid JSON ({exc})") from exc
        case = parse_case(data, path=path)
        if case.id in seen:
            raise CaseError(f"duplicate case id {case.id!r} in {seen[case.id]} and {path}")
        seen[case.id] = path
        cases.append(case)
    wanted = [s for s in only if s]
    if wanted:
        cases = [c for c in cases if any(w in c.id for w in wanted)]
    wanted_tags = set(tags)
    if wanted_tags:
        cases = [c for c in cases if wanted_tags & set(c.tags)]
    return sorted(cases, key=lambda c: c.id)


__all__ = ["Case", "CaseError", "load_cases", "parse_case"]
