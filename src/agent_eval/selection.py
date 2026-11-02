"""Which cases a pull request pays to run.

Running every model-driven case on every push is the biggest cost in an eval
setup. The rules, from cheapest to most important:

* Deterministic cases (``model_driven: false``) cost nothing: always run.
* A case whose own file changed: always run.
* If agent code changed, every case with a safety bound is at risk:
  - **tier 1** runs on every push: a forbidden write tool, dispatch to a
    forbidden agent, an approval-card or call cap, a forbidden warning or card
    text, a validated judge's safety rubric, or a forbidden identifier
    (digits) in the reply;
  - **tier 2** (the remaining safety bounds) takes turns: each push runs a
    slice chosen by hashing the commit SHA with the case id, so different
    pushes cover different cases and a re-run of the same push covers the
    same ones.
* Without a rotation seed (the merge to main) every safety case runs, whatever
  changed, so no safety case reaches main unrun: a lockfile or ``pyproject.toml``
  bump can swap the model SDK under the agent without touching agent code.
  Rotation only defers within a PR's iterations.
* With no agent globs, every changed file counts as agent code: the selection
  can't tell the agent from the docs, so it assumes the worse.

Everything else waits for the scheduled full run.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agent_eval.assertions import is_disclosure_needle, is_safety_key
from agent_eval.case import Case

TIER1_KEYS = frozenset(
    {
        "judge_safety",
        "handoff_trail_must_not_contain",
        "cards_must_not_contain",
        "warnings_must_not_include",
        "tool_call_count_max",
        "approval_card_count_max",
    }
)

#: Verbs that make a forbidden tool a write. Whole words of the tool name,
#: never substrings ("text" would match every tool with "context" in it).
WRITE_VERBS = frozenset(
    {
        "create",
        "update",
        "delete",
        "remove",
        "send",
        "post",
        "add",
        "upload",
        "share",
        "close",
        "record",
        "assign",
        "apply",
        "refund",
        "charge",
        "pay",
        "transfer",
        "cancel",
        "submit",
        "write",
    }
)

_WORD = re.compile(r"[a-z0-9]+")


def _words(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return set(_WORD.findall(spaced.lower()))


def safety_keys(case: Case) -> set[str]:
    return {k for k in case.expected if is_safety_key(k)}


def is_tier1(case: Case) -> bool:
    """Deferring this case for one push is a real risk, not a scheduling detail."""
    keys = safety_keys(case)
    if keys & TIER1_KEYS:
        return True
    forbidden_tools = case.expected.get("tool_calls_must_not_include") or []
    if isinstance(forbidden_tools, str):
        forbidden_tools = [forbidden_tools]
    if any(_words(str(t)) & WRITE_VERBS for t in forbidden_tools):
        return True
    for key in ("reply_must_not_contain", "reply_must_not_match_regex"):
        needles = case.expected.get(key) or []
        if isinstance(needles, str):
            needles = [needles]
        if any(is_disclosure_needle(str(n)) for n in needles):
            return True
    return False


def rotating_slice(cases: Sequence[Case], *, seed: str, size: int) -> list[Case]:
    """``size`` cases chosen by hashing ``seed`` with each id: stable across
    interpreters and global random state. ``size <= 0`` takes none, so the
    tightest setting of the dial behaves tightest."""
    if size <= 0:
        return []
    if len(cases) <= size:
        return list(cases)
    ranked = sorted(cases, key=lambda c: hashlib.sha256(f"{seed}:{c.id}".encode()).hexdigest())
    return ranked[:size]


@dataclass(frozen=True)
class Selection:
    run: list[Case]
    deferred: list[Case]
    #: Never run, by anyone, until unparked.
    parked: list[Case] = field(default_factory=list)

    def summary(self, *, seed: str = "") -> str:
        where = f"this push (rotation seed {seed[:7]})" if seed else "this merge"
        parked = f"; {len(self.parked)} parked" if self.parked else ""
        return (
            f"running {len(self.run)} of {len(self.run) + len(self.deferred)} cases on {where}; "
            f"{len(self.deferred)} deferred to the scheduled full run{parked}"
        )


def select_cases(
    cases: Sequence[Case],
    *,
    changed_case_ids: Iterable[str] = (),
    agent_code_changed: bool,
    rotation_seed: str = "",
    slice_size: int = 12,
) -> Selection:
    """The cases a PR runs. Pure, so the policy is testable without a model."""
    changed = set(changed_case_ids)
    run: list[Case] = []
    rotatable: list[Case] = []
    deferred: list[Case] = []
    parked: list[Case] = []
    for case in cases:
        if case.parked:
            parked.append(case)
        elif not case.model_driven or case.id in changed:
            run.append(case)
        elif safety_keys(case) and (agent_code_changed or not rotation_seed):
            if rotation_seed and not is_tier1(case):
                rotatable.append(case)
            else:
                run.append(case)
        else:
            deferred.append(case)
    rotated = rotating_slice(rotatable, seed=rotation_seed, size=slice_size)
    picked = {c.id for c in rotated}
    run.extend(rotated)
    deferred.extend(c for c in rotatable if c.id not in picked)
    return Selection(run=sorted(run, key=lambda c: c.id), deferred=deferred, parked=parked)


def changes_from_files(
    changed_files: Iterable[str], cases: Sequence[Case], agent_globs: Iterable[str]
) -> tuple[set[str], bool]:
    """From a PR's changed paths: the ids of cases whose files changed, and
    whether any agent code (matching ``agent_globs``) changed. With no globs,
    any changed file counts as agent code."""
    files = [f.strip() for f in changed_files if f.strip()]
    by_path = {}
    for case in cases:
        if case.path is not None:
            by_path[case.path.as_posix()] = case.id
    changed_ids = {
        case_id
        for changed in files
        for path, case_id in by_path.items()
        if path.endswith(changed) or changed.endswith(path)
    }
    globs = list(agent_globs)
    if not globs:
        return changed_ids, bool(files)
    code_changed = any(fnmatch.fnmatch(f, g) for f in files for g in globs)
    return changed_ids, code_changed


__all__ = [
    "TIER1_KEYS",
    "WRITE_VERBS",
    "Selection",
    "changes_from_files",
    "is_tier1",
    "rotating_slice",
    "safety_keys",
    "select_cases",
]
