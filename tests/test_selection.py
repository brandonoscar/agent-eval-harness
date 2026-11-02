from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.case import Case
from agent_eval.selection import changes_from_files, is_tier1, rotating_slice, select_cases

from .helpers import ids, make_case


def corpus() -> list[Case]:
    return [
        make_case({"reply_must_contain": ["x"]}, case_id="det", model_driven=False),
        make_case({"reply_must_contain": ["x"]}, case_id="quality_only"),
        make_case({"tool_calls_must_not_include": ["refund_order"]}, case_id="t1_write"),
        make_case({"approval_card_count_max": 1}, case_id="t1_cap"),
        make_case({"reply_must_not_contain": ["4111 1111"]}, case_id="t1_disclosure"),
        *[
            make_case({"reply_must_not_contain": [f"claim {i}"]}, case_id=f"t2_{i}")
            for i in range(6)
        ],
    ]


def test_tiers() -> None:
    by_id = {c.id: c for c in corpus()}
    assert is_tier1(by_id["t1_write"])
    assert is_tier1(by_id["t1_cap"])
    assert is_tier1(by_id["t1_disclosure"])
    assert not is_tier1(by_id["t2_0"])
    assert not is_tier1(make_case({"tool_calls_must_not_include": ["get_context"]}))


def test_a_push_without_an_agent_change_runs_only_free_and_changed_cases() -> None:
    selection = select_cases(
        corpus(), changed_case_ids={"t2_3"}, agent_code_changed=False, rotation_seed="sha"
    )
    assert ids(selection.run) == ["det", "t2_3"]


@pytest.mark.parametrize("agent_code_changed", [True, False])
def test_a_merge_runs_every_safety_case(agent_code_changed: bool) -> None:
    # No rotation seed means the merge to main: every safety case runs, even
    # when nothing matched the agent globs (a pyproject.toml bump can swap the
    # model SDK under the agent).
    selection = select_cases(corpus(), agent_code_changed=agent_code_changed)
    assert set(ids(selection.run)) == {c.id for c in corpus()} - {"quality_only"}
    assert ids(selection.deferred) == ["quality_only"]


def test_a_push_runs_tier1_and_a_rotating_slice_of_tier2() -> None:
    first = select_cases(corpus(), agent_code_changed=True, rotation_seed="sha-a", slice_size=2)
    again = select_cases(corpus(), agent_code_changed=True, rotation_seed="sha-a", slice_size=2)
    other = select_cases(corpus(), agent_code_changed=True, rotation_seed="sha-b", slice_size=2)
    tier2 = [cid for cid in ids(first.run) if cid.startswith("t2_")]
    assert len(tier2) == 2
    assert {"det", "t1_write", "t1_cap", "t1_disclosure"} <= set(ids(first.run))
    assert ids(first.run) == ids(again.run)  # a re-run covers the same cases
    assert len(first.run) + len(first.deferred) == len(corpus())
    assert ids(other.run) != ids(first.run) or len(tier2) == 6


def test_a_zero_slice_rotates_nothing_in() -> None:
    selection = select_cases(corpus(), agent_code_changed=True, rotation_seed="sha", slice_size=0)
    assert not [cid for cid in ids(selection.run) if cid.startswith("t2_")]
    assert rotating_slice(corpus(), seed="x", size=-1) == []


def test_parked_cases_are_outside_the_selection() -> None:
    cases = [*corpus(), make_case({"reply_must_contain": ["x"]}, case_id="p", parked="off")]
    selection = select_cases(cases, agent_code_changed=True)
    assert ids(selection.parked) == ["p"]
    assert "p" not in ids(selection.run) + ids(selection.deferred)
    assert selection.summary().endswith("1 parked")


def test_changes_from_files() -> None:
    cases = [
        Case("a", {}, {"reply_must_contain": "x"}, path=Path("evals/cases/a.json")),
        Case("b", {}, {"reply_must_contain": "x"}, path=Path("evals/cases/b.json")),
    ]
    changed, code = changes_from_files(["evals/cases/a.json", "README.md"], cases, ["src/agent/*"])
    assert changed == {"a"}
    assert not code
    _, code = changes_from_files(["src/agent/prompts.py"], cases, ["src/agent/*"])
    assert code


def test_without_agent_globs_any_change_counts_as_agent_code() -> None:
    # Without globs the selection can't tell agent code from docs, so it
    # assumes the worse: a changed file might be agent code.
    _, code = changes_from_files(["README.md"], [], [])
    assert code
    _, code = changes_from_files([], [], [])
    assert not code
