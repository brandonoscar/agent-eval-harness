from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_eval.case import CaseError, load_cases, parse_case


def write(directory: Path, name: str, data: Any) -> None:
    (directory / name).write_text(json.dumps(data))


def base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "001_x",
        "input": {"message": "hi"},
        "expected": {"reply_must_contain": ["hi"]},
    }
    data.update(overrides)
    return data


def test_parses_and_ignores_notes() -> None:
    case = parse_case(base(_why="because", expected={"reply_must_contain": "hi", "_note": "x"}))
    assert case.expected == {"reply_must_contain": "hi"}
    assert case.message == "hi"


def test_an_unknown_assertion_key_is_an_error() -> None:
    with pytest.raises(CaseError, match="unknown assertion key"):
        parse_case(base(expected={"reply_must_contian": ["hi"]}))


def test_an_unknown_field_is_an_error() -> None:
    with pytest.raises(CaseError, match="unknown field"):
        parse_case(base(expectd={}))


@pytest.mark.parametrize(
    "history",
    [
        "not a list",
        [{"role": "system", "content": "x"}],
        [{"role": "user", "content": 3}],
    ],
)
def test_a_bad_history_fails_loudly(history: Any) -> None:
    with pytest.raises(CaseError, match="history"):
        parse_case(base(input={"message": "hi", "history": history}))


def test_quarantine_rules() -> None:
    assert parse_case(base()).quarantined  # model-driven by default
    assert not parse_case(base(model_driven=False)).quarantined
    assert not parse_case(base(single_run_gate=True)).quarantined
    assert parse_case(base(model_driven=False, single_run_gate=False)).quarantined
    with pytest.raises(CaseError, match="single_run_gate"):
        parse_case(base(single_run_gate="yes"))


def test_load_filters_and_rejects_duplicates(tmp_path: Path) -> None:
    write(tmp_path, "a.json", base(id="001_orders", tags=["orders"]))
    write(tmp_path, "b.json", base(id="002_weather", tags=["routing"]))
    assert [c.id for c in load_cases(tmp_path)] == ["001_orders", "002_weather"]
    assert [c.id for c in load_cases(tmp_path, only=["weather"])] == ["002_weather"]
    assert [c.id for c in load_cases(tmp_path, tags=["orders"])] == ["001_orders"]
    write(tmp_path, "c.json", base(id="001_orders"))
    with pytest.raises(CaseError, match="duplicate case id"):
        load_cases(tmp_path)


def test_bad_json_and_missing_dirs(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(CaseError, match="not valid JSON"):
        load_cases(tmp_path)
    with pytest.raises(CaseError, match="no such directory"):
        load_cases(tmp_path / "missing")


def test_missing_id_or_expected() -> None:
    with pytest.raises(CaseError, match="string id"):
        parse_case({"expected": {"reply_must_contain": "x"}})
    with pytest.raises(CaseError, match="non-empty"):
        parse_case(base(expected={}))


def test_an_expected_block_of_notes_alone_checks_nothing() -> None:
    with pytest.raises(CaseError, match="no assertion"):
        parse_case(base(expected={"_note": "TODO: write the checks"}))


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"reply_must_not_match_regex": ["(4111"]}, "not a valid regular expression"),
        ({"approval_card_count_max": None}, "whole number"),
        ({"tool_call_count_max": {"x": "0"}}, "whole number"),
        ({"tool_calls_must_not_include": []}, "checks nothing"),
    ],
)
def test_a_bad_assertion_value_fails_at_load(expected: dict[str, Any], message: str) -> None:
    with pytest.raises(CaseError, match=message):
        parse_case(base(expected=expected))


def test_string_or_list_fields_take_either() -> None:
    case = parse_case(base(requires_env="OPENAI_API_KEY", tags="privacy"))
    assert case.requires_env == ("OPENAI_API_KEY",)
    assert case.tags == ("privacy",)
    case = parse_case(base(requires_env=["A_KEY", "B_KEY"], tags=["x", "y"]))
    assert case.requires_env == ("A_KEY", "B_KEY")
    assert case.tags == ("x", "y")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_driven", "false", "model_driven must be true or false"),
        ("model_driven", 0, "model_driven must be true or false"),
        ("requires_env", [1], "requires_env"),
        ("requires_env", {"KEY": 1}, "requires_env"),
        ("tags", [None], "tags"),
        ("parked", True, "parked"),
        ("parked", "", "parked"),
        ("description", ["x"], "description"),
    ],
)
def test_field_types_are_checked(field: str, value: Any, message: str) -> None:
    with pytest.raises(CaseError, match=message):
        parse_case(base(**{field: value}))


def test_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="no case files"):
        load_cases(tmp_path)


def test_the_example_corpus_loads() -> None:
    corpus = Path(__file__).parents[1] / "examples" / "support_bot" / "cases"
    cases = load_cases(corpus)
    assert len(cases) == 10
    assert sum(1 for c in cases if c.parked) == 1
