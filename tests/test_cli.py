"""The CLI end to end, against the example support bot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.cli import main

ROOT = Path(__file__).parents[1]
CASES = str(ROOT / "examples" / "support_bot" / "cases")
TARGET = "examples.support_bot.bot:target"
JUDGE = "examples.support_bot.judge:judge"


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("SUPPORT_BOT_BUG", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def test_the_example_suite_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.json"
    summary = tmp_path / "summary.md"
    code = main(
        ["run", "--cases", CASES, "--target", TARGET, "--json", str(out), "--summary", str(summary)]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "EVAL GATE PASSED" in printed
    assert "[PARKED] 009_live_chat_transfer" in printed
    assert "[SKIP] 010_judge_no_unapproved_refund_claim" in printed
    data = json.loads(out.read_text())
    assert data["passed"] is True
    assert {r["case"] for r in data["results"]} >= {"003_card_number_never_shown"}
    assert summary.read_text().startswith("### EVAL GATE PASSED")


@pytest.mark.parametrize(
    ("bug", "kind", "case"),
    [
        ("leak_card", "safety", "003_card_number_never_shown"),
        ("refund_without_approval", "safety", "002_refund_needs_approval"),
    ],
)
def test_a_safety_regression_blocks_the_merge(
    bug: str,
    kind: str,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SUPPORT_BOT_BUG", bug)
    out = tmp_path / "run.json"
    code = main(["run", "--cases", CASES, "--target", TARGET, "--json", str(out)])
    assert code == 1
    assert f"[FAIL] {case} (kind={kind})" in capsys.readouterr().out
    assert case in json.loads(out.read_text())["gating"]


def test_an_outage_is_reported_as_infrastructure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SUPPORT_BOT_BUG", "outage")
    monkeypatch.setattr("agent_eval.runner.RunPolicy.infra_backoff_s", (0.0,))
    code = main(["run", "--cases", CASES, "--target", TARGET, "--case", "003"])
    printed = capsys.readouterr().out
    assert code == 1
    assert "(kind=infrastructure)" in printed
    assert "kind=safety" not in printed


def test_pass_k(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--cases", CASES, "--target", TARGET, "--case", "003", "--pass-k", "3"])
    assert code == 0
    assert "safe on 3 of 3 runs" in capsys.readouterr().out


def test_pr_gate_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("examples/support_bot/cases/005_weather_handoff.json\n")
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--pr-gate",
            "--changed-files",
            str(changed),
            "--agent-glob",
            "examples/support_bot/bot.py",
            "--rotate-seed",
            "0123abc",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "[pr-gate] running 2 of 9 cases" in printed  # the deterministic case + the changed one
    assert "[PASS] 005_weather_handoff" in printed


def test_pr_gate_without_a_seed_runs_every_safety_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The merge to main: only pyproject.toml changed, nothing matched the glob,
    # and still no safety case may reach main unrun.
    changed = tmp_path / "changed.txt"
    changed.write_text("pyproject.toml\n")
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--pr-gate",
            "--changed-files",
            str(changed),
            "--agent-glob",
            "examples/support_bot/bot.py",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    for case in (
        "002_refund_needs_approval",
        "003_card_number_never_shown",
        "004_prompt_injection_refund",
    ):
        assert f"[PASS] {case}" in printed


def test_pr_gate_without_agent_globs_counts_every_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("docs/notes.md\n")
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--pr-gate",
            "--changed-files",
            str(changed),
            "--rotate-seed",
            "0123abc",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "no --agent-glob" in printed
    assert "[PASS] 002_refund_needs_approval" in printed  # a tier-1 case ran


def test_track_and_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = []
    for seed in ("1", "2"):
        monkeypatch.setenv("SUPPORT_BOT_SEED", seed)
        path = tmp_path / f"run-{seed}.json"
        main(
            [
                "run",
                "--cases",
                CASES,
                "--target",
                TARGET,
                "--reroll-quarantined",
                "--json",
                str(path),
            ]
        )
        runs.append(str(path))
    baseline = tmp_path / "baseline.json"
    report = tmp_path / "report.md"
    assert (
        main(
            [
                "track",
                *runs,
                "--cases",
                CASES,
                "--baseline",
                str(baseline),
                "--update-baseline",
                "--markdown",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(baseline.read_text())["003_card_number_never_shown"].startswith(
        "For your security"
    )

    monkeypatch.setenv("SUPPORT_BOT_BUG", "chatty")
    drifted = tmp_path / "drifted.json"
    main(["run", "--cases", CASES, "--target", TARGET, "--json", str(drifted)])
    findings = tmp_path / "findings.json"
    code = main(
        [
            "track",
            str(drifted),
            "--baseline",
            str(baseline),
            "--json",
            str(findings),
            "--fail-on",
            "drift",
        ]
    )
    assert code == 1
    kinds = {(f["case"], f["kind"]) for f in json.loads(findings.read_text())}
    assert ("001_order_status", "drift") in kinds


def test_validate_judge_is_advisory_for_the_toy_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "judge.json"
    traces = str(ROOT / "examples" / "support_bot" / "judge_traces.jsonl")
    code = main(["validate-judge", "--traces", traces, "--judge", JUDGE, "--out", str(out)])
    assert code == 1
    assert "advisory" in capsys.readouterr().out
    assert json.loads(out.read_text())["gating"] is False


def test_bad_cases_are_exit_2(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({"id": "x", "expected": {"nope": 1}}))
    assert main(["run", "--cases", str(tmp_path), "--target", TARGET]) == 2


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no_colon", "module:name"),
        ("examples.support_bot.no_such_module:target", "could not import"),
        ("examples.support_bot.bot:no_such_name", "has no 'no_such_name'"),
    ],
)
def test_a_bad_target_is_exit_2(
    spec: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--cases", CASES, "--target", spec]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "filters",
    [["--tag", "privcy"], ["--case", "no_such_case"], ["--case", "009"]],
)
def test_selecting_nothing_to_run_is_exit_2(
    filters: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    # A typo in a filter (or picking only a parked case) checks nothing, and
    # nothing checked is not a pass.
    assert main(["run", "--cases", CASES, "--target", TARGET, *filters]) == 2
    err = capsys.readouterr().err
    assert "nothing to run" in err


def test_an_empty_case_directory_is_exit_2(tmp_path: Path) -> None:
    assert main(["run", "--cases", str(tmp_path), "--target", TARGET]) == 2


def test_the_headline_counts_skips_apart_from_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--cases", CASES, "--target", TARGET]) == 0
    printed = capsys.readouterr().out
    # 010 needs a judge and skips: it was not run, so it isn't counted as run.
    assert "EVAL GATE PASSED: 8 case(s) run, 1 skipped" in printed


def test_a_total_outage_under_no_infra_gate_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SUPPORT_BOT_BUG", "outage")
    monkeypatch.setattr("agent_eval.runner.RunPolicy.infra_backoff_s", (0.0,))
    summary = tmp_path / "summary.md"
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--case",
            "003",
            "--no-infra-gate",
            "--summary",
            str(summary),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "EVAL GATE PASSED: 0 case(s) run, 1 outage(s) not gating" in printed
    assert "no (outages not gating)" in summary.read_text()
    assert "quarantined" not in summary.read_text()


def test_extra_outage_markers_from_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("examples.support_bot.bot.target", _napping_target)
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--case",
            "003",
            "--terminal-marker",
            "Model Is Napping",
        ]
    )
    assert code == 1
    assert "(kind=infrastructure)" in capsys.readouterr().out


async def _napping_target(case: object) -> object:
    raise RuntimeError("the model is napping")


def test_pass_k_skips_parked_cases_and_writes_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "passk.json"
    summary = tmp_path / "summary.md"
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--case",
            "003",
            "--case",
            "009",
            "--pass-k",
            "2",
            "--json",
            str(out),
            "--summary",
            str(summary),
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "[PARKED] 009_live_chat_transfer" in printed
    assert "[pass^2] 009" not in printed
    assert "PASS^2 PASSED: 1 case(s) safe on 2 of 2 runs" in printed
    data = json.loads(out.read_text())
    assert data["mode"] == "pass_k"
    assert [c["case"] for c in data["cases"]] == ["003_card_number_never_shown"]
    assert data["parked"] == [{"case": "009_live_chat_transfer", "reason": ANY_REASON}]
    assert summary.read_text().startswith("### PASS^2 PASSED")


ANY_REASON = "live chat is switched off in this deployment; unpark and run before switching it on"


def test_pass_k_must_be_at_least_1() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "--cases", CASES, "--target", TARGET, "--pass-k", "0"])
    assert exc.value.code == 2


def test_track_rejects_a_misspelt_finding_kind() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["track", "runs.json", "--fail-on", "safty"])
    assert exc.value.code == 2


def test_track_with_bad_input_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_cases = tmp_path / "cases"
    bad_cases.mkdir()
    (bad_cases / "bad.json").write_text("{nope")
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({"results": [{"case": "a", "status": "pass"}]}))
    assert main(["track", str(run_file), "--cases", str(bad_cases)]) == 2
    assert main(["track", str(tmp_path / "missing.json")]) == 2
    passk_file = tmp_path / "passk.json"
    passk_file.write_text(json.dumps({"mode": "pass_k", "cases": []}))
    assert main(["track", str(passk_file)]) == 2
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"results": []}))
    assert main(["track", str(empty)]) == 2
    assert "Nothing measured" in capsys.readouterr().out


def test_track_reports_a_case_no_run_could_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = []
    for i in range(2):
        path = tmp_path / f"run-{i}.json"
        main(["run", "--cases", CASES, "--target", TARGET, "--json", str(path)])
        runs.append(str(path))
    report = tmp_path / "report.md"
    main(["track", *runs, "--cases", CASES, "--markdown", str(report)])
    text = report.read_text()
    assert "No findings" not in text
    assert "| `010_judge_no_unapproved_refund_claim` | **unreached** |" in text


def test_a_judge_report_for_another_judge_is_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "judge.json"
    report.write_text(
        json.dumps({"n": 60, "tp": 30, "fp": 0, "tn": 30, "fn": 0, "judge": "other:judge"})
    )
    code = main(
        [
            "run",
            "--cases",
            CASES,
            "--target",
            TARGET,
            "--judge",
            JUDGE,
            "--judge-report",
            str(report),
        ]
    )
    assert code == 2
    assert "validated 'other:judge'" in capsys.readouterr().err


def test_validate_judge_records_which_judge_it_measured(tmp_path: Path) -> None:
    out = tmp_path / "judge.json"
    traces = str(ROOT / "examples" / "support_bot" / "judge_traces.jsonl")
    main(["validate-judge", "--traces", traces, "--judge", JUDGE, "--out", str(out)])
    assert json.loads(out.read_text())["judge"] == JUDGE


def test_validate_judge_with_unreadable_traces_is_exit_2(tmp_path: Path) -> None:
    bad = tmp_path / "t.jsonl"
    bad.write_text(json.dumps({"id": "1", "reply": "x"}) + "\n")
    assert main(["validate-judge", "--traces", str(bad), "--judge", JUDGE]) == 2
    assert main(["validate-judge", "--traces", str(tmp_path / "nope"), "--judge", JUDGE]) == 2
