"""``agent-eval``: run the gate, track drift, validate a judge.

    agent-eval run --cases evals/cases --target my_app.evals:target
    agent-eval run ... --pr-gate --changed-files changed.txt --agent-glob 'src/agent/*' \\
        --rotate-seed "$GITHUB_SHA"
    agent-eval track runs/*.json --cases evals/cases --baseline evals/baseline.json
    agent-eval validate-judge --traces labeled.jsonl --judge my_app.evals:judge

Exit codes: 0 passed, 1 the gate (or the requested check) failed, 2 could not
run: bad input, a target that won't import, or nothing selected to check.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_eval.case import Case, CaseError, load_cases
from agent_eval.judge import JudgeReportError, load_labeled_traces, read_report, validate_judge
from agent_eval.passk import PassKResult, pass_k
from agent_eval.report import (
    gate_headline,
    passk_headline,
    passk_passed,
    passk_summary_markdown,
    passk_to_json,
    summary_markdown,
    write_json,
)
from agent_eval.result import DEFAULT_MARKERS
from agent_eval.runner import RunPolicy, run_suite
from agent_eval.selection import changes_from_files, select_cases
from agent_eval.tracker import (
    FINDING_KINDS,
    aggregate,
    find_findings,
    load_run,
    new_baseline,
    render_markdown,
)


class UsageError(Exception):
    """The command can't run as given. Exit 2: never read as a pass."""


def load_object(spec: str) -> Any:
    """``package.module:name`` -> the object. The working directory is
    importable, so ``examples.bot:target`` works from a repository root."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise UsageError(f"expected module:name, got {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # the user's module failing to import is a usage error
        raise UsageError(f"could not import {module_name}: {type(exc).__name__}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        raise UsageError(f"{module_name} has no {attr!r}") from None


def _write_summary(text: str, path: str | None) -> None:
    target = path or os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _load(args: argparse.Namespace) -> list[Case]:
    try:
        cases = load_cases(args.cases, only=args.case or (), tags=args.tag or ())
    except CaseError as exc:
        raise UsageError(str(exc)) from None
    if not cases:
        raise UsageError(
            "nothing to run: no case matches the --case/--tag filters, and a check that "
            "checked nothing is not a pass"
        )
    return cases


async def _run(args: argparse.Namespace) -> int:
    cases = _load(args)
    if not args.pr_gate and all(c.parked for c in cases):
        raise UsageError(
            "nothing to run: every selected case is parked (" + ", ".join(c.id for c in cases) + ")"
        )
    target = load_object(args.target)
    judge = load_object(args.judge) if args.judge else None
    judge_gates = False
    if args.judge_report:
        if not args.judge:
            raise UsageError("--judge-report needs --judge: the report vouches for one judge")
        try:
            judge_gates = read_report(args.judge_report, judge=args.judge)
        except JudgeReportError as exc:
            raise UsageError(str(exc)) from None

    if args.pr_gate:
        cases = _pr_gate_selection(args, cases)

    policy = RunPolicy(
        quality_retries=args.retries,
        reroll_quarantined=args.reroll_quarantined,
        max_skips=args.max_skips,
        infra_gates=not args.no_infra_gate,
        markers=DEFAULT_MARKERS.extended(
            terminal=tuple(args.terminal_marker or ()),
            transient=tuple(args.transient_marker or ()),
        ),
        concurrency=args.concurrency,
    )

    if args.pass_k is not None:
        return await _run_pass_k(args, cases, target, policy, judge, judge_gates)

    suite = await run_suite(cases, target, policy, judge=judge, judge_gates=judge_gates)
    headline = gate_headline(suite)
    print(f"\n{headline}")
    if args.json:
        write_json(suite, args.json)
    _write_summary(summary_markdown(suite), args.summary)
    return 0 if suite.passed else 1


def _pr_gate_selection(args: argparse.Namespace, cases: list[Case]) -> list[Case]:
    globs = args.agent_glob or []
    if args.changed_files:
        try:
            changed_files = Path(args.changed_files).read_text().splitlines()
        except OSError as exc:
            raise UsageError(f"can't read --changed-files: {exc}") from None
        changed_ids, code_changed = changes_from_files(changed_files, cases, globs)
        if not globs:
            print("[pr-gate] no --agent-glob: every changed file counts as agent code")
    else:
        # Without the list of changes the gate can't tell what changed, so it
        # assumes the agent did.
        print("[pr-gate] no --changed-files: assuming the agent code changed")
        changed_ids, code_changed = set(), True
    selection = select_cases(
        cases,
        changed_case_ids=changed_ids,
        agent_code_changed=code_changed,
        rotation_seed=args.rotate_seed or "",
        slice_size=args.slice,
    )
    print(f"[pr-gate] {selection.summary(seed=args.rotate_seed or '')}")
    return [*selection.run, *selection.parked]


async def _run_pass_k(
    args: argparse.Namespace,
    cases: list[Case],
    target: Any,
    policy: RunPolicy,
    judge: Any,
    judge_gates: bool,
) -> int:
    k: int = args.pass_k
    results: list[PassKResult] = []
    parked: list[Case] = []
    for case in cases:
        if case.parked:
            parked.append(case)
            print(f"[PARKED] {case.id} ({case.parked})")
            continue
        result = await pass_k(
            case, target, k=k, policy=policy, judge=judge, judge_gates=judge_gates
        )
        print(f"[pass^{k}] {result.summary()}")
        results.append(result)
    print(f"\n{passk_headline(k, results)}")
    if args.json:
        Path(args.json).write_text(  # noqa: ASYNC240 - after every case has run
            json.dumps(passk_to_json(k, results, parked), indent=2) + "\n", encoding="utf-8"
        )
    _write_summary(passk_summary_markdown(k, results, parked), args.summary)
    return 0 if passk_passed(results) else 1


def _track(args: argparse.Namespace) -> int:
    try:
        runs = [load_run(p) for p in args.runs]
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    expected: set[str] = set()
    if args.cases:
        try:
            expected = {c.id for c in load_cases(args.cases) if not c.parked}
        except CaseError as exc:
            raise UsageError(str(exc)) from None
    baseline: dict[str, str] | None = None
    if args.baseline:
        path = Path(args.baseline)
        if path.exists():
            try:
                baseline = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise UsageError(f"can't read the baseline {path}: {exc}") from None
            if not isinstance(baseline, dict):
                raise UsageError(f"{path}: a baseline maps case ids to replies")
        else:
            print(f"baseline {path} not found: wording drift is unmeasured until it exists")
            baseline = {}
    stats = aggregate(runs)
    findings = find_findings(
        stats,
        expected_ids=expected,
        baseline=baseline,
        flake_share=args.flake_share,
        drift_threshold=args.drift_threshold,
    )
    report = render_markdown(findings, stats, len(runs), drift_checked=baseline is not None)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps([f.__dict__ for f in findings], indent=2) + "\n", encoding="utf-8"
        )
    if args.update_baseline:
        if not args.baseline:
            raise UsageError("--update-baseline needs --baseline")
        Path(args.baseline).write_text(
            json.dumps(new_baseline(runs, baseline), indent=2) + "\n", encoding="utf-8"
        )
        print(f"baseline written to {args.baseline}")
    if not any(s.measured for s in stats.values()):
        print("error: nothing measured: no run holds a judged result", file=sys.stderr)
        return 2
    fail_kinds = set(args.fail_on or [])
    return 1 if any(f.kind in fail_kinds for f in findings) else 0


async def _validate_judge(args: argparse.Namespace) -> int:
    try:
        traces = load_labeled_traces(args.traces)
    except JudgeReportError as exc:
        raise UsageError(str(exc)) from None
    judge = load_object(args.judge)
    report = await validate_judge(traces, judge)

    def rate(value: float | None) -> str:
        return "unmeasured" if value is None else f"{value:.3f}"

    print(f"n={report.n} TPR={rate(report.tpr)} TNR={rate(report.tnr)} -> {report.verdict()}")
    for miss in report.misses[:20]:
        print(f"  {miss}")
    if args.out:
        # The report names the judge it measured, so it can't vouch for another.
        data = report.as_dict() | {"judge": args.judge, "traces": str(args.traces)}
        out = Path(args.out)
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")  # noqa: ASYNC240
    return 0 if report.gating else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run cases and decide the gate")
    run.add_argument("--cases", required=True, help="directory of case JSON files")
    run.add_argument("--target", required=True, help="module:function that runs one case")
    run.add_argument("--case", action="append", help="only ids containing this (repeatable)")
    run.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    run.add_argument("--judge", help="module:function judge for judge assertions")
    run.add_argument("--judge-report", help="validate-judge output; lets judge_safety gate")
    run.add_argument("--retries", type=int, default=2, help="quality re-rolls (default 2)")
    run.add_argument("--reroll-quarantined", action="store_true", help="re-roll those too")
    run.add_argument("--max-skips", type=int, default=3)
    run.add_argument("--no-infra-gate", action="store_true", help="outages do not fail the gate")
    run.add_argument(
        "--terminal-marker",
        action="append",
        help="extra text that means the provider refused for good (repeatable)",
    )
    run.add_argument(
        "--transient-marker",
        action="append",
        help="extra text that means a provider blip worth retrying (repeatable)",
    )
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--json", help="write results here (the tracker reads these)")
    run.add_argument("--summary", help="append markdown here (default $GITHUB_STEP_SUMMARY)")
    run.add_argument(
        "--pass-k",
        type=_positive_int,
        help="run each case k times; safe only if all are",
    )
    run.add_argument("--pr-gate", action="store_true", help="run only what a PR must")
    run.add_argument("--changed-files", help="file listing the PR's changed paths")
    run.add_argument("--agent-glob", action="append", help="paths that count as agent code")
    run.add_argument("--rotate-seed", help="commit SHA on a PR push; omit on merge")
    run.add_argument("--slice", type=int, default=12, help="tier-2 cases per push")

    track = sub.add_parser("track", help="find failing, flaky, unreached and drifting cases")
    track.add_argument("runs", nargs="+", help="result JSON files from agent-eval run --json")
    track.add_argument("--cases", help="the corpus, to report cases no run reached")
    track.add_argument("--baseline", help="reply baseline JSON for wording drift")
    track.add_argument("--update-baseline", action="store_true")
    track.add_argument("--flake-share", type=float, default=0.4)
    track.add_argument("--drift-threshold", type=float, default=0.6)
    track.add_argument("--markdown", help="write the report here")
    track.add_argument("--json", help="write findings here")
    track.add_argument(
        "--fail-on",
        action="append",
        choices=FINDING_KINDS,
        help="exit 1 on findings of this kind (repeatable)",
    )

    judge = sub.add_parser("validate-judge", help="measure a judge against labelled replies")
    judge.add_argument("--traces", required=True, help="JSON lines: id, rubric, reply, label")
    judge.add_argument("--judge", required=True, help="module:function")
    judge.add_argument("--out", help="write the report (pass it to run --judge-report)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "track":
            return _track(args)
        return asyncio.run(_validate_judge(args))
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
