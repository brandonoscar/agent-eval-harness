"""Golden-case evals for tool-using agents.

Safety failures block the merge and are never re-rolled; quality misses
re-roll and, on model-driven cases, do not gate; outages are reported as
infrastructure, never as safety; and a nightly tracker reports cases that are
failing, flaky, unreached or drifting in wording.
"""

from agent_eval.assertions import AssertionResult
from agent_eval.case import Case, CaseError, load_cases, parse_case
from agent_eval.judge import JudgeReport, JudgeVerdict, LabeledTrace, validate_judge
from agent_eval.passk import PassKResult, pass_k
from agent_eval.report import gate_headline, summary_markdown
from agent_eval.result import CaseResult, OutageMarkers
from agent_eval.runner import RunPolicy, SuiteResult, Target, run_case, run_suite
from agent_eval.selection import Selection, select_cases
from agent_eval.tracker import Finding, aggregate, find_findings
from agent_eval.transcript import ToolCall, Transcript

__version__ = "0.1.0"

__all__ = [
    "AssertionResult",
    "Case",
    "CaseError",
    "CaseResult",
    "Finding",
    "JudgeReport",
    "JudgeVerdict",
    "LabeledTrace",
    "OutageMarkers",
    "PassKResult",
    "RunPolicy",
    "Selection",
    "SuiteResult",
    "Target",
    "ToolCall",
    "Transcript",
    "__version__",
    "aggregate",
    "find_findings",
    "gate_headline",
    "load_cases",
    "parse_case",
    "pass_k",
    "run_case",
    "run_suite",
    "select_cases",
    "summary_markdown",
    "validate_judge",
]
