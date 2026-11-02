# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-02

First release.

### Added

- Golden cases as JSON files, with assertions on the reply, tool calls, handoffs and approval cards, and verdicts split into safety, quality and infrastructure.
- A runner with parked cases, re-rolls for quality misses, and pass^k for model-driven cases.
- Pull-request selection: tier-1 cases plus a rotating slice seeded by the commit SHA.
- A nightly tracker that reports failing, flaky, unreached and drifting cases, and the wording drift it couldn't measure.
- LLM-as-judge with a validation bar (at least 50 labelled traces, true positive and true negative rates of 90% or more) before it can gate.
- The `agent-eval` CLI (`run`, `track`, `validate-judge`), an OpenAI Agents SDK adapter, a toy support-bot example and two example workflows.

[Unreleased]: https://github.com/brandonoscar/agent-eval-harness/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/brandonoscar/agent-eval-harness/releases/tag/v0.1.0
