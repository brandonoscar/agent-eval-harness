"""Evaluate an OpenAI Agents SDK agent (the ``openai-agents`` extra).

    from agent_eval.adapters.openai_agents import openai_agents_target

    target = openai_agents_target(lambda case: build_my_agent(), scrub=my_redactor)

``scrub`` is the post-processing your product applies before the user sees
the reply. Pass it: assertions must read what the user saw.

* **A run that fails partway keeps what it did.** If the run raises (the
  provider drops the connection on the second model call, a max-turns limit),
  the transcript still holds the tool calls already started, the handoffs and
  the error. A forbidden call made before an outage is still a safety failure;
  without its calls the case would read as an outage and be retried until it
  passed.
* **Hosted tools are recorded by kind**, since their calls have no name:
  ``web_search``, ``file_search``, ``code_interpreter``, ``image_generation``,
  ``computer``, ``local_shell``, ``shell``, ``apply_patch``. Forbid them by
  those names (``"tool_calls_must_not_include": ["web_search"]``).

Tool calls made inside a nested ``agent.as_tool(...)`` run are not in the
parent run's items; the nested agent's own result holds them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool
from agents.items import HandoffOutputItem, ModelResponse, ToolCallItem, TResponseInputItem
from agents.result import RunResult, RunResultStreaming
from agents.run_context import AgentHookContext

from agent_eval.case import Case
from agent_eval.runner import Target
from agent_eval.transcript import ToolCall, Transcript

#: Output item types the provider executes itself: no local tool hook fires
#: for them, so they are read from the model's response.
_HOSTED_TYPES = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
    }
)
#: Bookkeeping fields of a hosted call, left out of its arguments.
_NOT_ARGUMENTS = frozenset({"id", "type", "status", "call_id"})


def _field(raw: Any, name: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(name)
    return getattr(raw, name, None)


def _call_name(raw: Any) -> str | None:
    """A function call's name, or a hosted call's kind ("web_search_call" ->
    "web_search"). Never dropped for want of a name."""
    name = _field(raw, "name")
    if name:
        return str(name)
    kind = _field(raw, "type")
    if isinstance(kind, str) and kind:
        return kind.removesuffix("_call")
    return None


def _json_arguments(raw: Any) -> dict[str, Any]:
    try:
        arguments = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": raw}
    return arguments if isinstance(arguments, dict) else {"_value": arguments}


def _call_arguments(raw: Any) -> dict[str, Any]:
    arguments = _field(raw, "arguments")
    if arguments is not None:
        return _json_arguments(arguments)
    if hasattr(raw, "model_dump"):
        data = raw.model_dump(exclude_none=True)
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        return {}
    return {k: v for k, v in data.items() if k not in _NOT_ARGUMENTS}


def transcript_from_result(result: RunResult | RunResultStreaming) -> Transcript:
    calls: list[ToolCall] = []
    handoffs: list[str] = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            name = _call_name(item.raw_item)
            if name is None:
                continue
            calls.append(ToolCall(name, _call_arguments(item.raw_item), agent=item.agent.name))
        elif isinstance(item, HandoffOutputItem):
            handoffs.append(item.target_agent.name)
    return Transcript(
        reply="" if result.final_output is None else str(result.final_output),
        tool_calls=calls,
        handoffs=handoffs,
        final_agent=result.last_agent.name,
        usage=_usage(result.context_wrapper.usage),
    )


def _usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


class _Recorder(RunHooks[Any]):
    """What the run has done so far, for a run that raises before it returns."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.handoffs: list[str] = []
        self.last_agent: str | None = None
        self.usage: Any = None

    async def on_agent_start(self, context: AgentHookContext[Any], agent: Agent[Any]) -> None:
        self.last_agent = agent.name
        self.usage = getattr(context, "usage", self.usage)

    async def on_llm_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], response: ModelResponse
    ) -> None:
        self.usage = context.usage
        for raw in response.output:
            if _field(raw, "type") in _HOSTED_TYPES:
                name = _call_name(raw)
                if name is not None:
                    self.calls.append(ToolCall(name, _call_arguments(raw), agent=agent.name))

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Tool
    ) -> None:
        name = getattr(context, "tool_name", None) or getattr(tool, "name", None)
        arguments = _json_arguments(getattr(context, "tool_arguments", None))
        self.calls.append(ToolCall(str(name or type(tool).__name__), arguments, agent=agent.name))

    async def on_handoff(
        self, context: RunContextWrapper[Any], from_agent: Agent[Any], to_agent: Agent[Any]
    ) -> None:
        self.handoffs.append(to_agent.name)
        self.last_agent = to_agent.name

    def transcript(self, error: str) -> Transcript:
        return Transcript(
            tool_calls=list(self.calls),
            handoffs=list(self.handoffs),
            final_agent=self.last_agent,
            error=error,
            usage=_usage(self.usage),
        )


def run_input(case: Case) -> str | list[TResponseInputItem]:
    """The SDK input for a case: the message, after any history turns."""
    if not case.history:
        return case.message
    items: list[TResponseInputItem] = [
        {"role": turn["role"], "content": turn["content"]}  # type: ignore[misc]
        for turn in case.history
    ]
    items.append({"role": "user", "content": case.message})
    return items


def openai_agents_target(
    build_agent: Callable[[Case], Agent[Any]],
    *,
    scrub: Callable[[str], str] | None = None,
    max_turns: int = 10,
) -> Target:
    """A target that runs ``build_agent(case)`` with the SDK runner."""

    async def target(case: Case) -> Transcript:
        recorder = _Recorder()
        try:
            result = await Runner.run(
                build_agent(case), run_input(case), max_turns=max_turns, hooks=recorder
            )
        except Exception as exc:
            transcript = recorder.transcript(error=f"{type(exc).__name__}: {exc}"[:1000])
        else:
            transcript = transcript_from_result(result)
        if scrub is not None:
            transcript.reply = scrub(transcript.reply)
        return transcript

    return target


__all__ = ["openai_agents_target", "run_input", "transcript_from_result"]
