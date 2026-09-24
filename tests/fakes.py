"""A scripted stand-in for Gemini so the LangGraph agent can be tested offline."""

from __future__ import annotations

import itertools
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

_ids = itertools.count(1)


def tool_call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}",
                                              "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """`agent_script(system_prompt, messages, answer_only) -> AIMessage` decides every agent reply.

    - summarizer calls (system prompt about "running memory") get a fixed summary
    - `with_structured_output` returns the `analysis` dict validated into the requested schema
    - `bind_tools(..., tool_choice="none")` yields an "answer only" variant (as Gemini's NONE mode)
    """

    agent_script: Callable[[str, list[BaseMessage], bool], AIMessage]
    analysis: dict = {}
    answer_only: bool = False
    seen_system_prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        if "running memory" in system:
            reply = AIMessage(content="SUMMARY: the reader asked several questions about Darcy.")
        else:
            self.seen_system_prompts.append(system)
            reply = self.agent_script(system, list(messages), self.answer_only)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.model_copy(update={"answer_only": tool_choice == "none"})

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _input: schema(**self.analysis))
