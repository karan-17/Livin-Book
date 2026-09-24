"""The LangGraph agent.

    START
      │
      ▼
    manage_memory   long threads: fold old turns into a running summary, delete them from state
      │
      ▼
    analyze         structured output: standalone question, type, chapters, characters, research steps
      │
      ▼
    agent  ◄─────┐  LLM with tools; follows the research plan, one or more tool calls per turn
      │          │
      ├─► tools ─┘  ToolNode executes the calls (retrieval, knowledge lookups, memory writes)
      ▼
     END

Short-term memory: the checkpointer persists `AgentState` per `thread_id`.
Long-term memory: the store holds the reader profile (spoiler guard + preferences) per `user_id`,
which arrives through the run-scoped `Context`.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from . import prompts
from .config import get_llm
from .memory import get_reader_profile
from .resources import BookResources
from .tools import Context, make_tools

log = logging.getLogger("novel_agent")

MAX_TURNS_BEFORE_SUMMARY = 6  # summarize once a thread has more than this many questions ...
KEEP_TURNS = 3  # ... keeping the most recent turns verbatim
RECURSION_LIMIT = 16  # graph steps per question: ~5 tool rounds before the agent must answer
FINAL_ANSWER_AT = 3  # when this few steps remain, the agent answers without tools


class QuestionAnalysis(BaseModel):
    standalone_question: str = Field(description="The question rewritten to be understandable on its own")
    question_type: Literal[
        "character", "event", "chapter", "relationship", "theme", "ending",
        "comparison", "quote", "reader_update", "chitchat", "other",
    ]
    chapters: list[int] = Field(description="Chapter numbers referenced or implied; empty if none")
    characters: list[str] = Field(description="Characters involved (full names where known)")
    sub_questions: list[str] = Field(description="1-4 research steps that together answer the question")


class AgentState(MessagesState):
    summary: str  # running summary of turns removed from `messages`
    analysis: dict  # QuestionAnalysis for the current turn
    remaining_steps: RemainingSteps  # managed by LangGraph: steps left before the recursion limit


def run_config(thread_id: str, **extra) -> dict:
    """Standard config for invoking the graph."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT, **extra}


def render_transcript(messages: list[AnyMessage], max_chars: int = 700) -> str:
    """Human-readable transcript for the analyzer / summarizer (tool results are omitted)."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Reader: {m.text[:max_chars]}")
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                calls = ", ".join(f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc['args'].items())})"
                                  for tc in m.tool_calls)
                lines.append(f"Assistant looked up: {calls}")
            if m.text:
                lines.append(f"Assistant: {m.text[:max_chars]}")
    return "\n".join(lines)


def format_plan(analysis: dict | None) -> str:
    if not analysis:
        return "(no plan)"
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(analysis.get("sub_questions", []), start=1))
    return (
        f"Question: {analysis.get('standalone_question', '')}\n"
        f"Type: {analysis.get('question_type', 'other')}; chapters: {analysis.get('chapters') or '-'}; "
        f"characters: {', '.join(analysis.get('characters', [])) or '-'}\nSteps:\n{steps}"
    )


def build_graph(res: BookResources, *, checkpointer=None, store=None, llm=None):
    """Compile the agent for one book. Pass a checkpointer/store for memory (see memory.py)."""
    llm = llm or get_llm("agent")
    tools = make_tools(res)
    llm_with_tools = llm.bind_tools(tools)
    llm_answer_only = llm.bind_tools(tools, tool_choice="none")  # tools visible, but calls disabled
    analyzer = llm.with_structured_output(QuestionAnalysis)

    def manage_memory(state: AgentState) -> dict:
        messages = state["messages"]
        human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if len(human_positions) <= MAX_TURNS_BEFORE_SUMMARY:
            return {}
        # Cut at a turn boundary (a HumanMessage) so tool-call/tool-result pairs are never split,
        # and remove messages by id: AIMessages carry Gemini thought signatures and must not be rebuilt.
        cut = human_positions[-KEEP_TURNS]
        old = messages[:cut]
        request = (f"Existing summary:\n{state.get('summary') or '(none)'}\n\n"
                   f"Conversation excerpt to merge in:\n{render_transcript(old)}")
        summary = llm.invoke([SystemMessage(prompts.SUMMARIZE_SYSTEM), HumanMessage(request)]).text
        return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}

    def analyze(state: AgentState) -> dict:
        messages = state["messages"]
        question = messages[-1].text
        history = render_transcript(messages[:-1][-10:]) or "(this is the first message)"
        system = prompts.ANALYZER_SYSTEM.format(
            title=res.title, author=res.author, chapter_count=res.chapter_count,
            summary=state.get("summary") or "(none)",
        )
        request = f"Conversation so far:\n{history}\n\nNewest message from the reader:\n{question}"
        try:
            analysis = analyzer.invoke([SystemMessage(system), HumanMessage(request)])
            if analysis is not None:
                return {"analysis": analysis.model_dump()}
        except Exception as exc:  # the plan is a helpful hint, never a hard dependency
            log.warning("Question analysis failed (%s); continuing without a plan", exc)
        return {"analysis": {"standalone_question": question, "question_type": "other",
                             "chapters": [], "characters": [], "sub_questions": [question]}}

    def agent(state: AgentState, runtime: Runtime[Context]) -> dict:
        context = runtime.context or Context()
        profile = get_reader_profile(runtime.store, context.user_id)
        max_chapter = profile["progress"].get(res.book_id)
        preferences = ""
        if profile["preferences"]:
            preferences = "Reader preferences (always follow):\n" + "\n".join(
                f"- {p}" for p in profile["preferences"]) + "\n"
        system = prompts.AGENT_SYSTEM.format(
            title=res.title, author=res.author, chapter_count=res.chapter_count,
            spoiler_rule=(prompts.SPOILER_RULE_ON.format(max_chapter=max_chapter) if max_chapter
                          else prompts.SPOILER_RULE_OFF),
            preferences=preferences,
            summary=state.get("summary") or "(none)",
            plan=format_plan(state.get("analysis")),
        )
        must_answer = state["remaining_steps"] <= FINAL_ANSWER_AT
        model = llm_answer_only if must_answer else llm_with_tools
        if must_answer:
            system += prompts.FINAL_ANSWER_NUDGE
        response = model.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(AgentState, context_schema=Context)
    builder.add_node("manage_memory", manage_memory)
    builder.add_node("analyze", analyze)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "manage_memory")
    builder.add_edge("manage_memory", "analyze")
    builder.add_edge("analyze", "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, store=store, name="novel_agent")
