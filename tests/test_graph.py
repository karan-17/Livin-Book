"""End-to-end tests of the LangGraph agent with a scripted LLM (no network, no keys)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from novel_agent.graph import build_graph
from novel_agent.memory import get_max_chapter, in_memory
from novel_agent.runner import ask, conversation, stream_turn

from .fakes import ScriptedChatModel, tool_call

ANALYSIS = {"standalone_question": "q", "question_type": "character", "chapters": [], "characters": ["Darcy"],
            "sub_questions": ["look up Darcy"]}


def _search_then_answer(system, messages, answer_only):
    """Call search_book once per question, then answer quoting how many tool results it saw."""
    if not isinstance(messages[-1], ToolMessage) and not answer_only:
        question = messages[-1].text
        return tool_call("search_book", query=question)
    return AIMessage(content=f"Answer [Ch. 1] after {sum(isinstance(m, ToolMessage) for m in messages)} lookups")


def make_graph(resources, script=_search_then_answer):
    checkpointer, store = in_memory()
    llm = ScriptedChatModel(agent_script=script, analysis=ANALYSIS)
    return build_graph(resources, checkpointer=checkpointer, store=store, llm=llm), store, llm


def test_tool_loop_produces_cited_answer_with_sources(tiny_resources):
    graph, _, _ = make_graph(tiny_resources)
    events = list(stream_turn(graph, "Why does Elizabeth dislike Darcy?", thread_id="t1", user_id="u1"))
    types = [e["type"] for e in events]
    assert types[0] == "plan" and "tool_call" in types and "tool_result" in types and types[-1] == "final"
    final = events[-1]
    assert final["text"].startswith("Answer [Ch. 1]")
    assert final["sources"] and all("chunk_id" in s for s in final["sources"])


def test_thread_memory_keeps_follow_ups_in_context(tiny_resources):
    graph, _, _ = make_graph(tiny_resources)
    ask(graph, "Who is Darcy?", thread_id="t1")
    second = ask(graph, "And what about his letter?", thread_id="t1")
    assert "after 2 lookups" in second["text"]  # sees the tool results of both turns
    assert [role for role, _ in conversation(graph, "t1")] == ["user", "assistant", "user", "assistant"]
    other = ask(graph, "Who is Jane?", thread_id="t2")  # a new thread starts fresh
    assert "after 1 lookups" in other["text"]


def test_spoiler_guard_persists_across_threads(tiny_resources):
    def script(system, messages, answer_only):
        last = messages[-1]
        if isinstance(last, HumanMessage) and "read up to" in last.text:
            return tool_call("set_reading_progress", chapter=3)
        if isinstance(last, HumanMessage):
            return tool_call("search_book", query="Lydia elopes with Wickham")
        return AIMessage(content="ok")

    graph, store, llm = make_graph(tiny_resources, script)
    ask(graph, "I've read up to chapter 3", thread_id="a", user_id="reader")
    assert get_max_chapter(store, "reader", "tiny") == 3

    ask(graph, "What does Lydia do?", thread_id="b", user_id="reader")  # new thread, same reader
    state = graph.get_state({"configurable": {"thread_id": "b"}})
    tool_output = next(m for m in state.values["messages"] if isinstance(m, ToolMessage))
    assert "Ch. 5" not in tool_output.text and "Spoiler guard" in tool_output.text
    assert "SPOILER GUARD" in llm.seen_system_prompts[-1]

    ask(graph, "What does Lydia do?", thread_id="c", user_id="someone-else")  # other readers unaffected
    assert "SPOILER GUARD" not in llm.seen_system_prompts[-1]


def test_long_threads_are_summarised_at_turn_boundaries(tiny_resources):
    graph, _, _ = make_graph(tiny_resources)
    for i in range(7):
        ask(graph, f"Question {i} about Darcy", thread_id="long")
    values = graph.get_state({"configurable": {"thread_id": "long"}}).values
    messages = values["messages"]
    assert values["summary"].startswith("SUMMARY")
    assert isinstance(messages[0], HumanMessage) and messages[0].text == "Question 4 about Darcy"
    assert sum(isinstance(m, HumanMessage) for m in messages) == 3
    # every tool result still has its tool call
    call_ids = {tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    assert all(m.tool_call_id in call_ids for m in messages if isinstance(m, ToolMessage))


def test_sqlite_memory_survives_a_restart(tiny_resources, tmp_path):
    from novel_agent.memory import set_reading_progress, sqlite_memory

    llm = ScriptedChatModel(agent_script=_search_then_answer, analysis=ANALYSIS)
    checkpointer, store = sqlite_memory(tmp_path / "mem")
    graph = build_graph(tiny_resources, checkpointer=checkpointer, store=store, llm=llm)
    ask(graph, "Who is Darcy?", thread_id="persist", user_id="u")
    set_reading_progress(store, "u", "tiny", 4)

    checkpointer2, store2 = sqlite_memory(tmp_path / "mem")  # "restart": fresh connections, same files
    graph2 = build_graph(tiny_resources, checkpointer=checkpointer2, store=store2, llm=llm)
    assert [r for r, _ in conversation(graph2, "persist")] == ["user", "assistant"]
    assert get_max_chapter(store2, "u", "tiny") == 4


def test_agent_is_forced_to_answer_when_budget_runs_out(tiny_resources):
    def never_stops(system, messages, answer_only):
        if answer_only:
            return AIMessage(content="Best answer with what I have.")
        return tool_call("find_chapters", query="Darcy")

    graph, _, _ = make_graph(tiny_resources, never_stops)
    final = ask(graph, "Tell me everything", thread_id="loop")
    assert final["text"] == "Best answer with what I have."
