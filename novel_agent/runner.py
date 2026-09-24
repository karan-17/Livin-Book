"""Run one conversational turn and turn LangGraph's stream into simple, UI-neutral events.

Both the terminal chat and the Streamlit app consume these events:
    {"type": "memory",      "summary": str}                 old turns were folded into the summary
    {"type": "plan",        "analysis": dict}               output of the analyze node
    {"type": "tool_call",   "name": str, "args": dict}
    {"type": "tool_result", "name": str, "content": str, "sources": list[dict]}
    {"type": "token",       "text": str}                    streamed text from the agent
    {"type": "final",       "text": str, "sources": list[dict], "run_id": str}
"""

from __future__ import annotations

import uuid
from typing import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from .graph import Context, run_config


def stream_turn(graph, question: str, *, thread_id: str, user_id: str, book_id: str = "") -> Iterator[dict]:
    run_id = uuid.uuid4()
    config = run_config(
        thread_id,
        run_id=run_id,
        run_name="novel_agent_turn",
        tags=["novel-agent", book_id] if book_id else ["novel-agent"],
        metadata={"thread_id": thread_id, "user_id": user_id, "book_id": book_id},
    )
    final_text = ""
    sources: dict[str, dict] = {}
    for mode, payload in graph.stream(
        {"messages": [HumanMessage(question)]},
        config,
        context=Context(user_id=user_id),
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk) and chunk.text:
                yield {"type": "token", "text": chunk.text}
            continue

        for node, update in payload.items():
            if not update:
                continue
            if node == "manage_memory" and update.get("summary"):
                yield {"type": "memory", "summary": update["summary"]}
            elif node == "analyze":
                yield {"type": "plan", "analysis": update.get("analysis", {})}
            elif node == "agent":
                message = update["messages"][-1]
                if isinstance(message, AIMessage) and message.tool_calls:
                    for call in message.tool_calls:
                        yield {"type": "tool_call", "name": call["name"], "args": call["args"]}
                else:
                    final_text = message.text
            elif node == "tools":
                for message in update["messages"]:
                    if not isinstance(message, ToolMessage):
                        continue
                    found = message.artifact if isinstance(message.artifact, list) else []
                    for src in found:
                        sources.setdefault(src["chunk_id"], src)
                    yield {"type": "tool_result", "name": message.name, "content": message.text, "sources": found}

    ordered = sorted(sources.values(), key=lambda s: (s["chapter"], s["chunk_id"]))
    yield {"type": "final", "text": final_text, "sources": ordered, "run_id": str(run_id)}


def ask(graph, question: str, *, thread_id: str, user_id: str = "workshop-user", book_id: str = "") -> dict:
    """Blocking convenience wrapper: returns the final event (answer text + sources)."""
    final: dict = {}
    for event in stream_turn(graph, question, thread_id=thread_id, user_id=user_id, book_id=book_id):
        if event["type"] == "final":
            final = event
    return final


def conversation(graph, thread_id: str) -> list[tuple[str, str]]:
    """(role, text) pairs of the visible conversation in a thread (tool traffic hidden)."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    turns = []
    for m in state.values.get("messages", []) if state and state.values else []:
        if isinstance(m, HumanMessage):
            turns.append(("user", m.text))
        elif isinstance(m, AIMessage) and not m.tool_calls and m.text:
            turns.append(("assistant", m.text))
    return turns
