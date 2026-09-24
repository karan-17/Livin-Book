"""Streamlit chat UI for the novel agent.

    streamlit run app.py
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

from novel_agent import memory
from novel_agent.graph import build_graph
from novel_agent.indexing import list_ingested_books
from novel_agent.resources import load_resources
from novel_agent.runner import conversation, stream_turn

st.set_page_config(page_title="Novel Agent", page_icon="📚", layout="wide")
TRACING = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() == "true"


@st.cache_resource
def get_memory():
    return memory.sqlite_memory()


@st.cache_resource
def get_agent(book_id: str):
    resources = load_resources(book_id)
    checkpointer, store = get_memory()
    return build_graph(resources, checkpointer=checkpointer, store=store), resources


def new_thread() -> None:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
    st.session_state.last_turn = None


# ---------------------------------------------------------------------------- sidebar
books = list_ingested_books()
if not books:
    st.error("No books ingested yet. Run: `python -m novel_agent.ingest --book pride_and_prejudice`")
    st.stop()

if "thread_id" not in st.session_state:
    new_thread()
st.session_state.setdefault("feedback_sent", set())

with st.sidebar:
    st.title("📚 Novel Agent")
    titles = {b["book_id"]: b["title"] for b in books}
    book_id = st.selectbox("Book", list(titles), format_func=titles.get)
    graph, res = get_agent(book_id)
    _, store = get_memory()
    user_id = st.text_input("Reader id (long-term memory)", value="workshop-user").strip() or "workshop-user"

    st.subheader("Conversation")
    st.button("➕ New chat", on_click=new_thread, width="stretch")
    thread_input = st.text_input("Thread id (resume any thread)", value=st.session_state.thread_id)
    if thread_input.strip() and thread_input.strip() != st.session_state.thread_id:
        st.session_state.thread_id, st.session_state.last_turn = thread_input.strip(), None
        st.rerun()

    st.subheader("Spoiler guard")
    current = memory.get_max_chapter(store, user_id, book_id) or 0
    progress = st.slider("I've read up to chapter (0 = finished the book)", 0, res.chapter_count, current)
    if progress != current:
        memory.set_reading_progress(store, user_id, book_id, progress or None)
        st.toast(f"Spoiler guard {'ON up to chapter ' + str(progress) if progress else 'OFF'}")

    prefs = memory.get_reader_profile(store, user_id)["preferences"]
    if prefs:
        st.subheader("Saved preferences")
        for p in prefs:
            st.caption(f"• {p}")
        if st.button("Clear preferences"):
            memory.clear_preferences(store, user_id)
            st.rerun()

    show_steps = st.toggle("Show reasoning steps", value=True)
    st.caption(f"LangSmith tracing: {'on · project ' + os.getenv('LANGSMITH_PROJECT', 'default') if TRACING else 'off'}")

# ---------------------------------------------------------------------------- history
st.header(res.title)
st.caption(f"by {res.author} · {res.chapter_count} chapters · thread `{st.session_state.thread_id}`")

history = conversation(graph, st.session_state.thread_id)
if not history:
    st.info("Try: *How does Elizabeth's opinion of Darcy change over the novel?* then a follow-up like "
            "*What made her reconsider?* — or tell it *I've only read up to chapter 20*.")
for role, text in history:
    with st.chat_message(role):
        st.markdown(text)

last = st.session_state.get("last_turn")
if history and last and last["thread_id"] == st.session_state.thread_id:
    if last["sources"]:
        with st.expander(f"📖 Sources for the last answer ({len(last['sources'])} passages)"):
            for s in last["sources"]:
                st.markdown(f"**Chapter {s['chapter']}** · `{s['chunk_id']}`")
                st.caption(s["text"][:700])
    if TRACING:
        rating = st.feedback("thumbs", key=f"feedback_{last['run_id']}")
        if rating is not None and last["run_id"] not in st.session_state.feedback_sent:
            from langsmith import Client

            Client().create_feedback(run_id=last["run_id"], key="user_rating", score=rating)
            st.session_state.feedback_sent.add(last["run_id"])
            st.toast("Thanks! Feedback sent to LangSmith.")

# ---------------------------------------------------------------------------- new turn
if question := st.chat_input("Ask about characters, events, chapters, relationships, themes…"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Researching…", expanded=show_steps)
        placeholder = st.empty()
        answer, final, tool_calls = "", {}, 0
        try:
            for event in stream_turn(graph, question, thread_id=st.session_state.thread_id,
                                     user_id=user_id, book_id=book_id):
                kind = event["type"]
                if kind == "token":
                    answer += event["text"]
                    placeholder.markdown(answer + " ▌")
                elif kind == "plan":
                    plan = event["analysis"]
                    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(plan.get("sub_questions", []), 1))
                    status.markdown(f"**🧭 Plan** · _{plan.get('question_type')}_ — "
                                    f"{plan.get('standalone_question')}\n\n{steps}")
                elif kind == "tool_call":
                    if answer:  # text the model wrote before deciding to call a tool
                        status.markdown(f"💭 {answer}")
                        answer = ""
                        placeholder.empty()
                    tool_calls += 1
                    args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                    status.markdown(f"🔧 `{event['name']}({args})`")
                elif kind == "tool_result":
                    status.caption(event["content"][:300] + ("…" if len(event["content"]) > 300 else ""))
                elif kind == "memory":
                    status.markdown("🧠 Older turns were folded into the conversation summary.")
                elif kind == "final":
                    final = event
            status.update(label=f"Done · {tool_calls} tool call(s)", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()
        placeholder.markdown(final.get("text") or answer)
    st.session_state.last_turn = {"thread_id": st.session_state.thread_id, "run_id": final.get("run_id"),
                                  "sources": final.get("sources", [])}
    st.rerun()
