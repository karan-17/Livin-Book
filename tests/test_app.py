"""Smoke-test the Streamlit app in-process (streamlit.testing), with fake models."""

from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from .fakes import ScriptedChatModel, tool_call

APP = Path(__file__).resolve().parent.parent / "app.py"


def _script(system, messages, answer_only):
    if isinstance(messages[-1], ToolMessage) or answer_only:
        return AIMessage(content="Darcy is proud at first [Ch. 1].")
    return tool_call("search_book", query="Darcy pride")


def test_app_renders_and_answers(tiny_resources, monkeypatch):
    from novel_agent import graph as graph_module
    from novel_agent import resources as resources_module

    monkeypatch.setattr(resources_module, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(graph_module, "get_llm", lambda role="agent": ScriptedChatModel(
        agent_script=_script,
        analysis={"standalone_question": "Who is Darcy?", "question_type": "character", "chapters": [],
                  "characters": ["Darcy"], "sub_questions": ["look him up"]}))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.header[0].value == "Tiny Pride"

    at.chat_input[0].set_value("Who is Darcy?").run()
    assert not at.exception
    texts = [m.value for msg in at.chat_message for m in msg.markdown]
    assert "Who is Darcy?" in texts and "Darcy is proud at first [Ch. 1]." in texts
    assert any("Sources" in e.label for e in at.expander)
