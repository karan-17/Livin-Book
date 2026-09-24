"""Memory for the agent.

Short-term memory  = the conversation thread. A LangGraph *checkpointer* saves the full graph state
                     after every step, keyed by `thread_id`, so follow-up questions have context.
Long-term memory   = facts about the reader that outlive any one thread, kept in a LangGraph
                     *store*: reading progress per book (drives the spoiler guard) and preferences.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .config import get_settings

PROFILE_KEY = "profile"


def in_memory() -> tuple[InMemorySaver, InMemoryStore]:
    """Throwaway memory (notebook, tests): gone when the process exits."""
    return InMemorySaver(), InMemoryStore()


def sqlite_memory(directory: Path | None = None):
    """Persistent memory (CLI, Streamlit): survives restarts. Returns (checkpointer, store)."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore

    directory = directory or get_settings().memory_dir
    directory.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(directory / "checkpoints.sqlite", check_same_thread=False))
    # SqliteStore manages its own transactions, so its connection must be in autocommit mode.
    store = SqliteStore(sqlite3.connect(directory / "store.sqlite", check_same_thread=False, isolation_level=None))
    store.setup()
    return checkpointer, store


# --------------------------------------------------------------------------------------
# Reader profile helpers  —  namespace ("readers", <user_id>), key "profile"
# --------------------------------------------------------------------------------------


def _namespace(user_id: str) -> tuple[str, str]:
    return ("readers", user_id)


def get_reader_profile(store: BaseStore | None, user_id: str) -> dict:
    if store is None:
        return {"progress": {}, "preferences": []}
    item = store.get(_namespace(user_id), PROFILE_KEY)
    value = dict(item.value) if item else {}
    value.setdefault("progress", {})
    value.setdefault("preferences", [])
    return value


def _save(store: BaseStore, user_id: str, profile: dict) -> None:
    store.put(_namespace(user_id), PROFILE_KEY, profile)


def get_max_chapter(store: BaseStore | None, user_id: str, book_id: str) -> int | None:
    """The spoiler-guard limit for this reader and book (None = no limit)."""
    value = get_reader_profile(store, user_id)["progress"].get(book_id)
    return int(value) if value else None


def set_reading_progress(store: BaseStore, user_id: str, book_id: str, chapter: int | None) -> None:
    """Record how far the reader has read (None or 0 clears the spoiler guard)."""
    profile = get_reader_profile(store, user_id)
    if chapter:
        profile["progress"][book_id] = int(chapter)
    else:
        profile["progress"].pop(book_id, None)
    _save(store, user_id, profile)


def add_preference(store: BaseStore, user_id: str, preference: str, limit: int = 10) -> list[str]:
    profile = get_reader_profile(store, user_id)
    prefs = [p for p in profile["preferences"] if p.lower() != preference.lower()] + [preference.strip()]
    profile["preferences"] = prefs[-limit:]
    _save(store, user_id, profile)
    return profile["preferences"]


def clear_preferences(store: BaseStore, user_id: str) -> None:
    profile = get_reader_profile(store, user_id)
    profile["preferences"] = []
    _save(store, user_id, profile)
