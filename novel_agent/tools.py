"""The agent's tools. Each tool's docstring is what the LLM reads to decide when to call it.

Two design rules:
  * Arguments are flat primitives (str / int). Gemini handles these most reliably.
  * The spoiler guard is enforced HERE, in code, not just requested in the prompt: every read
    tool looks up the reader's progress in the long-term store and filters later chapters out.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.tools import tool

from . import memory
from .resources import BookResources

DEFAULT_USER = "workshop-user"


@dataclass
class Context:
    """Run-scoped context passed to every graph run (not persisted in the thread): who is asking."""

    user_id: str = DEFAULT_USER


BookToolRuntime = ToolRuntime[Context]  # gives tools typed access to store + context
NO_KNOWLEDGE = "Enrichment has not been run for this book, so this tool is unavailable. Use search_book instead."


def _passage_text(doc: Document) -> str:
    """Chunk text without its '[Title · Chapter N]' header line."""
    text = doc.page_content
    return text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text


def make_tools(res: BookResources) -> list:
    knowledge = res.knowledge
    last_chapter = res.chapter_count

    def user_id(runtime: BookToolRuntime) -> str:
        return getattr(runtime.context, "user_id", None) or DEFAULT_USER

    def max_chapter(runtime: BookToolRuntime) -> int | None:
        return memory.get_max_chapter(runtime.store, user_id(runtime), res.book_id)

    def spoiler_block(chapter: int, limit: int | None) -> str | None:
        if limit is not None and chapter > limit:
            return (f"SPOILER GUARD: the reader has only read up to chapter {limit}. "
                    f"Chapter {chapter} is off-limits; tell the reader this would be a spoiler.")
        return None

    @tool(response_format="content_and_artifact")
    def search_book(query: str, runtime: BookToolRuntime, chapter_from: int = 0, chapter_to: int = 0):
        """Search the full text of the novel for passages relevant to `query`.

        Hybrid keyword + semantic search, so include specific names, places and events
        (e.g. "Darcy letter to Elizabeth explaining Wickham"). Optionally restrict to a chapter
        range with chapter_from / chapter_to (0 means no limit). Returns up to 6 passages,
        labelled with their chapter, in story order.
        """
        limit = max_chapter(runtime)
        lo = chapter_from or None
        hi = chapter_to or None
        if lo and (blocked := spoiler_block(lo, limit)):
            return blocked, []
        guard_note = ""
        if limit is not None and (hi is None or hi > limit):
            hi, guard_note = limit, f"\n(Spoiler guard: searched chapters 1-{limit} only.)"
        docs = res.chunks.search(query, k=6, chapter_from=lo, chapter_to=hi)
        if not docs:
            return "No matching passages found. Try different keywords." + guard_note, []
        docs = sorted(docs, key=lambda d: d.metadata.get("position", 0))
        blocks = [f"[Ch. {d.metadata['chapter']} | {d.metadata['doc_id']}]\n{_passage_text(d)}" for d in docs]
        sources = [
            {"chapter": d.metadata["chapter"], "chunk_id": d.metadata["doc_id"], "text": _passage_text(d),
             "score": d.metadata.get("relevance_score", d.metadata.get("rrf_score"))}
            for d in docs
        ]
        return f"{len(docs)} passages (story order):\n\n" + "\n\n---\n\n".join(blocks) + guard_note, sources

    @tool
    def find_chapters(query: str, runtime: BookToolRuntime) -> str:
        """Find which chapters an event, scene or topic occurs in (e.g. "Lydia elopes with Wickham",
        "the Netherfield ball"). Returns the best-matching chapters with short summaries.
        Use this to locate events before reading details with search_book or get_chapter_summaries."""
        limit = max_chapter(runtime)
        if res.summaries is None:  # no enrichment: aggregate passage hits by chapter instead
            docs = res.chunks.search(query, k=10, chapter_to=limit, use_rerank=False)
            chapters = sorted({d.metadata["chapter"] for d in docs})
            return f"Chapters with matching passages: {', '.join(map(str, chapters)) or 'none'}"
        docs = res.summaries.search(query, k=5, chapter_to=limit, use_rerank=False)
        lines = []
        for d in docs:
            summary = _passage_text(d).split("\nKey events:")[0]
            lines.append(f"Chapter {d.metadata['chapter']} — {d.metadata['title']}: {summary}")
        guard = f"\n(Spoiler guard: only chapters 1-{limit} considered.)" if limit else ""
        return ("\n\n".join(lines) or "No matching chapters.") + guard

    @tool
    def get_chapter_summaries(chapter_from: int, runtime: BookToolRuntime, chapter_to: int = 0) -> str:
        """Get summaries, key events, characters and themes for a chapter or a range of chapters
        (at most 8 at a time). Use for "what happens in chapter N" or to review a stretch of the story."""
        if knowledge is None:
            return NO_KNOWLEDGE
        chapter_to = chapter_to or chapter_from
        if not (1 <= chapter_from <= last_chapter):
            return f"Chapters run from 1 to {last_chapter}."
        chapter_to = max(chapter_from, min(chapter_to, chapter_from + 7, last_chapter))
        return knowledge.chapter_summaries(chapter_from, chapter_to, max_chapter(runtime))

    @tool
    def get_character_profile(name: str, runtime: BookToolRuntime) -> str:
        """Get a character's profile: aliases, description, arc across the novel, the chapters they
        appear in and their role in each. Accepts nicknames and partial names (e.g. "Lizzy", "Darcy")."""
        if knowledge is None:
            return NO_KNOWLEDGE
        return knowledge.character_profile(name, max_chapter(runtime))

    @tool
    def get_relationship(character_a: str, runtime: BookToolRuntime, character_b: str = "") -> str:
        """Trace a relationship chapter by chapter. With two characters, returns the timeline of how
        their relationship develops. With one character, lists who they interact with most."""
        if knowledge is None:
            return NO_KNOWLEDGE
        return knowledge.relationship(character_a, character_b, max_chapter(runtime))

    @tool
    def get_book_overview(runtime: BookToolRuntime) -> str:
        """Get the big picture: synopsis, how the book ends, major themes (with key chapters), main
        characters and a chapter-by-chapter guide. Use for questions about themes, the ending, the
        overall plot, or to orient yourself before a detailed search."""
        if knowledge is None:
            return NO_KNOWLEDGE
        return knowledge.overview(max_chapter(runtime))

    @tool
    def set_reading_progress(chapter: int, runtime: BookToolRuntime) -> str:
        """Remember how far the reader has read (enables the spoiler guard across all conversations).
        Use when the reader says e.g. "I've just finished chapter 20". Pass 0 if they have finished
        the book or want spoilers allowed again."""
        if chapter < 0 or chapter > last_chapter:
            return f"Chapter must be between 0 and {last_chapter}."
        store = runtime.store
        memory.set_reading_progress(store, user_id(runtime), res.book_id, chapter or None)
        if chapter and chapter < last_chapter:
            return f"Saved: the reader has read up to chapter {chapter}. Spoiler guard is ON for later chapters."
        return "Saved: spoiler guard is OFF (the reader has finished the book)."

    @tool
    def save_preference(preference: str, runtime: BookToolRuntime) -> str:
        """Remember a lasting preference of the reader for all future conversations
        (e.g. "keep answers under 100 words", "always include a quote")."""
        prefs = memory.add_preference(runtime.store, user_id(runtime), preference)
        return f"Saved. Current preferences: {'; '.join(prefs)}"

    return [
        search_book,
        find_chapters,
        get_chapter_summaries,
        get_character_profile,
        get_relationship,
        get_book_overview,
        set_reading_progress,
        save_preference,
    ]
