"""Step 5 of ingestion: LLM "enrichment" — a map-reduce pass that turns raw chapters into knowledge.

    map    : batches of chapters  -> ChapterNotes (summary, events, characters, relationships, themes)
    reduce : all chapter notes    -> BookProfile  (synopsis, ending, characters + aliases + arcs, themes)
    python : alias resolution, per-character chapter appearances, relationship timelines

This gives the agent a second, *hierarchical* view of the book: chunk-level RAG answers detail
questions, while summaries/profiles answer "big picture" questions (themes, arcs, the ending)
that no single chunk contains.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import prompts
from .chapters import Chapter

log = logging.getLogger("novel_agent")


# --------------------------------------------------------------------------------------
# Structured-output schemas
# --------------------------------------------------------------------------------------


class CharacterMention(BaseModel):
    name: str = Field(description="Most complete name used in the book, e.g. 'Elizabeth Bennet'")
    role: str = Field(description="What this character does or reveals in this chapter (one sentence)")


class RelationshipNote(BaseModel):
    source: str = Field(description="Character name")
    target: str = Field(description="Character name")
    relation: str = Field(description="Short label, e.g. 'sisters', 'courtship', 'mutual dislike', 'friendship'")
    description: str = Field(description="How the relationship stands or changes in this chapter (one sentence)")


class ChapterNotes(BaseModel):
    chapter: int = Field(description="Chapter number exactly as given in the '=== CHAPTER N ===' delimiter")
    title: str = Field(description="Short descriptive title (4-8 words)")
    summary: str = Field(description="120-180 word summary")
    key_events: list[str] = Field(description="2-6 concrete plot events, in order")
    characters: list[CharacterMention]
    relationships: list[RelationshipNote]
    themes: list[str] = Field(description="1-4 themes developed in this chapter")


class ChapterBatch(BaseModel):
    chapters: list[ChapterNotes]


class CharacterProfile(BaseModel):
    name: str = Field(description="Canonical full name")
    aliases: list[str] = Field(description="Other names/titles used for this character (unambiguous only)")
    importance: Literal["main", "secondary", "minor"]
    description: str
    arc: str = Field(description="How the character changes over the novel, citing chapter numbers")


class ThemeNote(BaseModel):
    name: str
    description: str
    key_chapters: list[int]


class BookProfile(BaseModel):
    synopsis: str
    ending: str
    characters: list[CharacterProfile]
    themes: list[ThemeNote]


# --------------------------------------------------------------------------------------
# Map step
# --------------------------------------------------------------------------------------


def _format_chapters(chapters: list[Chapter]) -> str:
    return "\n\n".join(f"=== CHAPTER {c.number} ===\n{c.text}" for c in chapters)


def summarize_chapters(
    chapters: list[Chapter],
    *,
    title: str,
    author: str,
    llm,
    chapters_per_call: int = 8,
    max_concurrency: int = 2,
    progress: Callable[[str], None] = print,
) -> list[ChapterNotes]:
    """Map step: several chapters per LLM call (fewer requests = friendlier to free-tier quotas)."""
    structured = llm.with_structured_output(ChapterBatch)
    batches = [chapters[i : i + chapters_per_call] for i in range(0, len(chapters), chapters_per_call)]

    def messages_for(batch: list[Chapter]):
        return [
            ("system", prompts.ENRICH_MAP_SYSTEM),
            ("human", prompts.ENRICH_MAP_USER.format(title=title, author=author, chapters=_format_chapters(batch))),
        ]

    progress(f"  summarizing {len(chapters)} chapters in {len(batches)} LLM calls ...")
    results = structured.batch(
        [messages_for(b) for b in batches],
        config={"max_concurrency": max_concurrency, "run_name": "enrich_map"},
        return_exceptions=True,
    )

    notes: dict[int, ChapterNotes] = {}
    for batch, result in zip(batches, results):
        wanted = {c.number for c in batch}
        if isinstance(result, Exception):
            log.warning("Batch %s failed: %s", sorted(wanted), result)
        elif result is not None:
            for n in result.chapters:
                if n.chapter in wanted:
                    notes[n.chapter] = n

    # Retry anything missing one chapter at a time (malformed output, dropped chapters, errors).
    by_number = {c.number: c for c in chapters}
    missing = [n for n in by_number if n not in notes]
    for number in missing:
        progress(f"  retrying chapter {number} on its own ...")
        try:
            result = structured.invoke(messages_for([by_number[number]]), config={"run_name": "enrich_map_retry"})
        except Exception as exc:
            log.warning("Chapter %s failed again (%s); it will have no notes", number, exc)
            continue
        for n in result.chapters:
            if n.chapter == number:
                notes[number] = n
        if number not in notes and result.chapters:  # model renumbered it; trust position
            notes[number] = result.chapters[0].model_copy(update={"chapter": number})

    progress(f"  got notes for {len(notes)}/{len(chapters)} chapters")
    return [notes[n] for n in sorted(notes)]


# --------------------------------------------------------------------------------------
# Reduce step
# --------------------------------------------------------------------------------------


def _compact_notes(notes: list[ChapterNotes]) -> str:
    lines = []
    for n in notes:
        chars = ", ".join(c.name for c in n.characters)
        rels = "; ".join(f"{r.source}–{r.target} ({r.relation})" for r in n.relationships)
        lines.append(
            f"Ch {n.chapter} — {n.title}\n  Summary: {n.summary}\n  Characters: {chars}\n"
            f"  Relationships: {rels}\n  Themes: {', '.join(n.themes)}"
        )
    return "\n".join(lines)


def build_book_profile(notes: list[ChapterNotes], *, title: str, author: str, llm) -> BookProfile:
    structured = llm.with_structured_output(BookProfile)
    return structured.invoke(
        [
            ("system", prompts.ENRICH_REDUCE_SYSTEM),
            ("human", prompts.ENRICH_REDUCE_USER.format(
                title=title, author=author, chapter_count=len(notes), notes=_compact_notes(notes))),
        ],
        config={"run_name": "enrich_reduce"},
    )


# --------------------------------------------------------------------------------------
# Deterministic post-processing (no LLM): canonical names, appearances, relationship timelines
# --------------------------------------------------------------------------------------


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", name.lower())).strip()


def build_alias_map(profile: BookProfile) -> dict[str, str]:
    """lower-cased alias -> canonical name. Aliases claimed by two characters are dropped."""
    claims: dict[str, set[str]] = {}
    for ch in profile.characters:
        for alias in [ch.name, *ch.aliases]:
            claims.setdefault(_norm(alias), set()).add(ch.name)
    return {alias: next(iter(names)) for alias, names in claims.items() if len(names) == 1 and alias}


def resolve_name(name: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(_norm(name), name.strip())


def assemble_knowledge(
    notes: list[ChapterNotes], profile: BookProfile, *, book_id: str, title: str, author: str, chapter_count: int
) -> dict:
    """Combine map + reduce output into the knowledge.json structure used by the agent's tools."""
    alias_map = build_alias_map(profile)
    appearances: dict[str, set[int]] = {}
    chapters = []
    for n in notes:
        data = n.model_dump()
        for c in data["characters"]:
            c["name"] = resolve_name(c["name"], alias_map)
            appearances.setdefault(c["name"], set()).add(n.chapter)
        for r in data["relationships"]:
            r["source"] = resolve_name(r["source"], alias_map)
            r["target"] = resolve_name(r["target"], alias_map)
        chapters.append(data)
    return {
        "book_id": book_id,
        "title": title,
        "author": author,
        "chapter_count": chapter_count,
        "chapters": chapters,
        "profile": profile.model_dump(),
        "alias_map": alias_map,
        "appearances": {name: sorted(chs) for name, chs in appearances.items()},
    }


def enrich_book(
    chapters: list[Chapter],
    *,
    book_id: str,
    title: str,
    author: str,
    llm,
    chapters_per_call: int = 8,
    progress: Callable[[str], None] = print,
) -> dict:
    """Full enrichment pipeline -> knowledge dict (see `assemble_knowledge`)."""
    notes = summarize_chapters(
        chapters, title=title, author=author, llm=llm, chapters_per_call=chapters_per_call, progress=progress
    )
    progress("  building the book profile (characters, themes, synopsis, ending) ...")
    profile = build_book_profile(notes, title=title, author=author, llm=llm)
    return assemble_knowledge(
        notes, profile, book_id=book_id, title=title, author=author, chapter_count=len(chapters)
    )


def summary_documents(knowledge: dict):
    """One Document per chapter summary, indexed so `find_chapters` can locate events semantically."""
    from langchain_core.documents import Document

    docs = []
    for ch in knowledge["chapters"]:
        doc_id = f"summary-ch{ch['chapter']:03d}"
        text = (
            f"[{knowledge['title']} · Chapter {ch['chapter']}: {ch['title']}]\n{ch['summary']}\n"
            f"Key events: {' '.join(ch['key_events'])}\n"
            f"Characters: {', '.join(c['name'] for c in ch['characters'])}"
        )
        docs.append(Document(id=doc_id, page_content=text,
                             metadata={"doc_id": doc_id, "chapter": ch["chapter"], "title": ch["title"]}))
    return docs


def save_knowledge(knowledge: dict, path) -> None:
    path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
