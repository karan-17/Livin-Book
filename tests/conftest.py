"""Shared fixtures: an isolated DATA_DIR and a tiny synthetic book indexed with fake embeddings.

Nothing here calls a real API, so the whole suite runs offline and without keys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    from novel_agent import config

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "voyage")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    config.get_settings.cache_clear()
    yield tmp_path / "data"
    config.get_settings.cache_clear()


TINY_BOOK_CHAPTERS = {
    1: "Elizabeth Bennet meets Mr. Darcy at the Meryton assembly. Darcy refuses to dance with her and calls her "
       "tolerable, but not handsome enough to tempt him. Elizabeth laughs about his pride with her sister Jane.",
    2: "Jane Bennet rides to Netherfield in the rain and falls ill. Elizabeth walks three miles through the mud "
       "to nurse her sister. Darcy admires her fine eyes, while Caroline Bingley mocks her dirty petticoat.",
    3: "Mr. Wickham tells Elizabeth that Darcy cheated him of a living. Elizabeth believes Wickham and her "
       "prejudice against Darcy grows stronger. The Netherfield ball follows.",
    4: "At Hunsford, Darcy proposes to Elizabeth and she refuses him, accusing him of ruining Wickham. The next "
       "day Darcy gives her a letter explaining Wickham's true character and his role in separating Jane and Bingley.",
    5: "Lydia elopes with Wickham. Darcy secretly pays Wickham to marry her. Elizabeth accepts Darcy's second "
       "proposal, and Jane marries Bingley. Pemberley becomes Elizabeth's home.",
}


def _tiny_knowledge(book_id: str) -> dict:
    from novel_agent.enrich import (
        BookProfile, ChapterNotes, CharacterMention, CharacterProfile, RelationshipNote, ThemeNote,
        assemble_knowledge,
    )

    def notes(n, title, rel):
        return ChapterNotes(
            chapter=n, title=title, summary=TINY_BOOK_CHAPTERS[n], key_events=[TINY_BOOK_CHAPTERS[n][:60]],
            characters=[CharacterMention(name="Lizzy", role=f"role in ch {n}"),
                        CharacterMention(name="Mr. Darcy", role=f"Darcy in ch {n}")],
            relationships=[RelationshipNote(source="Elizabeth", target="Darcy", relation=rel, description=rel)],
            themes=["pride"],
        )

    chapter_notes = [
        notes(1, "The assembly snub", "mutual dislike"),
        notes(2, "A walk through the mud", "grudging admiration"),
        notes(3, "Wickham's tale", "prejudice deepens"),
        notes(4, "The first proposal", "rejected proposal"),
        notes(5, "Happy endings", "engaged"),
    ]
    profile = BookProfile(
        synopsis="Elizabeth and Darcy overcome pride and prejudice.",
        ending="Elizabeth marries Darcy; Jane marries Bingley.",
        characters=[
            CharacterProfile(name="Elizabeth Bennet", aliases=["Elizabeth", "Lizzy", "Eliza"], importance="main",
                             description="Witty second Bennet daughter.", arc="Learns to see past first impressions."),
            CharacterProfile(name="Fitzwilliam Darcy", aliases=["Darcy", "Mr. Darcy"], importance="main",
                             description="Proud, wealthy owner of Pemberley.", arc="Learns humility."),
        ],
        themes=[ThemeNote(name="Pride", description="Pride blinds both leads.", key_chapters=[1, 4, 5])],
    )
    return assemble_knowledge(chapter_notes, profile, book_id=book_id, title="Tiny Pride", author="Test Author",
                              chapter_count=5)


@pytest.fixture()
def tiny_resources(data_dir):
    """Index the tiny book exactly like the real pipeline, but with deterministic fake embeddings."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from novel_agent.chapters import Chapter
    from novel_agent.chunking import chunk_chapters
    from novel_agent.enrich import save_knowledge, summary_documents
    from novel_agent.indexing import (
        add_documents_resumable, index_paths, open_vectorstore, save_documents, write_manifest,
    )
    from novel_agent.resources import load_resources

    book_id = "tiny"
    embeddings = DeterministicFakeEmbedding(size=32)
    chapters = [Chapter(n, f"CHAPTER {n}", text) for n, text in TINY_BOOK_CHAPTERS.items()]
    chunks = chunk_chapters(chapters, book_id=book_id, title="Tiny Pride", chunk_size=120, chunk_overlap=20)
    paths = index_paths(book_id)
    save_documents(chunks, paths.chunks)
    add_documents_resumable(open_vectorstore(book_id, "chunks", embeddings), chunks, batch_tokens=500,
                            progress=lambda _: None)
    knowledge = _tiny_knowledge(book_id)
    save_knowledge(knowledge, paths.knowledge)
    add_documents_resumable(open_vectorstore(book_id, "summaries", embeddings), summary_documents(knowledge),
                            batch_tokens=500, progress=lambda _: None)
    write_manifest(book_id, title="Tiny Pride", author="Test Author", source="test", chapter_count=5,
                   chunk_count=len(chunks), embedded_chunks=len(chunks), embedded_summaries=5)
    return load_resources(book_id, embeddings=embeddings, reranker=None)
