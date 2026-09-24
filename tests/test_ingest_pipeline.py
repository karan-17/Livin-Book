"""Runs the real ingestion CLI end to end with fake embeddings and a fake enrichment LLM."""

import re

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

from novel_agent import ingest as ingest_module
from novel_agent.config import ResilientEmbeddings
from novel_agent.enrich import (
    BookProfile, ChapterBatch, ChapterNotes, CharacterMention, CharacterProfile, RelationshipNote, ThemeNote,
)
from novel_agent.resources import load_resources

WORDS = " ".join(["lorem"] * 120)
BOOK = "\n\n".join(
    f"CHAPTER {n}\n\n{name} walks to Meryton and talks with Jane about the ball. {WORDS}"
    for n, name in [(1, "Elizabeth"), (2, "Lizzy"), (3, "Elizabeth"), (4, "Eliza")]
)


class FakeEnrichLLM:
    """Structured-output stand-in. Drops the last chapter of every batch to exercise the retry path."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **_):
        def respond(messages):
            self.calls += 1
            human = messages[-1][1]
            if schema is ChapterBatch:
                numbers = [int(n) for n in re.findall(r"=== CHAPTER (\d+) ===", human)]
                keep = numbers[:-1] if len(numbers) > 1 else numbers
                return ChapterBatch(chapters=[
                    ChapterNotes(chapter=n, title=f"Title {n}", summary=f"Summary of chapter {n}.",
                                 key_events=[f"event {n}"],
                                 characters=[CharacterMention(name="Lizzy", role="walks")],
                                 relationships=[RelationshipNote(source="Eliza", target="Jane", relation="sisters",
                                                                 description="close")],
                                 themes=["sisterhood"])
                    for n in keep
                ])
            return BookProfile(
                synopsis="Two sisters walk to Meryton.", ending="They walk home.",
                characters=[CharacterProfile(name="Elizabeth Bennet", aliases=["Lizzy", "Eliza", "Elizabeth"],
                                             importance="main", description="A walker.", arc="Walks more.")],
                themes=[ThemeNote(name="Sisterhood", description="Sisters.", key_chapters=[1, 2])],
            )

        return RunnableLambda(respond)


def test_ingest_cli_end_to_end(data_dir, tmp_path, monkeypatch, capsys):
    embeddings = ResilientEmbeddings(DeterministicFakeEmbedding(size=32))
    llm = FakeEnrichLLM()
    monkeypatch.setattr(ingest_module, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(ingest_module, "get_llm", lambda role: llm)
    monkeypatch.setenv("CHAPTERS_PER_CALL", "2")
    from novel_agent import config

    config.get_settings.cache_clear()

    book = tmp_path / "mini.txt"
    book.write_text(BOOK, encoding="utf-8")
    argv = ["--file", str(book), "--book-id", "mini", "--title", "Mini", "--author", "Anon"]
    ingest_module.main(argv)
    out = capsys.readouterr().out
    assert "4 chapters" in out and "Done" in out
    assert llm.calls == 2 + 2 + 1  # 2 batches, 2 single-chapter retries, 1 reduce

    res = load_resources("mini", embeddings=embeddings, reranker=None)
    assert res.chapter_count == 4 and res.knowledge is not None and res.summaries is not None
    assert res.knowledge.appearances == {"Elizabeth Bennet": [1, 2, 3, 4]}
    assert "Jane" in res.knowledge.relationship("Lizzy")
    assert res.chunks.search("Meryton ball", k=2)

    # Re-running is cheap: chunks are already embedded and the enrichment is cached.
    ingest_module.main(argv)
    out = capsys.readouterr().out
    assert "already embedded" in out and "Enrichment cached" in out
    assert llm.calls == 5
