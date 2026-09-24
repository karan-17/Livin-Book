"""Load everything the agent needs for one ingested book (retrievers + knowledge)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import get_embeddings, get_reranker
from .indexing import embedding_identity, index_paths, load_documents, open_vectorstore, read_manifest
from .knowledge import BookKnowledge
from .retrieval import BM25Index, HybridRetriever


@dataclass
class BookResources:
    book_id: str
    title: str
    author: str
    chapter_count: int
    chunks: HybridRetriever  # passage-level retrieval over the full text
    summaries: HybridRetriever | None  # chapter-summary retrieval (needs enrichment)
    knowledge: BookKnowledge | None  # characters, relationships, themes (needs enrichment)


def load_resources(book_id: str, *, embeddings=None, reranker="default") -> BookResources:
    manifest = read_manifest(book_id)
    if not manifest or not manifest.get("embedded_chunks"):
        raise FileNotFoundError(
            f"Book {book_id!r} has not been ingested yet. Run: python -m novel_agent.ingest --book {book_id}"
        )
    if manifest["embeddings"] != embedding_identity():
        raise ValueError(
            f"{book_id!r} was embedded with {manifest['embeddings']} but .env selects {embedding_identity()}. "
            "Switch EMBEDDINGS_PROVIDER back, or re-run ingestion with the new provider."
        )
    paths = index_paths(book_id)
    embeddings = embeddings or get_embeddings()
    reranker = get_reranker() if reranker == "default" else reranker

    chunk_docs = load_documents(paths.chunks)
    chunks = HybridRetriever(
        open_vectorstore(book_id, "chunks", embeddings), BM25Index(chunk_docs), reranker=reranker, name="chunks"
    )

    knowledge = BookKnowledge.load(paths.knowledge)
    summaries = None
    if knowledge is not None and manifest.get("embedded_summaries"):
        from .enrich import summary_documents

        summaries = HybridRetriever(
            open_vectorstore(book_id, "summaries", embeddings),
            BM25Index(summary_documents(knowledge.data)),
            reranker=None,
            fetch_k=10,
            name="summaries",
        )
    return BookResources(
        book_id=book_id,
        title=manifest["title"],
        author=manifest["author"],
        chapter_count=manifest["chapter_count"],
        chunks=chunks,
        summaries=summaries,
        knowledge=knowledge,
    )
