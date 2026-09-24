"""Hybrid retrieval: dense (vectors) + sparse (BM25) -> Reciprocal Rank Fusion -> optional rerank.

Why hybrid for novels? Dense embeddings capture meaning ("the moment she realises she was
wrong"), but character and place names ("Wickham", "Pemberley") are rare tokens where keyword
search is far more precise. Fusing both ranked lists gets the best of each.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterable

from langchain_core.documents import Document
from langsmith import traceable
from rank_bm25 import BM25Okapi

log = logging.getLogger("novel_agent")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her hers him his how i in is it its "
    "me my no not of on or our she so that the their them then there they this to was we were what when "
    "where which who whom why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords or 1-letter tokens ("Darcy's" -> ["darcy"])."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def in_chapter_range(doc: Document, chapter_from: int | None, chapter_to: int | None) -> bool:
    chapter = doc.metadata.get("chapter", 0)
    return (not chapter_from or chapter >= chapter_from) and (not chapter_to or chapter <= chapter_to)


def chroma_chapter_filter(chapter_from: int | None, chapter_to: int | None) -> dict | None:
    conditions: list[dict] = []
    if chapter_from:
        conditions.append({"chapter": {"$gte": chapter_from}})
    if chapter_to:
        conditions.append({"chapter": {"$lte": chapter_to}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class BM25Index:
    """Keyword index over a list of Documents, with chapter-range filtering."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None

    def search(
        self, query: str, k: int = 20, chapter_from: int | None = None, chapter_to: int | None = None
    ) -> list[Document]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if in_chapter_range(self.docs[i], chapter_from, chapter_to):
                results.append(self.docs[i])
                if len(results) == k:
                    break
        return results


def doc_key(doc: Document) -> str:
    return doc.metadata.get("doc_id") or doc.id or doc.page_content[:80]


def reciprocal_rank_fusion(ranked_lists: Iterable[list[Document]], k: int = 60) -> list[Document]:
    """Merge ranked lists: score(d) = sum over lists of 1 / (k + rank). Robust, no score calibration needed."""
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [
        Document(id=by_key[key].id, page_content=by_key[key].page_content,
                 metadata={**by_key[key].metadata, "rrf_score": round(scores[key], 5)})
        for key in ordered
    ]


def _as_trace(docs: list[Document]) -> list[dict[str, Any]]:
    return [{"page_content": d.page_content, "metadata": d.metadata, "type": "Document"} for d in docs]


class HybridRetriever:
    """Dense + BM25 retrieval over one collection, fused with RRF and optionally reranked."""

    def __init__(self, vectorstore, bm25: BM25Index, *, reranker=None, fetch_k: int = 20, name: str = "chunks"):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.name = name

    def dense(self, query: str, k: int = 20, chapter_from: int | None = None, chapter_to: int | None = None):
        where = chroma_chapter_filter(chapter_from, chapter_to)
        for attempt in range(1, 4):
            try:
                return self.vectorstore.similarity_search(query, k=k, filter=where)
            except Exception as exc:
                # Local Chroma occasionally fails to open a just-written HNSW index ("Nothing found on
                # disk"); it recovers on retry. Other errors (e.g. embedding API auth) still raise.
                if not type(exc).__module__.startswith("chromadb"):
                    raise
                if attempt == 3:
                    log.warning("Vector search failed (%s); using keyword search only for this query", exc)
                    return []
                time.sleep(0.25 * attempt)
        return []

    def sparse(self, query: str, k: int = 20, chapter_from: int | None = None, chapter_to: int | None = None):
        return self.bm25.search(query, k=k, chapter_from=chapter_from, chapter_to=chapter_to)

    def hybrid(self, query: str, k: int = 20, chapter_from: int | None = None, chapter_to: int | None = None):
        dense = self.dense(query, self.fetch_k, chapter_from, chapter_to)
        sparse = self.sparse(query, self.fetch_k, chapter_from, chapter_to)
        return reciprocal_rank_fusion([dense, sparse])[:k]

    def rerank(self, query: str, docs: list[Document], k: int) -> list[Document]:
        if not self.reranker or not docs:
            return docs[:k]
        try:
            return list(self.reranker.compress_documents(docs, query))[:k]
        except Exception as exc:  # rate limit, network, ... -> degrade gracefully to RRF order
            log.warning("Reranker failed (%s); using RRF order instead", exc)
            return docs[:k]

    def search(
        self,
        query: str,
        k: int = 6,
        chapter_from: int | None = None,
        chapter_to: int | None = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        """The full pipeline: hybrid candidates -> rerank -> top k (most relevant first)."""
        return _traced_search(self, query, k, chapter_from, chapter_to, use_rerank)


@traceable(
    run_type="retriever",
    name="hybrid_search",
    process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "retriever"},
    process_outputs=lambda docs: {"documents": _as_trace(docs or [])},
)
def _traced_search(
    retriever: HybridRetriever,
    query: str,
    k: int,
    chapter_from: int | None,
    chapter_to: int | None,
    use_rerank: bool,
) -> list[Document]:
    candidates = retriever.hybrid(query, retriever.fetch_k, chapter_from, chapter_to)
    return retriever.rerank(query, candidates, k) if use_rerank else candidates[:k]
