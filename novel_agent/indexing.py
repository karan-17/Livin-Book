"""Step 4 of ingestion: embed chunks into a persistent Chroma collection (+ files for BM25).

On-disk layout per book (data/index/<book_id>/):
    chroma/           persistent Chroma DB (chunk + chapter-summary collections)
    chunks.jsonl      every chunk (BM25 is rebuilt from this at load time; it is fast)
    knowledge.json    enrichment output (chapter notes, characters, themes, ...)
    manifest.json     title/author/stats + which embedding model built the vectors
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from .chunking import estimate_tokens
from .config import get_settings


@dataclass(frozen=True)
class IndexPaths:
    root: Path

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def index_paths(book_id: str) -> IndexPaths:
    return IndexPaths(get_settings().index_dir(book_id))


def embedding_identity() -> dict[str, str]:
    s = get_settings()
    return {"provider": s.embeddings_provider, "model": s.embed_model}


def collection_name(book_id: str, kind: str) -> str:
    """e.g. 'pride_and_prejudice__chunks__voyage_voyage-4' (vectors from different models never mix)."""
    ident = embedding_identity()
    raw = f"{book_id}__{kind}__{ident['provider']}_{ident['model']}"
    return re.sub(r"[^a-zA-Z0-9_-]", "-", raw)[:120].strip("-_")  # Chroma: must start/end alphanumeric


def open_vectorstore(book_id: str, kind: str, embeddings):
    from langchain_chroma import Chroma

    paths = index_paths(book_id)
    paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name(book_id, kind),
        embedding_function=embeddings,
        persist_directory=str(paths.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _token_batches(docs: list[Document], max_tokens: int) -> list[list[Document]]:
    batches: list[list[Document]] = []
    current: list[Document] = []
    used = 0
    for doc in docs:
        cost = estimate_tokens(doc.page_content)
        if current and used + cost > max_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(doc)
        used += cost
    if current:
        batches.append(current)
    return batches


def add_documents_resumable(
    vectorstore, docs: list[Document], *, batch_tokens: int, progress: Callable[[str], None] = print
) -> int:
    """Embed + store docs in token-budgeted batches, skipping IDs already stored. Returns #added.

    Re-running after a crash / rate-limit abort continues where it stopped instead of paying again.
    """
    existing = set(vectorstore.get(include=[])["ids"])
    todo = [d for d in docs if d.metadata["doc_id"] not in existing]
    if not todo:
        progress(f"  all {len(docs)} documents already embedded")
        return 0
    batches = _token_batches(todo, batch_tokens)
    for i, batch in enumerate(batches, start=1):
        vectorstore.add_documents(batch, ids=[d.metadata["doc_id"] for d in batch])
        progress(f"  embedded batch {i}/{len(batches)} ({len(batch)} docs)")
    return len(todo)


def save_documents(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False))
            f.write("\n")


def load_documents(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]


def write_manifest(book_id: str, **fields) -> dict:
    paths = index_paths(book_id)
    manifest = read_manifest(book_id) or {}
    manifest.update(fields)
    manifest.update(book_id=book_id, embeddings=embedding_identity(),
                    updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def read_manifest(book_id: str) -> dict | None:
    path = index_paths(book_id).manifest
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def list_ingested_books() -> list[dict]:
    root = get_settings().data_dir / "index"
    if not root.exists():
        return []
    books = []
    for manifest in sorted(root.glob("*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("embedded_chunks"):
            books.append(data)
    return books
