"""Step 3 of ingestion: cut chapters into retrieval-sized chunks with rich metadata."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .chapters import Chapter


def chunk_header(title: str, chapter: int) -> str:
    return f"[{title} · Chapter {chapter}]"


def chunk_chapters(
    chapters: list[Chapter],
    *,
    book_id: str,
    title: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split each chapter separately so no chunk ever straddles two chapters.

    Each chunk starts with a short contextual header (book + chapter). It costs a few tokens but
    gives the embedding model and BM25 the context that a bare paragraph of dialogue lacks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    docs: list[Document] = []
    for chapter in chapters:
        for i, piece in enumerate(splitter.split_text(chapter.text)):
            chunk_id = f"ch{chapter.number:03d}-{i:03d}"
            docs.append(
                Document(
                    id=chunk_id,
                    page_content=f"{chunk_header(title, chapter.number)}\n{piece}",
                    metadata={
                        "doc_id": chunk_id,
                        "book_id": book_id,
                        "chapter": chapter.number,
                        "heading": chapter.heading,
                        "chunk_index": i,
                        "position": len(docs),  # global story order, used to sort evidence chronologically
                    },
                )
            )
    return docs


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English prose)."""
    return max(1, len(text) // 4)
