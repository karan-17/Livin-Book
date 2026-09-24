"""Ingestion pipeline CLI.

    python -m novel_agent.ingest --book pride_and_prejudice            # full pipeline
    python -m novel_agent.ingest --book pride_and_prejudice --dry-run  # no API calls: inspect chapters/chunks
    python -m novel_agent.ingest --file my_novel.epub --book-id my_novel --title "..." --author "..."
    python -m novel_agent.ingest --gutenberg 84 --book-id frankenstein

Steps: load -> clean & split chapters -> chunk -> embed into Chroma -> enrich (LLM map-reduce)
-> embed chapter summaries -> write manifest. Every step is cached/resumable, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from langsmith import traceable

from .books import GENERIC_CHAPTER_PATTERN, PRESETS
from .chapters import Chapter, split_chapters
from .chunking import chunk_chapters, estimate_tokens
from .config import get_embeddings, get_llm, get_settings
from .enrich import enrich_book, save_knowledge, summary_documents
from .indexing import add_documents_resumable, index_paths, open_vectorstore, save_documents, write_manifest
from .loaders import RawBook, load_book


def resolve_source(args) -> tuple[str, RawBook, str]:
    """Returns (book_id, raw book, chapter regex) from CLI args."""
    if args.book and args.book in PRESETS and not (args.file or args.url or args.gutenberg):
        p = PRESETS[args.book]
        return p.book_id, load_book(gutenberg_id=p.gutenberg_id, title=p.title, author=p.author), p.chapter_pattern
    pattern = args.chapter_regex or GENERIC_CHAPTER_PATTERN
    if args.gutenberg:
        raw = load_book(gutenberg_id=args.gutenberg, title=args.title, author=args.author)
    elif args.url:
        raw = load_book(url=args.url, title=args.title, author=args.author)
    elif args.file:
        raw = load_book(path=args.file, title=args.title, author=args.author)
    else:
        sys.exit(f"Unknown --book {args.book!r}. Presets: {', '.join(PRESETS)}; or use --file/--url/--gutenberg.")
    book_id = args.book_id or args.book or re.sub(r"[^a-z0-9]+", "_", raw.title.lower()).strip("_")
    return book_id, raw, pattern


def print_stats(chapters: list[Chapter], chunks) -> None:
    words = sum(c.word_count for c in chapters)
    tokens = sum(estimate_tokens(d.page_content) for d in chunks)
    print(f"  {len(chapters)} chapters, {words:,} words -> {len(chunks)} chunks (~{tokens:,} tokens to embed)")
    for c in chapters[:3] + (chapters[-2:] if len(chapters) > 5 else []):
        first_line = c.text.strip().split("\n", 1)[0][:70]
        print(f"    #{c.number:<3} {c.heading:<16} {c.word_count:>6,} words | {first_line}")


@traceable(name="ingest_book", run_type="chain")
def ingest(args) -> None:
    settings = get_settings()
    t0 = time.time()
    book_id, raw, pattern = resolve_source(args)
    print(f"[1/5] Loaded '{raw.title}' by {raw.author} ({raw.source}) as book_id={book_id}")

    chapters = split_chapters(raw.text, pattern)
    chunks = chunk_chapters(chapters, book_id=book_id, title=raw.title,
                            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    print("[2/5] Split into chapters and chunks")
    print_stats(chapters, chunks)
    if args.dry_run:
        print("Dry run: stopping before any API calls.")
        return

    paths = index_paths(book_id)
    save_documents(chunks, paths.chunks)
    write_manifest(book_id, title=raw.title, author=raw.author, source=raw.source,
                   chapter_count=len(chapters), chunk_count=len(chunks))

    embeddings = get_embeddings()
    print(f"[3/5] Embedding chunks with {settings.embeddings_provider}:{settings.embed_model} -> Chroma")
    chunk_store = open_vectorstore(book_id, "chunks", embeddings)
    add_documents_resumable(chunk_store, chunks, batch_tokens=settings.embed_batch_tokens)
    write_manifest(book_id, embedded_chunks=len(chunks))

    if args.skip_enrich:
        print("[4/5] Skipping enrichment (--skip-enrich): the agent will rely on passage search only.")
    else:
        if paths.knowledge.exists() and not args.force:
            print(f"[4/5] Enrichment cached at {paths.knowledge} (use --force to rebuild)")
            import json

            knowledge = json.loads(paths.knowledge.read_text(encoding="utf-8"))
        else:
            print(f"[4/5] Enriching with {settings.enrich_model} "
                  f"({settings.chapters_per_call} chapters per call)")
            knowledge = enrich_book(chapters, book_id=book_id, title=raw.title, author=raw.author,
                                    llm=get_llm("enrich"), chapters_per_call=settings.chapters_per_call)
            save_knowledge(knowledge, paths.knowledge)
            print(f"  {len(knowledge['profile']['characters'])} characters, "
                  f"{len(knowledge['profile']['themes'])} themes -> {paths.knowledge}")
        print("[5/5] Embedding chapter summaries")
        summaries = summary_documents(knowledge)
        summary_store = open_vectorstore(book_id, "summaries", embeddings)
        add_documents_resumable(summary_store, summaries, batch_tokens=settings.embed_batch_tokens)
        write_manifest(book_id, embedded_summaries=len(summaries), enriched=True)

    print(f"Done in {time.time() - t0:.0f}s. Chat with it: python -m novel_agent.chat --book {book_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a book for the novel agent.")
    parser.add_argument("--book", help=f"Preset ({', '.join(PRESETS)}) or the book_id to use")
    parser.add_argument("--file", type=Path, help="Local .txt, .pdf or .epub")
    parser.add_argument("--url", help="URL of a plain-text book")
    parser.add_argument("--gutenberg", type=int, help="Project Gutenberg ebook number")
    parser.add_argument("--book-id", help="Identifier for a custom book (default: from the title)")
    parser.add_argument("--title")
    parser.add_argument("--author")
    parser.add_argument("--chapter-regex", help="Regex matching chapter heading lines (multiline mode)")
    parser.add_argument("--dry-run", action="store_true", help="Only load/split/chunk; no API calls")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip the LLM enrichment step")
    parser.add_argument("--force", action="store_true", help="Rebuild enrichment even if cached")
    args = parser.parse_args(argv)
    if not (args.book or args.file or args.url or args.gutenberg):
        parser.error("choose a source: --book, --file, --url or --gutenberg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ingest(args)


if __name__ == "__main__":
    main()
