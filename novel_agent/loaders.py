"""Step 1 of ingestion: get the raw text of a book from Gutenberg, a URL or a local file."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import get_settings

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
]
_START_RE = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)


@dataclass
class RawBook:
    text: str
    title: str
    author: str
    source: str


def normalize_text(text: str) -> str:
    """CRLF -> LF, drop BOM / non-breaking spaces, strip trailing whitespace on each line."""
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def strip_gutenberg_boilerplate(text: str) -> tuple[str, dict[str, str]]:
    """Return (body, metadata) — the text between the START/END markers plus Title/Author."""
    meta: dict[str, str] = {}
    start, end = _START_RE.search(text), _END_RE.search(text)
    header = text[: start.start()] if start else text[:5000]
    for key in ("Title", "Author"):
        m = re.search(rf"^{key}:\s*(.+)$", header, re.M)
        if m:
            meta[key.lower()] = m.group(1).strip()
    body = text[start.end() if start else 0 : end.start() if end else len(text)]
    return body.strip("\n"), meta


def fetch_gutenberg(gutenberg_id: int, cache_dir: Path | None = None) -> str:
    """Download a Gutenberg plain-text book (cached under data/raw/)."""
    cache_dir = cache_dir or get_settings().raw_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"pg{gutenberg_id}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    last_error: Exception | None = None
    for template in GUTENBERG_URLS:
        url = template.format(id=gutenberg_id)
        try:
            text = _download(url)
            cached.write_text(text, encoding="utf-8")
            return text
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Gutenberg book #{gutenberg_id}: {last_error}")


def _download(url: str) -> str:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "novel-agent-workshop/1.0"})
    resp.raise_for_status()
    resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else "utf-8"
    return resp.text


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _read_epub(path: Path) -> str:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    parts = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            parts.append(soup.get_text("\n"))
    return "\n\n".join(parts)


def load_book(
    *,
    gutenberg_id: int | None = None,
    url: str | None = None,
    path: str | Path | None = None,
    title: str | None = None,
    author: str | None = None,
) -> RawBook:
    """Load a book from exactly one source and return its cleaned, normalized body text."""
    if sum(x is not None for x in (gutenberg_id, url, path)) != 1:
        raise ValueError("Pass exactly one of gutenberg_id, url or path")

    if gutenberg_id is not None:
        raw, source = fetch_gutenberg(gutenberg_id), f"gutenberg:{gutenberg_id}"
    elif url is not None:
        raw, source = _download(url), url
    else:
        p = Path(path)  # type: ignore[arg-type]
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            raw = _read_pdf(p)
        elif suffix == ".epub":
            raw = _read_epub(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="replace")
        source = str(p)

    body, meta = strip_gutenberg_boilerplate(normalize_text(raw))
    return RawBook(
        text=body,
        title=title or meta.get("title", "Unknown title"),
        author=author or meta.get("author", "Unknown author"),
        source=source,
    )
