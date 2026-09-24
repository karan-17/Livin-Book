"""Built-in book presets (public-domain texts from Project Gutenberg)."""

from __future__ import annotations

from dataclasses import dataclass

# Matches "CHAPTER I", "Chapter 12.", "CHAPTER IV. The Ball", "Chapter I.]" (tail of an illustration block).
GENERIC_CHAPTER_PATTERN = r"^[ \t]*(?:CHAPTER|Chapter)[ \t]+(?:[IVXLCDM]+|\d+)\b[.:]?\]?[^\n]{0,80}$"


@dataclass(frozen=True)
class BookPreset:
    book_id: str
    title: str
    author: str
    gutenberg_id: int
    chapter_pattern: str = GENERIC_CHAPTER_PATTERN


PRESETS: dict[str, BookPreset] = {
    p.book_id: p
    for p in [
        BookPreset(
            book_id="pride_and_prejudice",
            title="Pride and Prejudice",
            author="Jane Austen",
            gutenberg_id=1342,
            # #1342 quirks: chapter 1 is "Chapter I.]" (end of an illustration block), chapters 13-14
            # have no trailing period. This pattern yields exactly 61 chapters.
            chapter_pattern=r"^(?:CHAPTER|Chapter) ([IVXLC]+)\.?\]?[ \t]*$",
        ),
        BookPreset(
            book_id="frankenstein",
            title="Frankenstein; Or, The Modern Prometheus",
            author="Mary Wollstonecraft Shelley",
            gutenberg_id=84,
            chapter_pattern=r"^[ \t]*(?:Letter|Chapter)[ \t]+\d+[ \t]*$",
        ),
        BookPreset(
            book_id="hound_of_the_baskervilles",
            title="The Hound of the Baskervilles",
            author="Arthur Conan Doyle",
            gutenberg_id=2852,
        ),
    ]
}
