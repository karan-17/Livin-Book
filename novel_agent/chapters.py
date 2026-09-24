"""Step 2 of ingestion: split the body text into clean chapters.

Chapters are the natural unit of a novel: every chunk, summary and citation is tied to one,
which is what lets the agent answer "what happens in chapter 12?" or filter out spoilers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .books import GENERIC_CHAPTER_PATTERN


@dataclass
class Chapter:
    number: int  # sequential 1..N (robust to editions that restart numbering per volume)
    heading: str  # the heading as printed, e.g. "CHAPTER XXXIV."
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def remove_bracketed_blocks(text: str, opener: str = "[Illustration") -> str:
    """Remove `[Illustration ...]` blocks, respecting nested brackets like `[_Copyright 1894._]`.

    A block that is never closed is removed only up to the end of its paragraph, so a stray
    bracket can never swallow the rest of a chapter.
    """
    out: list[str] = []
    i = 0
    while True:
        start = text.find(opener, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        depth, j = 0, start
        while j < len(text):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            i = j + 1
        else:  # unclosed: drop to the end of the paragraph
            para_end = text.find("\n\n", start)
            i = len(text) if para_end == -1 else para_end
    return "".join(out)


def _trim_trailing_noise(text: str) -> str:
    """Drop short ALL-CAPS paragraphs at the very end (printer's colophons, 'THE END')."""
    paras = text.rstrip().split("\n\n")
    while len(paras) > 1:
        last = paras[-1].strip()
        if last and not re.search(r"[a-z]", last) and len(last.split()) < 25:
            paras.pop()
        else:
            break
    return "\n\n".join(paras)


def clean_body(text: str) -> str:
    text = remove_bracketed_blocks(text)
    text = re.sub(r"^[ \t]*[\[\]][ \t]*$", "", text, flags=re.M)  # stray bracket-only lines
    text = re.sub(r"\^\{([^}]*)\}", r"\1", text)  # M^{r.} -> Mr.
    text = re.sub(r"_([^_]+?)_", r"\1", text)  # _italics_ -> italics
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _trim_trailing_noise(text).strip()


def split_chapters(
    text: str,
    pattern: str = GENERIC_CHAPTER_PATTERN,
    *,
    min_words: int = 100,
    fallback_section_words: int = 3000,
) -> list[Chapter]:
    """Split a book body into chapters using a heading regex.

    - Text before the first real chapter (preface, table of contents) is dropped.
    - Headings followed by fewer than `min_words` words are table-of-contents entries: before the
      first real chapter they are discarded, later they are merged into the previous chapter.
    - If no headings are found, the text is cut into fixed-size "sections" instead.
    """
    matches = list(re.finditer(pattern, text, flags=re.M))
    sections = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = clean_body(text[m.end() : end])
        sections.append((m.group(0).strip(), body))

    chapters: list[Chapter] = []
    for heading, body in sections:
        if len(body.split()) >= min_words:
            chapters.append(Chapter(number=len(chapters) + 1, heading=heading, text=body))
        elif chapters:  # a short section after the story started: keep its text
            chapters[-1].text = f"{chapters[-1].text}\n\n{heading}\n\n{body}".strip()

    if chapters:
        return chapters
    return _fixed_sections(clean_body(text), fallback_section_words)


def _fixed_sections(text: str, words_per_section: int) -> list[Chapter]:
    paras = [p for p in text.split("\n\n") if p.strip()]
    chapters: list[Chapter] = []
    current: list[str] = []
    count = 0
    for para in paras:
        current.append(para)
        count += len(para.split())
        if count >= words_per_section:
            chapters.append(Chapter(len(chapters) + 1, f"Section {len(chapters) + 1}", "\n\n".join(current)))
            current, count = [], 0
    if current:
        chapters.append(Chapter(len(chapters) + 1, f"Section {len(chapters) + 1}", "\n\n".join(current)))
    return chapters
