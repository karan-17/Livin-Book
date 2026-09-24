import pytest

from novel_agent.books import GENERIC_CHAPTER_PATTERN, PRESETS
from novel_agent.chapters import clean_body, remove_bracketed_blocks, split_chapters
from novel_agent.loaders import normalize_text, strip_gutenberg_boilerplate

PP_PATTERN = PRESETS["pride_and_prejudice"].chapter_pattern
FILLER = " ".join(["word"] * 150)

# Reproduces the quirks of Gutenberg #1342: CRLF, front matter with a list of illustrations,
# chapter 1's heading inside an illustration block, nested brackets, a heading without a period,
# "Chapter XLVI."-style headings, _italics_, M^{r.} and a printer's colophon at the very end.
GUTENBERG_LIKE = (
    "Title: Test Novel\r\nAuthor: A. Writer\r\n\r\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK TEST NOVEL ***\r\n"
    "PREFACE\r\n\r\nSome critic writes about the novel.\r\n\r\n"
    "Heading to Chapter IV.                     18\r\n\r\n"
    "[Illustration: ·TEST NOVEL·\r\n\r\n\r\nChapter I.]\r\n\r\n"
    f"It is a truth universally acknowledged. {FILLER}\r\n\r\n"
    "[Illustration:\r\n\r\n“He came down”\r\n\r\n[_Copyright 1894 by George Allen._]]\r\n\r\n"
    f"More of chapter one with _emphasis_ and M^{{r.}} Bennet. {FILLER}\r\n\r\n"
    f"CHAPTER II\r\n\r\nSecond chapter text. {FILLER}\r\n\r\n"
    f"Chapter III.\r\n\r\nThird chapter text. {FILLER}\r\n\r\n"
    "[Illustration:\r\n\r\n THE\r\n END\r\n ]\r\n\r\n"
    "CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO.\r\nTOOKS COURT, CHANCERY LANE, LONDON.\r\n\r\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK TEST NOVEL ***\r\nLicense text...\r\n"
)


def test_gutenberg_quirks_are_handled():
    body, meta = strip_gutenberg_boilerplate(normalize_text(GUTENBERG_LIKE))
    assert meta == {"title": "Test Novel", "author": "A. Writer"}
    assert "License" not in body

    chapters = split_chapters(body, PP_PATTERN)
    assert [c.number for c in chapters] == [1, 2, 3]
    assert [c.heading for c in chapters] == ["Chapter I.]", "CHAPTER II", "Chapter III."]
    assert chapters[0].text.startswith("It is a truth universally acknowledged")
    full = "\n".join(c.text for c in chapters)
    for artifact in ("[", "]", "Illustration", "_", "^{", "PREFACE", "CHISWICK", "TOOKS"):
        assert artifact not in full, artifact
    assert "Mr. Bennet" in chapters[0].text and "emphasis" in chapters[0].text


def test_nested_and_unclosed_brackets():
    assert remove_bracketed_blocks("a [Illustration: x [y] z] b") == "a  b"
    # An unclosed block only swallows its own paragraph, never the rest of the text.
    assert remove_bracketed_blocks("a [Illustration: oops\n\nnext paragraph") == "a \n\nnext paragraph"


def test_table_of_contents_is_skipped_and_short_sections_merged():
    toc = "Contents\n\nChapter 1\nChapter 2\nChapter 3\n\n"
    text = toc + f"Chapter 1\n\nOne. {FILLER}\n\nChapter 2\n\nTiny.\n\nChapter 3\n\nThree. {FILLER}"
    chapters = split_chapters(text, GENERIC_CHAPTER_PATTERN)
    assert len(chapters) == 2  # TOC dropped, the tiny "Chapter 2" merged into chapter 1
    assert chapters[0].text.startswith("One.") and "Tiny." in chapters[0].text
    assert chapters[1].text.startswith("Three.")


def test_fallback_to_fixed_sections_without_headings():
    text = "\n\n".join(f"Paragraph {i}. " + " ".join(["w"] * 100) for i in range(10))
    chapters = split_chapters(text, GENERIC_CHAPTER_PATTERN, fallback_section_words=300)
    assert len(chapters) >= 3 and chapters[0].heading == "Section 1"


def test_clean_body_trims_all_caps_colophon_only_at_end():
    text = "Story text here.\n\nMORE STORY IN CAPS but lowercase too.\n\nTHE END"
    assert clean_body(text).endswith("lowercase too.")


@pytest.mark.network
def test_real_pride_and_prejudice_has_61_clean_chapters(data_dir):
    from novel_agent.loaders import load_book

    try:
        raw = load_book(gutenberg_id=1342)
    except Exception as exc:  # offline
        pytest.skip(f"Gutenberg not reachable: {exc}")
    chapters = split_chapters(raw.text, PP_PATTERN)
    assert len(chapters) == 61
    assert chapters[0].text.startswith("It is a truth universally acknowledged")
    assert all("[Illustration" not in c.text and "CHISWICK" not in c.text for c in chapters)
