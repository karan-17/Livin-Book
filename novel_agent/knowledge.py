"""Read-side of the enrichment output: fast, structured lookups for the agent's tools.

Every lookup takes `max_chapter` — the spoiler guard. When set, anything after that chapter is
filtered out, and book-level fields that summarise the whole story (synopsis, ending, character
arcs) are withheld.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from .enrich import _norm

_TITLES = {"mr", "mrs", "miss", "lady", "sir", "colonel", "dr", "the", "captain", "lord"}
_IMPORTANCE_RANK = {"main": 0, "secondary": 1, "minor": 2}


def _visible(chapter: int, max_chapter: int | None) -> bool:
    return max_chapter is None or chapter <= max_chapter


def _spread(items: list, n: int) -> list:
    """Pick up to n items spread evenly across the list (keeps first and last)."""
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


class BookKnowledge:
    def __init__(self, data: dict):
        self.data = data
        self.title: str = data["title"]
        self.author: str = data["author"]
        self.chapter_count: int = data["chapter_count"]
        self.chapters: dict[int, dict] = {c["chapter"]: c for c in data["chapters"]}
        self.profile: dict = data["profile"]
        self.alias_map: dict[str, str] = data["alias_map"]
        self.appearances: dict[str, list[int]] = data["appearances"]
        self.characters: dict[str, dict] = {c["name"]: c for c in self.profile["characters"]}

    @classmethod
    def load(cls, path: Path) -> "BookKnowledge | None":
        return cls(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None

    # ------------------------------------------------------------------ name resolution

    def _rank(self, name: str) -> tuple:
        importance = self.characters.get(name, {}).get("importance", "minor")
        return (_IMPORTANCE_RANK.get(importance, 3), -len(self.appearances.get(name, [])))

    def resolve_character(self, name: str) -> str | None:
        """Map any reference ("Lizzy", "Mr. Darcy", "darcy") to a canonical character name."""
        key = _norm(name)
        if not key:
            return None
        if key in self.alias_map:
            return self.alias_map[key]
        known = {**{_norm(n): n for n in self.appearances}, **self.alias_map}
        tokens = set(key.split()) - _TITLES or set(key.split())
        candidates = {canon for alias, canon in known.items() if tokens <= set(alias.split())}
        if candidates:
            return min(candidates, key=self._rank)
        close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.75)
        return known[close[0]] if close else None

    # ------------------------------------------------------------------ lookups

    def main_characters(self) -> list[str]:
        return sorted(self.characters, key=self._rank)

    def chapter_summaries(self, chapter_from: int, chapter_to: int, max_chapter: int | None = None) -> str:
        if max_chapter is not None and chapter_from > max_chapter:
            return f"Spoiler guard: the reader has only read up to chapter {max_chapter}."
        end = min(chapter_to, max_chapter) if max_chapter is not None else chapter_to
        blocks = []
        for n in range(chapter_from, end + 1):
            c = self.chapters.get(n)
            if not c:
                continue
            events = "\n".join(f"  - {e}" for e in c["key_events"])
            chars = ", ".join(x["name"] for x in c["characters"])
            blocks.append(
                f"Chapter {n}: {c['title']}\n{c['summary']}\nKey events:\n{events}\n"
                f"Characters: {chars}\nThemes: {', '.join(c['themes'])}"
            )
        note = f"\n\n(Spoiler guard: stopped at chapter {max_chapter}.)" if end < chapter_to else ""
        return ("\n\n".join(blocks) or "No chapter notes for that range.") + note

    def character_profile(self, name: str, max_chapter: int | None = None) -> str:
        canon = self.resolve_character(name)
        if not canon:
            return f"No character matching {name!r}. Main characters: {', '.join(self.main_characters()[:15])}."
        profile = self.characters.get(canon, {})
        chapters = [c for c in self.appearances.get(canon, []) if _visible(c, max_chapter)]
        lines = [f"Character: {canon}"]
        if profile.get("aliases"):
            lines.append(f"Also called: {', '.join(profile['aliases'])}")
        if profile.get("importance"):
            lines.append(f"Importance: {profile['importance']}")
        if max_chapter is None:
            if profile.get("description"):
                lines.append(f"Description: {profile['description']}")
            if profile.get("arc"):
                lines.append(f"Arc over the novel: {profile['arc']}")
        else:
            lines.append(f"(Spoiler guard: description and arc hidden; showing chapters 1-{max_chapter} only.)")
        lines.append(f"Appears in chapters: {', '.join(map(str, chapters)) or 'none yet'}")
        roles = []
        for n in chapters:
            for c in self.chapters.get(n, {}).get("characters", []):
                if c["name"] == canon:
                    roles.append(f"  Ch {n}: {c['role']}")
                    break
        if roles:
            lines.append("Role in selected chapters:")
            lines.extend(_spread(roles, 12))
        return "\n".join(lines)

    def relationship(self, character_a: str, character_b: str = "", max_chapter: int | None = None) -> str:
        a = self.resolve_character(character_a)
        if not a:
            return f"No character matching {character_a!r}."
        b = self.resolve_character(character_b) if character_b else None
        if character_b and not b:
            return f"No character matching {character_b!r}."
        records = []
        for n in sorted(self.chapters):
            if not _visible(n, max_chapter):
                continue
            for r in self.chapters[n]["relationships"]:
                pair = {r["source"], r["target"]}
                if a in pair and (b is None or b in pair) and len(pair) == 2:
                    other = (pair - {a}).pop()
                    records.append((n, other, r["relation"], r["description"]))
        if not records:
            who = f"{a} and {b}" if b else a
            return f"No recorded relationship notes for {who}. Try search_book instead."
        guard = f" (chapters 1-{max_chapter} only, spoiler guard)" if max_chapter is not None else ""
        if b:
            lines = [f"Relationship timeline: {a} & {b}{guard}"]
            lines += [f"  Ch {n}: {rel} — {desc}" for n, _, rel, desc in _spread(records, 25)]
            return "\n".join(lines)
        by_other: dict[str, list] = {}
        for rec in records:
            by_other.setdefault(rec[1], []).append(rec)
        lines = [f"Relationships of {a}{guard}:"]
        for other, recs in sorted(by_other.items(), key=lambda kv: -len(kv[1]))[:12]:
            first, last = recs[0], recs[-1]
            lines.append(f"- {other}: {len(recs)} chapter(s); first Ch {first[0]} ({first[2]}), "
                         f"latest Ch {last[0]} ({last[2]}: {last[3]})")
        return "\n".join(lines)

    def overview(self, max_chapter: int | None = None) -> str:
        lines = [f'"{self.title}" by {self.author} — {self.chapter_count} chapters.']
        if max_chapter is None:
            lines += ["", f"Synopsis: {self.profile['synopsis']}", "", f"Ending: {self.profile['ending']}"]
        else:
            lines += ["", f"(Spoiler guard: the reader has read up to chapter {max_chapter}; "
                          "synopsis and ending hidden.)"]
        lines += ["", "Major themes:"]
        for t in self.profile["themes"]:
            chapters = [c for c in t["key_chapters"] if _visible(c, max_chapter)]
            lines.append(f"- {t['name']}: {t['description']} (key chapters: {', '.join(map(str, chapters)) or '-'})")
        lines += ["", "Main characters:"]
        for name in self.main_characters()[:12]:
            ch = self.characters[name]
            desc = f" — {ch['description']}" if max_chapter is None else ""
            lines.append(f"- {name} ({ch['importance']}){desc}")
        lines += ["", "Chapter guide:"]
        lines += [f"  {n}. {self.chapters[n]['title']}" for n in sorted(self.chapters) if _visible(n, max_chapter)]
        return "\n".join(lines)
