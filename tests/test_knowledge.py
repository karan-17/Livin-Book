def test_alias_resolution(tiny_resources):
    k = tiny_resources.knowledge
    assert k.resolve_character("Lizzy") == "Elizabeth Bennet"
    assert k.resolve_character("mr darcy") == "Fitzwilliam Darcy"
    assert k.resolve_character("Darcy's") is not None
    assert k.resolve_character("Elizabth") == "Elizabeth Bennet"  # typo -> fuzzy match
    assert k.resolve_character("Sherlock") is None


def test_chapter_mentions_are_canonicalised(tiny_resources):
    k = tiny_resources.knowledge
    assert k.appearances["Elizabeth Bennet"] == [1, 2, 3, 4, 5]
    rel = k.relationship("Elizabeth", "Darcy")
    assert "Ch 1: mutual dislike" in rel and "Ch 5: engaged" in rel


def test_spoiler_guard_filters_every_lookup(tiny_resources):
    k = tiny_resources.knowledge
    rel = k.relationship("Elizabeth", "Darcy", max_chapter=3)
    assert "Ch 3" in rel and "Ch 4" not in rel and "engaged" not in rel

    profile = k.character_profile("Darcy", max_chapter=2)
    assert "Learns humility" not in profile and "Ch 3" not in profile

    overview = k.overview(max_chapter=2)
    assert "marries" not in overview and "The first proposal" not in overview

    assert "Spoiler guard" in k.chapter_summaries(4, 5, max_chapter=3)
    assert "Happy endings" in k.overview()
