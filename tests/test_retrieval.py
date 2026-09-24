from langchain_core.documents import Document

from novel_agent.retrieval import BM25Index, chroma_chapter_filter, reciprocal_rank_fusion, tokenize


def _doc(doc_id: str, chapter: int, text: str) -> Document:
    return Document(id=doc_id, page_content=text, metadata={"doc_id": doc_id, "chapter": chapter})


def test_tokenize_drops_stopwords_and_possessives():
    assert tokenize("What did Darcy's letter say to Elizabeth?") == ["darcy", "letter", "say", "elizabeth"]


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    a, b, c = _doc("a", 1, "a"), _doc("b", 1, "b"), _doc("c", 1, "c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [d.metadata["doc_id"] for d in fused] == ["b", "a", "c"]
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_bm25_finds_names_and_respects_chapter_range():
    docs = [
        _doc("1", 1, "Wickham charms Elizabeth at Meryton"),
        _doc("2", 2, "Mr. Collins proposes to Elizabeth"),
        _doc("3", 5, "Lydia elopes with Wickham"),
    ]
    index = BM25Index(docs)
    assert [d.metadata["doc_id"] for d in index.search("Wickham")] == ["1", "3"] or \
           [d.metadata["doc_id"] for d in index.search("Wickham")] == ["3", "1"]
    assert [d.metadata["doc_id"] for d in index.search("Wickham", chapter_to=3)] == ["1"]
    assert [d.metadata["doc_id"] for d in index.search("Wickham", chapter_from=4)] == ["3"]
    assert index.search("Pemberley") == []


def test_chroma_filter_shapes():
    assert chroma_chapter_filter(None, None) is None
    assert chroma_chapter_filter(3, None) == {"chapter": {"$gte": 3}}
    assert chroma_chapter_filter(3, 9) == {"$and": [{"chapter": {"$gte": 3}}, {"chapter": {"$lte": 9}}]}


def test_hybrid_retriever_filters_dense_and_sparse(tiny_resources):
    docs = tiny_resources.chunks.search("Wickham letter", k=4, chapter_to=3)
    assert docs and all(d.metadata["chapter"] <= 3 for d in docs)
    top = tiny_resources.chunks.search("Lydia elopes Wickham", k=1)
    assert top[0].metadata["chapter"] == 5  # BM25 pulls the exact scene even with random dense vectors
