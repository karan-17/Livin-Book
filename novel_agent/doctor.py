"""Pre-flight check for the workshop: keys, model access, embeddings, reranker, LangSmith.

    python -m novel_agent.doctor
"""

from __future__ import annotations

import os
import sys
import time

from .config import PROJECT_ROOT, get_embeddings, get_llm, get_reranker, get_settings

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def _check(label: str, fn) -> bool:
    t0 = time.time()
    try:
        detail = fn()
        print(f"{OK} {label}: {detail} ({time.time() - t0:.1f}s)")
        return True
    except Exception as exc:
        print(f"{FAIL} {label}: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    results = []
    print(f"Python {sys.version.split()[0]}  ·  project {PROJECT_ROOT}\n")

    if not (PROJECT_ROOT / ".env").exists():
        print(f"{WARN} no .env file — copy .env.example to .env and fill in your keys")

    has_google = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    print(f"{OK if has_google else FAIL} GOOGLE_API_KEY {'set' if has_google else 'missing'}")
    results.append(has_google)
    if s.embeddings_provider == "voyage":
        has_voyage = bool(os.getenv("VOYAGE_API_KEY"))
        print(f"{OK if has_voyage else FAIL} VOYAGE_API_KEY {'set' if has_voyage else 'missing'}")
        results.append(has_voyage)

    results.append(_check(f"Gemini chat ({s.agent_model})",
                          lambda: get_llm("agent").invoke("Reply with exactly: OK").text.strip()[:40]))
    if s.enrich_model != s.agent_model:
        results.append(_check(f"Gemini chat ({s.enrich_model})",
                              lambda: get_llm("enrich").invoke("Reply with exactly: OK").text.strip()[:40]))

    results.append(_check(f"Embeddings ({s.embeddings_provider}:{s.embed_model})",
                          lambda: f"{len(get_embeddings().embed_query('Mr. Darcy'))} dimensions"))

    reranker = get_reranker()
    if reranker is not None:
        from langchain_core.documents import Document

        def rerank():
            docs = [Document("Darcy proposes at Hunsford."), Document("Jane catches a cold.")]
            best = reranker.compress_documents(docs, "Where does Darcy propose?")[0]
            return f"top result: {best.page_content!r}"

        results.append(_check(f"Reranker ({s.voyage_rerank_model})", rerank))
    else:
        print(f"{WARN} reranker disabled (RERANK_ENABLED=false or not using Voyage)")

    tracing = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() == "true"
    if tracing:
        def langsmith():
            from langsmith import Client

            next(iter(Client().list_projects(limit=1)), None)
            return f"authenticated · project={os.getenv('LANGSMITH_PROJECT', 'default')}"

        results.append(_check("LangSmith", langsmith))
    else:
        print(f"{WARN} LangSmith tracing is off (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")

    from .indexing import list_ingested_books

    books = list_ingested_books()
    print(f"\nIngested books: {', '.join(b['book_id'] for b in books) or 'none yet'}")
    if s.embeddings_provider == "voyage":
        print("Tip: Voyage accounts without a payment method are limited to 3 requests/min — add a card "
              "(usage stays within the free token allowance) before the workshop.")
    print("\nAll checks passed." if all(results) else "\nSome checks failed — see above.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
