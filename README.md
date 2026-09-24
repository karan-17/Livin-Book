# 📚 Novel Agent — agentic RAG over a book with LangChain, LangGraph & LangSmith

An AI agent that ingests a novel and holds a multi-turn conversation about its **characters, events,
chapters, relationships, themes and ending**, using:

- **Retrieval**: chapter-aware chunking, Voyage embeddings in Chroma, BM25, Reciprocal Rank Fusion and Voyage reranking
- **Hierarchical knowledge**: a one-off Gemini map-reduce that writes chapter notes, character profiles (with aliases), relationship timelines, themes, a synopsis and the ending
- **Multi-step reasoning**: a LangGraph agent that plans (structured "analyze" step) and then loops over 8 tools until it has enough evidence, citing `[Ch. N]` for its claims
- **Memory**: per-thread conversation state (a checkpointer), rolling summaries of long threads, and a long-term reader profile (a store) that holds the **spoiler guard** and reader preferences
- **Observability and evaluation**: LangSmith tracing of every node, tool and retriever call, plus an LLM-judged evaluation dataset

It is built for a **2-hour hands-on workshop**; see [WORKSHOP.md](WORKSHOP.md) for the facilitator guide and
[notebooks/workshop.ipynb](notebooks/workshop.ipynb) for the step-by-step walkthrough.

```
INGEST  python -m novel_agent.ingest                         QUERY  (LangGraph)
  Gutenberg / .txt / .pdf / .epub                               START
    → clean (boilerplate, illustrations, CRLF)                    → manage_memory   rolling summary of old turns
    → chapters (regex + TOC guard)                                → analyze         standalone question, type, research steps
    → chunks (1200/200, contextual header)                        → agent ⇄ tools   Gemini + 8 tools (multi-step loop)
    → Voyage embeddings → Chroma     + BM25 corpus                → END
    → Gemini map-reduce → knowledge.json + summary index        memory: SqliteSaver (threads) + SqliteStore (reader profile)
                                                                retrieval: dense + BM25 → RRF → rerank (chapter filters)
```

## Quickstart (Windows / macOS / Linux, Python 3.11+)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env   — then add your keys
python -m novel_agent.doctor      # every line should be [ OK ]

python -m novel_agent.ingest --book pride_and_prejudice --dry-run   # free: inspect chapters/chunks
python -m novel_agent.ingest --book pride_and_prejudice             # embed + enrich (~2-5 min)

python -m novel_agent.chat --book pride_and_prejudice               # terminal chat
streamlit run app.py                                                 # web UI
python eval/run_eval.py --limit 4                                    # LangSmith evaluation
```

**Keys you need:** `GOOGLE_API_KEY` ([AI Studio](https://aistudio.google.com/apikey)), `VOYAGE_API_KEY`
([Voyage](https://dash.voyageai.com); **add a payment method**, because without one Voyage allows only 3 requests/min, and
usage stays within the free tokens), and `LANGSMITH_API_KEY` ([LangSmith](https://smith.langchain.com)).

### Things to ask
- *How does Elizabeth's opinion of Darcy change over the novel?* → then *What made her reconsider?*
- *What happens in chapter 34?* · *Who is Charlotte Lucas and why does she marry Mr. Collins?*
- *What are the main themes, and where are they strongest?* · *How does the novel end?*
- *I've only read up to chapter 20. What do you make of Wickham?* (turns on the spoiler guard for all your threads)
- *Keep answers under 100 words from now on.* (saved as a long-term preference)

## Project layout

| Path | What it does |
|---|---|
| `novel_agent/config.py` | `.env` settings; Gemini chat models (rate limiter, `thinking_level`), embeddings with retry/throttle, reranker |
| `novel_agent/loaders.py`, `chapters.py`, `chunking.py` | Loading a book (Gutenberg / URL / txt / pdf / epub), cleaning it, splitting it into chapters and chunks |
| `novel_agent/indexing.py` | Chroma collections (resumable, batched by token budget), `chunks.jsonl`, `manifest.json` |
| `novel_agent/enrich.py` | Map-reduce structured extraction → `knowledge.json`; alias resolution |
| `novel_agent/retrieval.py` | BM25, RRF, `HybridRetriever` (dense + sparse + rerank, chapter filters, LangSmith retriever traces) |
| `novel_agent/knowledge.py` | Character, relationship, chapter and overview lookups, with the spoiler guard applied |
| `novel_agent/tools.py` | The agent's 8 tools (the spoiler guard is enforced in code) |
| `novel_agent/graph.py` | LangGraph `StateGraph`: manage_memory → analyze → agent ⇄ tools |
| `novel_agent/memory.py` | Checkpointer and store factories (SQLite / in-memory), reader profile helpers |
| `novel_agent/runner.py` | Streams a turn as UI-neutral events (plan, tool calls, tokens, final answer with sources) |
| `novel_agent/ingest.py`, `chat.py`, `doctor.py` | Command-line tools |
| `app.py` | Streamlit UI (live reasoning steps, sources, spoiler slider, feedback sent to LangSmith) |
| `eval/` | Reference Q&A (`questions.jsonl`) and the LangSmith evaluation runner |
| `notebooks/workshop.ipynb` | The workshop walkthrough, with exercises |
| `tests/` | Offline tests (no keys needed): `python -m pytest` |

Data is written to `data/` (`raw/` downloads, `index/<book>/` vectors and knowledge, `memory/` SQLite threads and profiles).

## Other books

```bash
python -m novel_agent.ingest --book frankenstein
python -m novel_agent.ingest --gutenberg 11 --book-id alice --title "Alice's Adventures in Wonderland" --author "Lewis Carroll"
python -m novel_agent.ingest --file path/to/novel.epub --book-id my_novel --chapter-regex "^Chapter \d+$"
```
If your book's headings aren't found, the text is cut into ~3,000-word sections instead. Use `--dry-run` to check the chapter split before you pay for any API calls.

## Configuration highlights (`.env`)

| Setting | Default | Notes |
|---|---|---|
| `AGENT_MODEL` / `ENRICH_MODEL` / `JUDGE_MODEL` | `gemini-3.5-flash-lite` | Best daily quota on the free tier. With billing, `gemini-3.8-flash` reasons better |
| `AGENT_THINKING_LEVEL` | `low` | `minimal`/`low`/`medium`/`high` (`minimal` is not allowed on 3.8 Flash) |
| `GEMINI_RPM` | `10` | client-side throttle for each model |
| `EMBEDDINGS_PROVIDER` | `voyage` | `google` = `gemini-embedding-001` (re-ingest after switching) |
| `RERANK_ENABLED` | `true` | Voyage `rerank-3-lite`; if it fails, the fused (RRF) order is used |
| `CHAPTERS_PER_CALL` | `8` | enrichment batching (about 8 calls for 61 chapters) |

## Troubleshooting
- **`429` / `RESOURCE_EXHAUSTED` from Gemini**: you've hit the free-tier quota. Lower `GEMINI_RPM`, use Flash-Lite, or enable billing. Quotas are per project, so each attendee needs their own key.
- **Voyage is very slow or returns 429**: add a payment method, or set `EMBED_RPM=1` and `EMBED_BATCH_TOKENS=8000`. Re-running `ingest` resumes where it stopped.
- **The first Voyage call hangs**: it downloads a tokenizer from Hugging Face; allow `huggingface.co` through your proxy or firewall.
- **Garbled quotes in the Windows terminal**: the CLIs switch stdout to UTF-8. Use Windows Terminal rather than the legacy console.
- **"was embedded with … but .env selects …"**: you switched embedding provider or model; re-run `ingest`.
