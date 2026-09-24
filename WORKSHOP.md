# Facilitator guide — "Build a Novel Agent" (2 hours, hands-on)

**Audience:** developers who know Python and have used an LLM API; no prior LangGraph experience needed.
**Outcome:** each attendee leaves with a working agent over *Pride and Prejudice* that answers multi-turn
questions with chapter citations, remembers the reader, and is traced and evaluated in LangSmith.
**Format:** the complete code is provided; attendees run `notebooks/workshop.ipynb` cell by cell while you
explain each layer, then change things in the ✏️ exercises. The CLI and Streamlit app use the same package.

---

## Before the workshop (send to attendees 2-3 days ahead)

1. **Python 3.11+** and git. Clone the repo, then:
   ```bash
   python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt                      # ~3-5 min; chromadb and streamlit are the big ones
   ```
2. **Keys** in `.env` (copy `.env.example`):
   - `GOOGLE_API_KEY`: [aistudio.google.com/apikey](https://aistudio.google.com/apikey). **Everyone needs their own project**, because free quotas are per project.
   - `VOYAGE_API_KEY`: [dash.voyageai.com](https://dash.voyageai.com). **Add a payment method.** Without one the limit is 3 requests/min and 10K tokens/min, which turns a 1-minute embedding job into ~25 minutes and stalls live queries. With a card, the `voyage-4` / `rerank-3` free tokens (200M) still make the workshop cost $0.
   - `LANGSMITH_API_KEY`: [smith.langchain.com](https://smith.langchain.com) (free Developer plan).
3. Run `python -m novel_agent.doctor`. Everything must be `[ OK ]`.
4. **Strongly recommended:** pre-run `python -m novel_agent.ingest --book pride_and_prejudice` at home
   (2-5 minutes, about 9 Gemini calls plus about 25 Voyage calls). The notebook re-runs ingestion, but cached steps are skipped instantly.

**Facilitator prep**
- Run the whole notebook yourself the day before with the same models the attendees will use.
- Check the current free-tier limits for `gemini-3.5-flash-lite` in AI Studio (Google no longer publishes them). If they are very low, ask attendees to enable billing; a workshop costs cents.
- Build a fallback index: after ingesting, zip `data/index/pride_and_prejudice/` and have it ready on a USB stick or shared drive. Attendees whose ingestion fails can unzip it into `data/index/`. It only works if their `.env` uses the same embedding provider and model (`voyage` / `voyage-4`).
- Open LangSmith on the projector, in the `novel-agent-workshop` project.

---

## Timeline

| Time | Module | Notebook section | Key message |
|---|---|---|---|
| 0:00 | Intro and setup | 0 | The architecture on one slide: ingest (offline) vs. query (online) |
| 0:10 | 1. Ingestion | 1 | RAG quality is decided at ingestion: cleaning, structure and metadata |
| 0:30 | 2. Enrichment | 2 | Chunks answer detail questions; *summaries* answer big-picture ones |
| 0:45 | 3. Retrieval lab | 3 | Hybrid beats either method alone; filters let you control scope |
| 1:00 | 4. The agent | 4 | A graph is explicit control flow; the LLM chooses tools inside it |
| 1:25 | 5. Memory | 5 | Threads for short-term memory, the store for long-term, summaries for long threads |
| 1:40 | 6. Evaluate and UI | 6 | No evaluation, no progress: datasets, judges, experiments |
| 1:55 | Wrap-up | — | Extensions and Q&A |

### 0 · Intro and setup (10 min)
- Show the diagram in the README. Ingestion runs **once** per book; querying runs **every turn**.
- Everyone runs the setup cell and `doctor`. Pair anyone who has problems with a neighbour; don't hold up the room.
- **Checkpoint:** `doctor` is all `[ OK ]`.

### 1 · Ingestion (20 min)
Talking points:
- Show `raw.text[:1200]`: the preface and the list of illustrations are *noise that would be retrieved*.
- The chapter regex story: chapter 1's heading is `Chapter I.]`, the tail of an illustration block, and chapters 13-14 have no period. Real data is messy.
- Why chunk **within** chapters: every chunk carries `chapter` and `position` metadata, which gives citations, chronology and filters.
- The contextual header `[Pride and Prejudice · Chapter 34]` costs a few tokens and makes a dialogue-only chunk self-describing.
- Chunk size trade-off (the exercise): small chunks are precise but lack context; large chunks have context but are diluted, and it costs more tokens to stuff them into the prompt.
- **Checkpoint:** `ingest --skip-enrich` prints `Done`, with about 780 chunks.

### 2 · Enrichment (15 min)
Talking points:
- A bare vector index can't answer "what are the themes?", because no single chunk contains the answer. That is why we build a **hierarchical** index: chunks → chapter notes → book profile.
- `with_structured_output(Pydantic)` produces validated objects, not free text to parse. Gemini uses its native JSON-schema mode.
- Batching about 8 chapters per call keeps free-tier request quotas happy thanks to Gemini's 1M-token context. Missing chapters are retried individually.
- Alias resolution is deterministic Python on top of LLM output: *Lizzy*, *Eliza* and *Miss Elizabeth Bennet* all map to *Elizabeth Bennet*.
- Exercise answer: "Miss Bennet" is properly *Jane's* title, because she is the eldest daughter. If the LLM lists it under Jane, it resolves correctly. Aliases claimed by two characters are dropped as ambiguous. If there is no exact alias, the partial-name fallback picks the most prominent Bennet, which is Elizabeth. Use this to discuss where deterministic post-processing of LLM output needs domain knowledge.
- **Checkpoint:** `knowledge.relationship("Elizabeth", "Darcy")` prints a chapter-by-chapter timeline.

### 3 · Retrieval lab (15 min)
Talking points:
- Dense retrieval wins on paraphrase ("the moment she realises she has been blind"); BM25 wins on names ("Pemberley", "Rosings", "Hunsford").
- RRF: `1/(60+rank)` fuses rank positions, not raw scores, so there's no score calibration between very different systems.
- Reranking: a cross-encoder reads the query and passage *together*, which is more accurate but slower, so it's used only on the top 20.
- Metadata filters (`chapter_to=20`) are what the spoiler guard uses later.
- Exercise hints: BM25-friendly queries use rare names; dense-friendly ones describe emotions or events without names.

### 4 · The agent (25 min)
Talking points (show the Mermaid diagram):
- **Why a graph and not just a prompt:** explicit, testable control flow; state you can inspect; persistence and streaming for free.
- `AgentState = MessagesState + summary + analysis`; nodes return *partial updates*; `add_messages` appends.
- The **analyze** node is multi-step reasoning made visible: it rewrites the follow-up as a standalone question, classifies it, and plans the research steps. The plan is injected into the agent's system prompt.
- The **agent ⇄ tools loop**: tools are ordinary Python functions whose docstrings are the LLM's "API docs". Read `search_book`'s docstring aloud.
- `RemainingSteps` and `recursion_limit` bound the loop. Near the limit, the agent is switched to `tool_choice="none"` and must answer.
- Gemini-specific details: `AIMessage.content` is a list of blocks, so use `.text`; thought signatures ride on messages, so never rebuild AIMessages.
- **Live in LangSmith:** open the trace for "How does Elizabeth's opinion of Darcy change?" and walk through analyze → several tool calls (usually `get_relationship` and `search_book` with chapter ranges) → the answer with `[Ch. N]` citations. Point out the `hybrid_search` retriever spans with their documents.
- **Checkpoint:** the agent answers a theme question with citations.

### 5 · Memory (15 min)
Talking points:
- The **checkpointer** saves state after every node, per `thread_id`. The follow-up "What made her reconsider?" works in the same thread and fails in a new one. Show `get_state_history`, where every step is a checkpoint (time travel).
- The **store** is cross-thread, per-user memory. Telling the agent "I've only read up to chapter 20" calls `set_reading_progress`; a *new* thread for the same user still refuses spoilers.
- The spoiler guard is enforced **in the tools** (code), not just requested in the prompt. Prompts are suggestions; code is a guarantee.
- **Summarization** keeps long threads cheap. It removes whole turns only, because a tool result without its tool call is an invalid conversation.
- CLI and Streamlit use SQLite (`data/memory/`), so memory survives restarts. Demo `python -m novel_agent.chat --thread <id>` to resume a thread.

### 6 · Evaluate and UI (15 min)
Talking points:
- Reference answers with chapter citations cover every question type, including multi-turn examples.
- An LLM-as-judge for correctness, plus cheap deterministic checks (cites a chapter? used a tool?).
- Experiments are comparable: change one variable (reranker off, model, prompt), re-run, and compare side by side in LangSmith.
- `streamlit run app.py`: show the live plan and tool steps, the sources expander, the spoiler slider, and 👍/👎 feedback arriving on the LangSmith trace.
- The eval costs about 5 Gemini calls per question. `--limit 4` keeps it under free-tier quotas during the session.

### Wrap-up (5 min)
Extensions (also listed at the end of the notebook): a self-check/citation-verification node, GraphRAG over the relationship
notes, `langgraph dev` and Studio, human-in-the-loop with `interrupt()`, other books with `--file`.

---

## Troubleshooting during the session

| Symptom | Fix |
|---|---|
| `429 RESOURCE_EXHAUSTED` from Gemini | Free quota hit: wait a minute, lower `GEMINI_RPM`, or switch that attendee to a billing-enabled key |
| Ingestion stuck at "embedded batch 1/…" | Voyage free trial (no card) at 3 RPM: add a card, or hand out the pre-built index zip |
| First Voyage call hangs | It downloads a Hugging Face tokenizer; corporate proxies may block `huggingface.co` |
| `was embedded with … but .env selects …` | Embedding provider or model changed since ingestion; revert it or re-run `ingest` |
| Answer has no citations | Model variance; show it in the eval (`cites_chapters`), then tighten `AGENT_SYSTEM` as an exercise |
| `No books ingested yet` in Streamlit | Run `streamlit` from the repo root after ingesting |
| Notebook can't import `novel_agent` | The first setup cell `chdir`s to the repo root; make sure the notebook sits in `notebooks/` |

## Cost and quota budget per attendee (Pride and Prejudice, defaults)

| Step | Gemini requests | Voyage requests | Notes |
|---|---|---|---|
| Ingestion (embed) | 0 | ~25 embed | ~190K tokens, within the free Voyage tokens |
| Enrichment | ~9 | ~1 embed | ~200K input tokens to Flash-Lite |
| Each chat question | 3-6 | 1-3 embed + 1-3 rerank | analyze + agent loop |
| Eval (`--limit 4`) | ~25 | ~10 | + 4 judge calls |
| **Whole workshop** | **~90-120** | **~80** | Fits Flash-Lite's reported free-tier daily quota |
