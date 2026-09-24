# 📖🤖 Talk to a Book (student edition)

Build an AI agent that reads a novel and chats with you about its characters, events, chapters, relationships,
themes and ending, in one Jupyter notebook (about 200 lines of code) plus a small provided helper, `voyage_free_tier.py`.
It is designed to be built from scratch by an undergraduate in **1.5–2 hours**.

| Library | Used for |
|---|---|
| **LangChain** | `Document`s, text splitter, Voyage embeddings, in-memory vector store, `init_chat_model` (Gemini), `@tool` |
| **LangGraph** | The agent graph (`agent` ⇄ `tools` loop), `ToolNode`, conditional edges, `InMemorySaver` memory with `thread_id` |
| **LangSmith** | Automatic tracing of every step, `@traceable`, a small dataset and an LLM-as-judge evaluation |

This is the simplified sibling of the full project in the parent folder (`../novel_agent`). The full project adds
hybrid BM25 search, reranking, LLM enrichment, a spoiler guard, long-term memory, a CLI and Streamlit. This version
keeps only the essentials.

## Setup (≈10 min)

```bash
cd talk_to_a_book
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # macOS/Linux: cp .env.example .env   — then paste your keys
jupyter notebook talk_to_a_book.ipynb
```

Keys (all have free tiers):
- `GOOGLE_API_KEY`: Gemini, from https://aistudio.google.com/apikey
- `VOYAGE_API_KEY`: embeddings, from https://dash.voyageai.com. The free tier is enough: `voyage_free_tier.py` keeps every
  request inside its limits of 3 requests and 10K tokens per minute. Embedding a novel then takes about 20 minutes, once.
  With a payment method it takes seconds (pass `free_tier=False`).
- `LANGSMITH_API_KEY`: tracing, from https://smith.langchain.com

You can skip the `.env` file: the notebook asks for any missing key when it starts.

## Using your own book

Change two lines in the first code cell:

```python
BOOK_PATH = "books/my_novel.txt"    # any .txt or .pdf
BOOK_TITLE = "My Novel"
```

Then check that the chapter count printed in Step 2 matches the book. If it doesn't, adapt `CHAPTER_PATTERN` to
the way your book writes its headings. The embedded index is saved to `index/<book file name>.json`, so re-runs
start instantly. Delete that file if you change the chunking or the embedding model.

Also change the book-specific example questions in Steps 3, 4 and 7, and the reference answers in Step 8.

### Several books

Every book gets its own index file, so any number of books can live side by side in `index/`. Tested with
*Pride and Prejudice* and *The Hound of the Baskervilles*:
- **Switching books** means changing `BOOK_PATH` / `BOOK_TITLE` and re-running. An already-embedded book reloads in under a second.
- **The books stay separate.** Asked about Mr. Darcy, the Hound agent answers that he isn't in that book.
- **One book per notebook session.** The tools search the current book's `vector_store` and `chapters`. To chat with two books at the same time, open two copies of the notebook (embed them one after the other, since they share the Voyage free-tier budget), or extend the tools with a `book` argument.
- `books/` already contains *The Hound of the Baskervilles* to try.

## Measured performance

These are live runs with Gemini `gemini-3.5-flash-lite` and Voyage `voyage-4` on the **free tier** (no payment method).
The benchmarks are reference Q&A sets graded by the notebook's LLM judge. They cover characters, events, chapters,
relationships, themes, the ending, multi-hop questions and multi-turn follow-ups.

| | *Pride and Prejudice* | *The Hound of the Baskervilles* |
|---|---|---|
| Chapters / chunks | 61 / 786 | 15 / 396 |
| One-time embedding (free tier) | 20.3 min, 60 requests, 0 rate-limit errors | 10.3 min, 31 requests, 0 rate-limit errors |
| Saved index / reload time | 24 MB / 0.5 s | 12 MB / 0.3 s |
| Benchmark accuracy | **14/14** (two separate runs) | **8/8** |
| Answers citing a chapter | 14/14 | 8/8 |
| Time per question (median) | 48 s, of which ~42 s is Voyage free-tier pacing (**~6 s without it**) | 29 s (**~5 s** without pacing) |
| LLM calls / tool calls per question | 3.1 / 2.1 | 3.2 / 2.1 |
| Tokens per question (in / out) | ~8K / ~600 | ~8K / ~400 |

- **Search accuracy on its own** (*Pride and Prejudice*, 12 well-known events, plain vector search): the right chapter comes first 42% of the time and is in the top 6 results 75% of the time. Every miss was the neighbouring chapter describing the same scene. The agent makes up for this by searching several times and reading whole chapters.
- **Cost:** $0 on the free tiers. With paid Gemini Flash-Lite pricing, a question costs about $0.004.
- **With a Voyage payment method** (`free_tier=False`), the pacing disappears: embedding takes seconds and answers take about 5–6 s.

## The build, step by step

| Step | Time | You build | Concepts |
|---|---|---|---|
| 0 | 10 min | Config and keys | environment variables, LangSmith tracing on |
| 1–2 | 20 min | Load the book, split it into chapters and chunks | cleaning data, regex, `Document` + metadata, chunk size and overlap |
| 3 | 15 min (+~20 min of embedding on Voyage's free tier) | Embeddings and a vector store (saved to disk, resumable) | embeddings, similarity search, the "R" in RAG, respecting API rate limits |
| 4–5 | 15 min | `search_book` and `read_chapter` tools; the Gemini chat model | tools and docstrings, `bind_tools`, `init_chat_model` |
| 6 | 20 min | The LangGraph agent | state, nodes, edges, conditional edges, the ReAct loop |
| 7 | 15 min | Multi-turn chat | checkpointer memory, `thread_id`, streaming the steps, an interactive chat loop |
| 8 | 10 min | LangSmith | reading traces, datasets, LLM-as-judge evaluation |

**Checkpoints for instructors:**
- Step 2 prints **61 chapters** for *Pride and Prejudice*.
- Step 3 ends with "Index ready" (about 60 small requests, ~20 min on the free tier; re-running resumes).
- Step 7 shows 🔧 tool calls followed by an answer with `[Ch. N]` citations.
- The follow-up ("the turning point for her") is answered in context, and "Remind me: what have we talked about so far?"
  lists the earlier questions in `reader-1` but finds nothing in the new `someone-else` thread.

## Troubleshooting

| Problem | Fix |
|---|---|
| `RateLimitError: … 3 RPM and 10K TPM` (Voyage) | Make sure Step 3 uses `FreeTierVoyageEmbeddings` (it paces and retries by itself). If you run two notebooks with the same key they share the limit, so run one at a time |
| Embedding is slow (~20 min) | Expected on Voyage's free tier. Progress is saved after every request, so re-running the cell resumes. A payment method plus `free_tier=False` makes it take seconds |
| `429 RESOURCE_EXHAUSTED` from Gemini | Free-tier quota reached: wait a minute. Each question uses 2–4 model calls |
| `400 API_KEY_INVALID` from Gemini | The key was mistyped or deleted. Create a new one at https://aistudio.google.com/apikey |
| LangSmith `403 Forbidden` | Your LangSmith account is in the EU: add `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` to `.env` |
| Answer stops mid-sentence with "⚠️ Gemini stopped early (PROHIBITED_CONTENT)" | Gemini's safety filter sometimes misfires on sensitive plot points of classic novels (seen once with Wickham and 15-year-old Georgiana). Rephrase the question, or ask it in a new thread |
| Wrong number of chapters | Adjust `CHAPTER_PATTERN` (Step 2) to match the headings in your file |
| `GraphRecursionError` | The agent looped more than 20 steps; rephrase the question or raise `recursion_limit` in `ask()` |
| The graph picture doesn't show | It's rendered online by mermaid.ink; paste the printed text into https://mermaid.live instead |
| First Voyage call is slow | It downloads a small tokenizer from Hugging Face once |
| Want to use Gemini embeddings instead of Voyage | In Step 3, replace the embeddings line with `from langchain_google_genai import GoogleGenerativeAIEmbeddings; embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`, remove the `batches` loop (use `vector_store.add_documents(docs)`), and delete `index/` |
