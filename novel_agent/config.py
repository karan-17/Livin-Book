"""Central configuration: reads `.env` and builds the chat models, embeddings and reranker.

Everything that talks to an external API is created here, so switching models or providers
is a one-line change in `.env` rather than a code change.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, TypeVar

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

log = logging.getLogger("novel_agent")

Role = Literal["agent", "enrich", "judge"]
T = TypeVar("T")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Gemini chat models (see .env.example for free-tier notes)
    agent_model: str
    enrich_model: str
    judge_model: str
    agent_thinking_level: str
    enrich_thinking_level: str
    gemini_rpm: float
    # Embeddings + reranking
    embeddings_provider: str
    voyage_embed_model: str
    google_embed_model: str
    embed_batch_tokens: int
    embed_rpm: float
    rerank_enabled: bool
    voyage_rerank_model: str
    # Ingestion
    chunk_size: int
    chunk_overlap: int
    chapters_per_call: int
    # Paths
    data_dir: Path

    @property
    def embed_model(self) -> str:
        return self.voyage_embed_model if self.embeddings_provider == "voyage" else self.google_embed_model

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    def index_dir(self, book_id: str) -> Path:
        return self.data_dir / "index" / book_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = _env("EMBEDDINGS_PROVIDER", "voyage").lower()
    if provider not in {"voyage", "google"}:
        raise ValueError(f"EMBEDDINGS_PROVIDER must be 'voyage' or 'google', got {provider!r}")
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    return Settings(
        agent_model=_env("AGENT_MODEL", "gemini-3.5-flash-lite"),
        enrich_model=_env("ENRICH_MODEL", "gemini-3.5-flash-lite"),
        judge_model=_env("JUDGE_MODEL", "gemini-3.5-flash-lite"),
        agent_thinking_level=_env("AGENT_THINKING_LEVEL", "low"),
        enrich_thinking_level=_env("ENRICH_THINKING_LEVEL", ""),
        gemini_rpm=float(_env("GEMINI_RPM", "10")),
        embeddings_provider=provider,
        voyage_embed_model=_env("VOYAGE_EMBED_MODEL", "voyage-4"),
        google_embed_model=_env("GOOGLE_EMBED_MODEL", "gemini-embedding-001"),
        embed_batch_tokens=int(_env("EMBED_BATCH_TOKENS", "8000")),
        embed_rpm=float(_env("EMBED_RPM", "0")),
        rerank_enabled=_env_bool("RERANK_ENABLED", True) and provider == "voyage",
        voyage_rerank_model=_env("VOYAGE_RERANK_MODEL", "rerank-3-lite"),
        chunk_size=int(_env("CHUNK_SIZE", "1200")),
        chunk_overlap=int(_env("CHUNK_OVERLAP", "200")),
        chapters_per_call=int(_env("CHAPTERS_PER_CALL", "8")),
        data_dir=data_dir,
    )


# --------------------------------------------------------------------------------------
# Chat models (Gemini)
# --------------------------------------------------------------------------------------

_rate_limiters: dict[str, object] = {}


def _rate_limiter_for(model: str, rpm: float):
    """One token-bucket limiter per model: free-tier quotas are per model, per project."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    if rpm <= 0:
        return None
    if model not in _rate_limiters:
        _rate_limiters[model] = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0, check_every_n_seconds=0.1, max_bucket_size=2
        )
    return _rate_limiters[model]


def get_llm(role: Role = "agent"):
    """Gemini chat model for a given role.

    Gemini 3.x deprecates temperature/top_p/top_k, so none are set; depth of reasoning is
    controlled with `thinking_level` instead (blank = the model's default).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    model = {"agent": s.agent_model, "enrich": s.enrich_model, "judge": s.judge_model}[role]
    thinking = {"agent": s.agent_thinking_level, "enrich": s.enrich_thinking_level, "judge": ""}[role]
    kwargs: dict = {
        "model": model,
        "max_retries": 6,
        "timeout": 180,
        "rate_limiter": _rate_limiter_for(model, s.gemini_rpm),
    }
    if thinking:
        kwargs["thinking_level"] = thinking
    return ChatGoogleGenerativeAI(**kwargs)


# --------------------------------------------------------------------------------------
# Embeddings (Voyage by default, Gemini as fallback) with retries + throttling
# --------------------------------------------------------------------------------------

_RETRYABLE = re.compile(r"429|rate.?limit|resource.?exhausted|quota|timeout|timed out|503|502|500|unavailable", re.I)


def is_retryable(exc: Exception) -> bool:
    return bool(_RETRYABLE.search(f"{type(exc).__name__} {exc}"))


def with_retries(fn: Callable[[], T], *, what: str, max_attempts: int = 6, base_delay: float = 5.0) -> T:
    """Call `fn`, backing off exponentially on rate-limit / transient errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # provider SDKs raise different error types
            if attempt == max_attempts or not is_retryable(exc):
                raise
            delay = min(60.0, base_delay * 2 ** (attempt - 1))
            log.warning("%s failed (%s); retrying in %.0fs [%d/%d]", what, exc, delay, attempt, max_attempts)
            time.sleep(delay)
    raise AssertionError("unreachable")


class ResilientEmbeddings(Embeddings):
    """Wraps any LangChain `Embeddings` with retry/backoff, optional RPM throttling and a query cache.

    The Voyage wrapper has no retries of its own and the free trial allows only 3 requests/minute,
    so a single 429 would otherwise abort an ingestion run.
    """

    def __init__(self, inner: Embeddings, *, rpm: float = 0.0, name: str = "embeddings"):
        self.inner = inner
        self.name = name
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0
        self._lock = threading.Lock()
        self._query_cache: dict[str, list[float]] = {}

    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _call(self, fn: Callable[[], T], what: str) -> T:
        def attempt() -> T:
            self._throttle()
            return fn()

        return with_retries(attempt, what=f"{self.name}.{what}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(lambda: self.inner.embed_documents(texts), "embed_documents")

    def embed_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            self._query_cache[text] = self._call(lambda: self.inner.embed_query(text), "embed_query")
        return self._query_cache[text]


def get_embeddings() -> ResilientEmbeddings:
    s = get_settings()
    if s.embeddings_provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        inner = VoyageAIEmbeddings(model=s.voyage_embed_model, batch_size=128)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        inner = GoogleGenerativeAIEmbeddings(model=s.google_embed_model)
    return ResilientEmbeddings(inner, rpm=s.embed_rpm, name=f"{s.embeddings_provider}:{s.embed_model}")


def get_reranker():
    """Voyage reranker (returns all candidates re-ordered; callers slice), or None when disabled."""
    s = get_settings()
    if not s.rerank_enabled:
        return None
    from langchain_voyageai import VoyageAIRerank

    return VoyageAIRerank(model=s.voyage_rerank_model, top_k=None)
