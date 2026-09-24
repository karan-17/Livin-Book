"""Voyage AI embeddings that stay inside the FREE tier.

Without a payment method, Voyage allows only 3 requests and 10,000 tokens per minute. Plain
`VoyageAIEmbeddings` sends the whole book in one big request and is rejected. This drop-in
replacement does three things:

  1. splits the texts into small requests of at most 3,000 tokens (counted locally, which is not an API call),
  2. before every request, waits until the last 60 seconds contain fewer than 3 requests and at most
     9,000 tokens including the new one (a "sliding window" rate limiter with a safety margin).
     A lock makes parallel calls (the agent may run several searches at once) wait their turn,
  3. if Voyage still answers "rate limited", backs off for 1, 2, 3... minutes and retries.
     Rejected requests seem to count against the limit too, so retrying quickly would never recover.

Embedding a whole novel (~170K tokens) takes about 20 minutes this way: slow, but free and error-free.
Every search is also one small request, so on the free tier a search may pause for up to ~20 seconds.
If you add a payment method to your Voyage account, pass free_tier=False.
"""

import threading
import time

import voyageai
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import PrivateAttr

REQUESTS_PER_MINUTE = 3
TOKENS_PER_MINUTE = 9_000  # Voyage's limit is 10,000; keep a margin
MAX_TOKENS_PER_REQUEST = 3_000
WINDOW_SECONDS = 60.0
MAX_ATTEMPTS = 6


class FreeTierVoyageEmbeddings(VoyageAIEmbeddings):
    free_tier: bool = True
    _recent: list = PrivateAttr(default_factory=list)  # [time, tokens] of requests in the last minute
    _blocked_until: float = PrivateAttr(default=0.0)  # set after a rejection
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def count_tokens(self, texts: list[str]) -> int:
        return self._client.count_tokens(texts, model=self.model)

    def batches(self, docs: list) -> list[list]:
        """Group Documents into lists that each fit in one request."""
        limit = MAX_TOKENS_PER_REQUEST if self.free_tier else 100_000
        groups, current, used = [], [], 0
        for doc in docs:
            tokens = self.count_tokens([doc.page_content])
            if current and used + tokens > limit:
                groups.append(current)
                current, used = [], 0
            current.append(doc)
            used += tokens
        return groups + [current] if current else groups

    def _reserve(self, tokens: int) -> list:
        """Wait until a request of `tokens` fits in the window, then book a slot for it."""
        with self._lock:  # one caller at a time, so parallel searches can't overbook the window
            while True:
                now = time.monotonic()
                self._recent = [r for r in self._recent if now - r[0] < WINDOW_SECONDS]
                used = sum(r[1] for r in self._recent)
                if now >= self._blocked_until and len(self._recent) < REQUESTS_PER_MINUTE \
                        and used + tokens <= TOKENS_PER_MINUTE:
                    slot = [now, tokens]
                    self._recent.append(slot)
                    return slot
                oldest_expires = min(r[0] for r in self._recent) + WINDOW_SECONDS if self._recent else now
                wait = max(self._blocked_until, oldest_expires) - now + 0.5
                if wait >= 2:
                    print(f"  ⏳ Voyage free tier: waiting {wait:.0f}s")
                time.sleep(wait)

    def _send(self, request, tokens: int):
        if not self.free_tier:
            return request()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            slot = self._reserve(tokens)
            try:
                return request()
            except voyageai.error.RateLimitError:
                if attempt == MAX_ATTEMPTS:
                    raise
                print(f"  ⏳ Voyage says 'slow down': backing off {attempt} min")
                self._blocked_until = time.monotonic() + 60 * attempt
            finally:
                slot[0] = time.monotonic()  # count the request from when it finished (the safe side)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embed = super().embed_documents
        limit = MAX_TOKENS_PER_REQUEST if self.free_tier else 100_000
        vectors, start = [], 0
        while start < len(texts):  # split into requests that respect the per-request limit
            end, used = start, 0
            while end < len(texts):
                tokens = self.count_tokens([texts[end]])
                if end > start and used + tokens > limit:
                    break
                used += tokens
                end += 1
            batch = texts[start:end]
            vectors.extend(self._send(lambda: embed(batch), used))
            start = end
        return vectors

    def embed_query(self, text: str) -> list[float]:
        embed = super().embed_query
        return self._send(lambda: embed(text), self.count_tokens([text]))
