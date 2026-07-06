"""
Supabase pgvector-backed idea memory using a LOCAL embedding model.

Uses sentence-transformers `all-MiniLM-L6-v2` (384-dim) — runs fully offline,
no API key needed. Model is downloaded once (~90 MB) and cached automatically.

Stores every successfully used "What If" idea as an embedding in Supabase.
Before accepting a new candidate, performs a nearest-neighbour cosine similarity
search — candidates above the threshold are rejected as too similar to past ideas.
"""

from __future__ import annotations
import uuid
from typing import Optional
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

from config import (
    LOCAL_EMBEDDING_MODEL,
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SUPABASE_RPC,
    SIMILARITY_THRESHOLD,
)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the local embedding model once and cache it for the process lifetime."""
    print(f"[embedding] Loading local model: {LOCAL_EMBEDDING_MODEL} ...")
    return SentenceTransformer(LOCAL_EMBEDDING_MODEL)


class IdeaVectorMemory:
    """
    Wraps Supabase + pgvector to provide semantic idea deduplication.

    Uses the local `all-MiniLM-L6-v2` model for embeddings — no cloud calls,
    no API key, fully offline after the first model download.

    Usage:
        memory = IdeaVectorMemory()

        if not memory.is_too_similar("What if humans could fly?"):
            memory.add_idea("What if humans could fly?")
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self._model: SentenceTransformer = _load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_idea(self, idea: str) -> None:
        """Embed an idea and insert it into Supabase."""
        embedding = self._embed(idea)
        self._client.table(SUPABASE_TABLE).insert(
            {
                "id": str(uuid.uuid4()),
                "idea": idea,
                "embedding": embedding,
            }
        ).execute()

    def is_too_similar(self, candidate: str) -> bool:
        """
        Returns True if `candidate` is semantically too close to any stored idea
        (cosine similarity > self.threshold).
        Returns False when the table is empty or no match exceeds the threshold.
        """
        sim = self.get_similarity(candidate)
        if sim is None:
            return False
        return sim >= self.threshold

    def get_similarity(self, candidate: str) -> Optional[float]:
        """Return the highest cosine similarity score against stored ideas, or None."""
        embedding = self._embed(candidate)
        return self._find_closest(embedding)

    def count(self) -> int:
        """Return total number of stored ideas."""
        result = (
            self._client.table(SUPABASE_TABLE).select("id", count="exact").execute()
        )
        return result.count or 0

    def get_recent_ideas(self, limit: int = 25) -> list[str]:
        """Fetch the most recently stored ideas."""
        try:
            result = (
                self._client.table(SUPABASE_TABLE)
                .select("idea")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return [row["idea"] for row in result.data] if result.data else []
        except Exception:
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Encode text with the local model and return a Python list of floats."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _find_closest(self, embedding: list[float]) -> Optional[float]:
        """
        Call the `match_ideas` Supabase RPC and return the highest similarity score.
        Returns None if no results or if the table is empty.
        """
        try:
            result = self._client.rpc(
                SUPABASE_RPC,
                {
                    "query_embedding": embedding,
                    "match_threshold": 0.0,  # fetch any match; filter in Python
                    "match_count": 1,
                },
            ).execute()

            if result.data:
                return float(result.data[0]["similarity"])
            return None
        except Exception:
            return None
