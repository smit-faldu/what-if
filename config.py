"""
Central configuration for the What If pipeline.
All values are read from environment variables (or a .env file).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Local Embedding Model ─────────────────────────────────────────────────────
# Uses sentence-transformers all-MiniLM-L6-v2 — runs fully locally, no API key.
# Downloads ~90 MB on first run and caches to ~/.cache/huggingface/
LOCAL_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimension

# ── Supabase ─────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE: str = "what_if_ideas"
SUPABASE_RPC: str = "match_ideas"

# ── Similarity threshold (0.0 – 1.0) ────────────────────────────────────────
# Candidates with similarity > this value against any stored idea are rejected.
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))

# ── LangGraph SQLite checkpointer ────────────────────────────────────────────
DB_PATH: str = str(ROOT_DIR / "langgraph_state.db")

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Brainstorm count ──────────────────────────────────────────────────────────
DEFAULT_IDEA_COUNT: int = int(os.getenv("IDEA_COUNT", "10"))

# ── Qwen3 TTS ─────────────────────────────────────────────────────────────────
# Set TTS_ENABLED=false to skip audio generation on CPU-only machines.
# The pipeline will still save tts_script.txt for manual Colab use.
TTS_ENABLED: bool = os.getenv("TTS_ENABLED", "true").lower() == "true"
TTS_MODEL: str    = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_SPEAKER: str  = os.getenv("TTS_SPEAKER", "Ryan")
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "English")
TTS_INSTRUCT: str = os.getenv(
    "TTS_INSTRUCT",
    (
        "A charismatic, highly engaging male voice in his early thirties, "
        "with a medium-low pitch. Speak quickly and briskly, noticeably faster "
        "than a normal conversational pace, with minimal gaps between words. "
        "The delivery is confident and intellectually authoritative, energetic and upbeat. "
        "Keep the exact same pitch, energy, volume, and speaking rate from the very first "
        "word to the very last word — do not slow down, speed up, soften, or get quieter "
        "at any point, and do not treat any part of the line as calmer or more dramatic "
        "than the rest. "
        "Speak at a strong, clearly audible volume throughout. "
        "Maintain a completely clean, plain speaking voice: do not chuckle, laugh, gasp, "
        "sigh, whisper, or make any non-verbal noises of any kind."
    ),
)

# ── Flux Image API ────────────────────────────────────────────────────────────
# Self-hosted Cloudflare Workers wrapping flux-2-klein-4b.
# FLUX_API_URLS — comma-separated list of worker endpoints.
#   Requests are distributed round-robin across all URLs.
#   To add more workers, just append another URL to the list.
# Set FLUX_ENABLED=false to skip image generation (prompts are still saved).
_raw_urls: str = os.getenv(
    "FLUX_API_URLS",
    os.getenv("FLUX_API_URL", ""),   # backwards-compat fallback
)
FLUX_API_URLS: list[str] = [
    u.strip().rstrip("/")
    for u in _raw_urls.split(",")
    if u.strip()
]
FLUX_API_TOKEN: str  = os.environ["FLUX_API_TOKEN"]   # required — add to .env
FLUX_ENABLED:   bool = os.getenv("FLUX_ENABLED", "true").lower() == "true"

# ── OpenAI Whisper (subtitle generation) ──────────────────────────────────────
# Model sizes (accuracy ↑ / speed ↓): tiny < base < small < medium < large-v3
# "base" is the default — fast on CPU, good enough for clean TTS audio.
# Set WHISPER_MODEL=small or larger for more accurate transcription.
# Set WHISPER_ENABLED=false to skip subtitle generation entirely.
WHISPER_MODEL:   str  = os.getenv("WHISPER_MODEL",   "base")
WHISPER_ENABLED: bool = os.getenv("WHISPER_ENABLED", "true").lower() == "true"