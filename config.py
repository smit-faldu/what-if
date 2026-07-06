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
        "with a medium-low pitch and a fast, dynamic, and upbeat speaking rate. "
        "The delivery is extremely confident and intellectually authoritative, "
        "yet completely grounded by a natural curiosity and sharp, witty humor. "
        "He uses natural conversational pauses and expressive intonation to keep "
        "the listener hooked, making complex topics sound incredibly fun. "
        "Maintain a completely clean speaking voice: do not chuckle, laugh, sigh, or make any non-verbal noises."
    ),
)
