"""
config/config.py
----------------
Central configuration for the Universal Knowledge Assistant.
All values are loaded from environment variables with sensible defaults.
Never hardcode secrets — set GROQ_API_KEY in your environment or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384          # Dimension for all-MiniLM-L6-v2

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "3.0"))
# Lower L2 distance = more similar; threshold controls fallback to web search

# ── Web Search ────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "4"))

# ── NEW: Tavily (optional — falls back to DuckDuckGo if not set) ──────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE: str = "Universal Knowledge Assistant"
APP_ICON: str = "🧠"
MAX_HISTORY: int = 20
