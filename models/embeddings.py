"""
models/embeddings.py
--------------------
Embedding model using sentence-transformers/all-MiniLM-L6-v2.
Provides embed_text() — the single entry point for all embedding needs.
"""

import logging
from typing import List, Optional

import numpy as np

from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Lazy singleton ────────────────────────────────────────────────────────────
_model = None


def _get_model():
    """Load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def embed_text(text_list: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.

    Args:
        text_list: List of text strings to embed.

    Returns:
        2-D numpy array of shape (len(text_list), embedding_dim),
        dtype float32 — compatible with FAISS IndexFlatL2.

    Raises:
        ValueError: If text_list is empty.
        Exception:  Re-raises model errors after logging.
    """
    if not text_list:
        raise ValueError("embed_text received an empty list.")

    try:
        model = _get_model()
        embeddings = model.encode(
            text_list,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings.astype("float32")
    except Exception as e:
        logger.error("Embedding error: %s", e)
        raise
