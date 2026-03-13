"""
utils/rag_pipeline.py
---------------------
FAISS-backed RAG pipeline.
Manages the vector store, document indexing, similarity search,
and the retrieve_context() logic that decides whether to use
local knowledge or fall back to web search.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import EMBEDDING_DIM, TOP_K, SIMILARITY_THRESHOLD
from models.embeddings import embed_text

logger = logging.getLogger(__name__)

Chunk = Dict[str, Any]


# ─────────────────────────────────────────────
# Vector store creation
# ─────────────────────────────────────────────

def create_vector_store():
    """
    Create a new FAISS IndexFlatL2 vector store.

    Returns:
        A fresh faiss.IndexFlatL2 instance.

    Raises:
        ImportError: If faiss-cpu is not installed.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu is required. Run: pip install faiss-cpu"
        ) from e

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    logger.info("Created FAISS IndexFlatL2 with dim=%d.", EMBEDDING_DIM)
    return index


# ─────────────────────────────────────────────
# RAGPipeline class
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline backed by FAISS.

    Responsibilities:
    - Accept document/video chunks and embed + index them.
    - On query, embed the question and retrieve the top-k most similar chunks.
    - Evaluate similarity scores against a threshold to decide whether
      to return local context or signal that web search is needed.
    """

    def __init__(self):
        self.index = create_vector_store()
        self.chunks: List[Chunk] = []       # Parallel list to FAISS vectors
        self.total_docs: int = 0            # Total source documents indexed

    # ── Indexing ──────────────────────────────

    def add_documents(self, chunks: List[Chunk]) -> None:
        """
        Embed a list of text chunks and add them to the FAISS index.

        Args:
            chunks: List of {"text": str, "metadata": dict} dicts.
        """
        if not chunks:
            logger.warning("add_documents called with empty chunk list.")
            return

        texts = [c["text"] for c in chunks]

        try:
            embeddings = embed_text(texts)           # (N, dim) float32
            self.index.add(embeddings)
            self.chunks.extend(chunks)
            self.total_docs += 1
            logger.info(
                "Indexed %d chunks (store total: %d).", len(chunks), len(self.chunks)
            )
        except Exception as e:
            logger.error("Error adding documents to index: %s", e)
            raise

    # ── Retrieval ─────────────────────────────

    def retrieve_context(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> Tuple[str, List[Chunk], bool]:
        """
        Retrieve the most relevant context for a query.

        Decision logic:
        - If the store is empty → signal web search needed.
        - If the best L2 distance exceeds the threshold → signal web search needed.
        - Otherwise → return formatted context from local knowledge base.

        Args:
            query:     The user's question.
            top_k:     Number of chunks to retrieve.
            threshold: Maximum acceptable L2 distance (lower = stricter).

        Returns:
            Tuple of:
                context_str  — formatted text ready for LLM injection
                source_chunks — list of raw chunk dicts (for citation display)
                used_web     — True if caller should fall back to web search
        """
        if self.is_empty():
            logger.info("Vector store is empty — web search will be used.")
            return "", [], True

        try:
            query_emb = embed_text([query])      # (1, dim)
            distances, indices = self.index.search(query_emb, min(top_k, len(self.chunks)))
        except Exception as e:
            logger.error("FAISS search error: %s", e)
            return "", [], True

        # Check if best match is above threshold
        best_distance = float(distances[0][0]) if len(distances[0]) > 0 else float("inf")
        if best_distance > threshold:
            logger.info(
                "Best similarity distance %.4f exceeds threshold %.4f — falling back to web search.",
                best_distance, threshold,
            )
            return "", [], True

        # Collect valid results
        retrieved: List[Chunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks) and dist <= threshold:
                chunk = self.chunks[idx]
                retrieved.append(chunk)

        if not retrieved:
            return "", [], True

        context_str = _format_context(retrieved)
        logger.info("Retrieved %d chunks (best dist=%.4f).", len(retrieved), best_distance)
        return context_str, retrieved, False

    # ── Utilities ─────────────────────────────

    def is_empty(self) -> bool:
        """Return True if no documents have been indexed."""
        return self.index.ntotal == 0

    def document_count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Reset the vector store and chunk registry."""
        self.index = create_vector_store()
        self.chunks = []
        self.total_docs = 0
        logger.info("Vector store cleared.")


# ─────────────────────────────────────────────
# Context formatter
# ─────────────────────────────────────────────

def _format_context(chunks: List[Chunk]) -> str:
    """
    Format a list of retrieved chunks into a single LLM-ready context string.

    Args:
        chunks: Retrieved DocumentChunk dicts.

    Returns:
        A multi-section string with source labels.
    """
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        extra = _format_meta_detail(meta)
        label = f"[Context {i}] Source: {source}{extra}"
        sections.append(f"{label}\n{chunk['text']}")
    return "\n\n".join(sections)


def _format_meta_detail(meta: Dict[str, Any]) -> str:
    """Format extra metadata fields (page, timestamp, etc.) as a short annotation."""
    parts = []
    if "page" in meta:
        parts.append(f"page {meta['page']}")
    if "timestamp" in meta:
        parts.append(f"timestamp {meta['timestamp']}")
    if parts:
        return f" — {', '.join(parts)}"
    return ""


# ─────────────────────────────────────────────
# Source citation formatter (for UI display)
# ─────────────────────────────────────────────

def format_sources(chunks: List[Chunk]) -> List[str]:
    """
    Generate a human-readable list of source citations from retrieved chunks.

    Args:
        chunks: List of retrieved chunk dicts.

    Returns:
        List of citation strings for display in the UI.
    """
    seen = set()
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        detail = _format_meta_detail(meta)
        citation = f"{source}{detail}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
