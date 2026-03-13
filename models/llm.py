"""
models/llm.py
-------------
Groq LLM client initialisation and answer generation.
Uses llama3-70b-8192 for all completions.
"""

import logging
from typing import Optional

from groq import Groq

from config.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# ── Lazy singleton client ─────────────────────────────────────────────────────
_client: Optional[Groq] = None


def get_client() -> Groq:
    """Return a cached Groq client, creating one if necessary."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your environment or .env file."
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Prompt templates ──────────────────────────────────────────────────────────

_CONCISE_SYSTEM = (
    "You are a precise knowledge assistant. "
    "Answer questions using ONLY the provided context. "
    "Keep answers short and direct (2–4 sentences). "
    "Do not hallucinate. If the context does not contain the answer, say so. "
    "At the end of your answer, list the sources used in the format:\n"
    "Sources:\n- <source name> — <page/timestamp/detail if available>"
)

_DETAILED_SYSTEM = (
    "You are a thorough knowledge assistant. "
    "Answer questions using ONLY the provided context. "
    "Provide a comprehensive, well-structured answer with explanations and examples where relevant. "
    "Do not hallucinate. If the context does not fully cover the question, acknowledge the gap. "
    "At the end of your answer, list the sources used in the format:\n"
    "Sources:\n- <source name> — <page/timestamp/detail if available>"
)


def _build_user_message(question: str, context: str) -> str:
    """Combine retrieved context and the user question into a single prompt."""
    return (
        f"### Retrieved Context\n\n{context}\n\n"
        f"### Question\n\n{question}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str, mode: str = "concise") -> str:
    """
    Generate an answer using the Groq LLM.

    Args:
        question: The user's question.
        context:  Retrieved text context (from RAG or web search).
        mode:     "concise" for short answers, "detailed" for in-depth ones.

    Returns:
        The assistant's answer as a plain string.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
        Exception:  Re-raises Groq API errors after logging.
    """
    system_prompt = _CONCISE_SYSTEM if mode == "concise" else _DETAILED_SYSTEM
    user_message = _build_user_message(question, context)

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,        # Low temperature for factual answers
            max_tokens=1024 if mode == "concise" else 2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Groq API error: %s", e)
        raise
