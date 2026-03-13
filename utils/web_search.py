"""
utils/web_search.py
-------------------
Live web search using DuckDuckGo (no API key required).
This module is the fallback when the vector store has no relevant results.
"""

import logging
from typing import List

from config.config import MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Perform a DuckDuckGo text search and return formatted results as a string
    suitable for injection into an LLM prompt.

    Args:
        query:       The search query (typically the user's question).
        max_results: Maximum number of results to include.

    Returns:
        A formatted string of search results, or an empty string on failure.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise ImportError(
            "duckduckgo-search is required. Run: pip install duckduckgo-search"
        ) from e

    logger.info("Web search: %r (max_results=%d)", query, max_results)

    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)

        if not results:
            logger.warning("Web search returned no results for: %r", query)
            return ""

        # Format results for the LLM
        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "Untitled")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted.append(
                f"[Web Result {i}] {title}\n"
                f"URL: {href}\n"
                f"{body}"
            )

        context = "\n\n".join(formatted)
        logger.info("Web search returned %d results.", len(results))
        return context

    except Exception as e:
        logger.error("Web search error: %s", e)
        return ""
