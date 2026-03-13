"""
utils/agent_manager.py
----------------------
Agentic RAG layer built on LangGraph + LangChain.

Wraps the existing RAGPipeline and Tavily search into LangChain tools,
creates a ReAct agent backed by Groq, attaches MemorySaver for
per-conversation memory, and exposes a single entry point:

    run_agent(question, rag_pipeline, thread_id) -> AgentResult

The existing app.py pipeline (document indexing, YouTube ingestion,
UI, chat history) is completely untouched — only _handle_question()
needs to call run_agent() instead of the old direct calls.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.config import GROQ_API_KEY, GROQ_MODEL, MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result dataclass returned to app.py
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    """Structured result returned from run_agent()."""
    answer: str                          # Final text answer for display
    sources: List[str] = field(default_factory=list)   # Citation strings
    used_web: bool = False               # True if Tavily was the primary tool
    tool_calls: List[str] = field(default_factory=list)  # Names of tools invoked


# ─────────────────────────────────────────────
# Tool 1: RAG Retriever
# ─────────────────────────────────────────────

def _make_rag_tool(rag_pipeline):
    """
    Wrap the existing RAGPipeline.retrieve_context() as a LangChain Tool.

    The tool returns a plain string of retrieved context so the agent
    can reason over it. Source citations are extracted separately by
    the run_agent() function after the agent finishes.

    Args:
        rag_pipeline: Live RAGPipeline instance from st.session_state.

    Returns:
        A LangChain Tool (StructuredTool or Tool).
    """
    from langchain_core.tools import Tool
    from utils.rag_pipeline import format_sources

    def _retrieve(query: str) -> str:
        """Search the uploaded document knowledge base for relevant context."""
        if rag_pipeline.is_empty():
            return "NO_DOCUMENTS: The knowledge base is empty. No documents have been uploaded."

        context, chunks, needs_web = rag_pipeline.retrieve_context(query)

        if needs_web or not context:
            return "NO_MATCH: No relevant content found in uploaded documents for this query."

        # Attach source info at the top so the agent can cite it
        sources = format_sources(chunks)
        sources_str = "\n".join(f"- {s}" for s in sources)
        return f"[Sources]\n{sources_str}\n\n[Content]\n{context}"

    return Tool(
        name="rag_retrieve",
        func=_retrieve,
        description=(
            "Search the user's uploaded documents (PDFs, Word files, CSVs, "
            "YouTube transcripts, etc.) for information relevant to the question. "
            "Use this tool FIRST whenever the question might be answered by uploaded content. "
            "Input: the user's question or a focused search query."
        ),
    )


# ─────────────────────────────────────────────
# Tool 2: Tavily Web Search
# ─────────────────────────────────────────────

def _make_tavily_tool() -> Any:
    """
    Create a Tavily web-search tool.

    Falls back gracefully to DuckDuckGo if TAVILY_API_KEY is not set,
    so the agent still works without a paid key.

    Returns:
        A LangChain Tool instance.
    """
    import os
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if tavily_key:
        try:
            from langchain_tavily import TavilySearch
            tool = TavilySearch(max_results=MAX_SEARCH_RESULTS)
            logger.info("Tavily web search tool initialised.")
            return tool
        except ImportError:
            logger.warning("langchain-tavily not installed; falling back to DuckDuckGo tool.")

    # ── DuckDuckGo fallback ───────────────────────────────────────────────────
    from langchain_core.tools import Tool
    from utils.web_search import web_search as ddg_search

    def _ddg(query: str) -> str:
        """Search the web using DuckDuckGo."""
        result = ddg_search(query)
        return result if result else "No web results found."

    logger.info("Using DuckDuckGo as web search tool (no TAVILY_API_KEY set).")
    return Tool(
        name="web_search",
        func=_ddg,
        description=(
            "Search the internet for current information not found in uploaded documents. "
            "Use this when the document retriever returns no useful results, "
            "or when the question requires up-to-date external knowledge. "
            "Input: a concise web search query."
        ),
    )


# ─────────────────────────────────────────────
# LLM: Groq via LangChain
# ─────────────────────────────────────────────

def _make_groq_llm():
    """
    Initialise a LangChain-compatible Groq chat model.

    Uses langchain-groq which wraps the Groq Python SDK.

    Returns:
        ChatGroq instance.

    Raises:
        ImportError: If langchain-groq is not installed.
        ValueError:  If GROQ_API_KEY is not set.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "langchain-groq is required for the agent. "
            "Run: pip install langchain-groq"
        ) from e

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=2048,
    )


# ─────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────


_SYSTEM_PROMPT = """
You are an intelligent Universal Knowledge Assistant.

You have access to exactly two tools:

1. rag_retrieve
   - Searches the user's uploaded documents (PDFs, Word files, YouTube transcripts, CSVs, etc.).
   - Use this tool whenever the question might relate to uploaded content.

2. web_search
   - Searches the internet for external or current information.
   - Use this only when document_retriever returns NO_DOCUMENTS, NO_MATCH, or insufficient information.

Important rules:
- Only call tools that exist: document_retriever or web_search.
- Never invent new tool names.
- If the question is a greeting or simple conversation, answer directly without using tools.
- When information comes from document_retriever, cite the returned sources.
- Be clear, concise, and helpful.
"""

def _build_agent(rag_pipeline, memory):
    """
    Build a LangGraph ReAct agent with tools and memory.

    Args:
        rag_pipeline: Live RAGPipeline from session state.
        memory:       LangGraph MemorySaver instance.

    Returns:
        A compiled LangGraph agent (StateGraph).
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage

    llm = _make_groq_llm()
    tools = [
        _make_rag_tool(rag_pipeline),
        _make_tavily_tool(),
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=_SYSTEM_PROMPT,
    )
    logger.info("ReAct agent created with %d tools.", len(tools))
    return agent


# ─────────────────────────────────────────────
# MemorySaver singleton (shared across reruns)
# ─────────────────────────────────────────────

_memory_store: Optional[Any] = None


def _get_memory():
    """Return a cached MemorySaver instance."""
    global _memory_store
    if _memory_store is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            _memory_store = MemorySaver()
            logger.info("MemorySaver initialised.")
        except ImportError as e:
            raise ImportError(
                "langgraph is required. Run: pip install langgraph"
            ) from e
    return _memory_store


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def run_agent(
    question: str,
    rag_pipeline,
    thread_id: str,
    mode: str = "concise",
) -> AgentResult:
    """
    Run the agentic RAG pipeline for a single user question.

    The agent decides autonomously whether to use document retrieval,
    web search, or direct LLM reasoning. Conversation memory is
    maintained per thread_id, so follow-up questions work naturally.

    Args:
        question:     The user's question string.
        rag_pipeline: Live RAGPipeline instance (from st.session_state.rag).
        thread_id:    Unique ID for this conversation thread (from session state).
        mode:         "concise" or "detailed" — prepended to the question.

    Returns:
        AgentResult with answer, sources, used_web flag, and tool_calls list.
    """
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        raise ImportError(
            "langchain-core is required. Run: pip install langchain-core"
        ) from e

    # Inject response-mode instruction into the question
    mode_prefix = (
        "Answer concisely in 2-4 sentences. " if mode == "concise"
        else "Provide a thorough, well-structured explanation. "
    )
    augmented_question = f"{mode_prefix}\n\nQuestion: {question}"

    memory = _get_memory()
    agent = _build_agent(rag_pipeline, memory)

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=augmented_question)]},
            config=config,
        )
    except Exception as e:
        logger.error("Agent invocation failed: %s", e)
        raise

    # ── Parse the agent result ─────────────────────────────────────────────
    messages = result.get("messages", [])
    answer = ""
    tool_calls_used: List[str] = []
    sources: List[str] = []
    used_web = False

    # Walk messages to collect tool calls and the final AI answer
    for msg in messages:
        msg_type = type(msg).__name__

        # Track which tools were called
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                if tool_name:
                    tool_calls_used.append(tool_name)

        # ToolMessage contains the raw tool output — extract sources from RAG
        if msg_type == "ToolMessage":
            content = msg.content or ""
            tool_name = getattr(msg, "name", "") or ""

            if tool_name == "rag_retrieve" and "[Sources]" in content:
                # Parse source lines from the retriever's formatted output
                for line in content.splitlines():
                    if line.startswith("- "):
                        src = line[2:].strip()
                        if src and src not in sources:
                            sources.append(src)

            if tool_name in ("web_search", "tavily_search_results_json", "TavilySearch"):
                used_web = True
                sources = ["Web search (Tavily / DuckDuckGo)"]

    # The last AIMessage with no tool_calls is the final answer
    for msg in reversed(messages):
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            has_tc = bool(getattr(msg, "tool_calls", None))
            if not has_tc and msg.content:
                answer = msg.content.strip()
                break

    if not answer:
        answer = "I was unable to generate an answer. Please try rephrasing your question."

    # Deduplicate tool call names for display
    tool_calls_used = list(dict.fromkeys(tool_calls_used))

    return AgentResult(
        answer=answer,
        sources=sources,
        used_web=used_web,
        tool_calls=tool_calls_used,
    )


def new_thread_id() -> str:
    """Generate a fresh unique thread ID for a new conversation."""
    return str(uuid.uuid4())
