"""
app.py
------
Universal Knowledge Assistant — Streamlit UI.
Orchestrates document ingestion, YouTube loading, and the
Agentic RAG pipeline (LangGraph ReAct agent with memory).

Changes from v1:
  - _handle_question() now calls run_agent() instead of
    retrieve_context() + web_search() directly.
  - Thread ID added to session state for conversation memory.
  - Agent tool-call trace shown in an expander for transparency.
  - All other UI, ingestion, and chat-history logic is unchanged.
"""

import logging
import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from config.config import APP_TITLE, APP_ICON, MAX_HISTORY
from utils.document_loader import load_file
from utils.youtube_loader import load_youtube
from utils.rag_pipeline import RAGPipeline, format_sources

# ── NEW: agent imports ────────────────────────────────────────────────────────
from utils.agent_manager import run_agent, new_thread_id

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "rag": RAGPipeline(),
        "chat_history": [],
        "indexed_sources": [],
        # ── NEW: unique thread ID per browser session ──
        "thread_id": new_thread_id(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# Helpers (UNCHANGED)
# ─────────────────────────────────────────────

def _add_to_index(chunks: list, label: str) -> None:
    st.session_state.rag.add_documents(chunks)
    if label not in st.session_state.indexed_sources:
        st.session_state.indexed_sources.append(label)


def _reset_knowledge_base() -> None:
    st.session_state.rag.clear()
    st.session_state.indexed_sources = []
    st.session_state.chat_history = []
    # ── NEW: reset memory thread so agent forgets old conversation ──
    st.session_state.thread_id = new_thread_id()
    st.rerun()


# ─────────────────────────────────────────────
# Sidebar (UNCHANGED except thread_id display)
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption("Turn your documents and videos into a queryable knowledge base.")
        st.divider()

        st.subheader("💬 Response Mode")
        mode = st.radio(
            "Select mode",
            options=["Concise", "Detailed"],
            horizontal=True,
            help="Concise: 2–4 sentence answers. Detailed: full explanations.",
        )

        st.divider()

        st.subheader("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, TXT, MD, CSV, JSON",
            type=["pdf", "docx", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files:
            if st.button("📥 Index uploaded files", use_container_width=True):
                _process_uploaded_files(uploaded_files)

        st.divider()

        st.subheader("🎬 Add YouTube Videos")
        yt_input = st.text_area(
            "One URL per line",
            placeholder="https://www.youtube.com/watch?v=...",
            height=100,
            key="yt_input",
        )
        if st.button("📥 Load YouTube transcripts", use_container_width=True):
            _process_youtube_urls(yt_input)

        st.divider()

        st.subheader("📚 Knowledge Base")
        chunk_count = st.session_state.rag.document_count()
        if chunk_count > 0:
            st.success(f"✅ {chunk_count} chunks indexed")
            for src in st.session_state.indexed_sources:
                st.markdown(f"- `{src}`")
        else:
            st.info("No documents indexed yet.")

        st.divider()

        if st.button("🗑️ Clear everything", use_container_width=True):
            _reset_knowledge_base()

        # ── NEW: show memory thread ID for debugging ──
        with st.expander("🔧 Debug", expanded=False):
            st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}…`")

        st.caption("Powered by Groq · LangGraph · FAISS · all-MiniLM-L6-v2")

    return {"mode": mode.lower()}


def _process_uploaded_files(uploaded_files) -> None:
    progress = st.sidebar.progress(0)
    total = len(uploaded_files)
    success_count = 0
    for i, f in enumerate(uploaded_files):
        with st.sidebar.status(f"Indexing {f.name}…"):
            try:
                chunks = load_file(file_bytes=f.read(), filename=f.name)
                _add_to_index(chunks, f.name)
                st.write(f"✅ {f.name} — {len(chunks)} chunks")
                success_count += 1
            except Exception as e:
                st.write(f"❌ {f.name}: {e}")
                logger.error("Failed to index '%s': %s", f.name, e)
        progress.progress((i + 1) / total)
    progress.empty()
    if success_count > 0:
        st.sidebar.success(f"Indexed {success_count}/{total} file(s).")


def _process_youtube_urls(raw_input: str) -> None:
    urls = [line.strip() for line in raw_input.splitlines() if line.strip()]
    if not urls:
        st.sidebar.warning("Please enter at least one YouTube URL.")
        return
    for url in urls:
        with st.sidebar.status(f"Loading: {url[:60]}…"):
            try:
                chunks = load_youtube(url)
                _add_to_index(chunks, url)
                st.write(f"✅ {url} — {len(chunks)} chunks")
            except Exception as e:
                st.write(f"❌ {url}: {e}")
                logger.error("Failed to load YouTube '%s': %s", url, e)


# ─────────────────────────────────────────────
# Chat interface (UNCHANGED except _handle_question)
# ─────────────────────────────────────────────

def render_chat(settings: dict) -> None:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Upload documents or add YouTube URLs in the sidebar, then ask questions below."
    )
    st.divider()

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("sources"):
                _render_sources(
                    entry["sources"],
                    entry.get("used_web", False),
                    entry.get("tool_calls", []),
                )

    question = st.chat_input("Ask a question about your knowledge base…")
    if question:
        _handle_question(question, settings["mode"])


def _render_sources(sources: list, used_web: bool, tool_calls: list = None) -> None:
    tool_calls = tool_calls or []
    label = "🌐 Web search used" if used_web else "📎 Sources from knowledge base"
    with st.expander(label, expanded=False):
        if used_web:
            st.info("Tavily / DuckDuckGo web search was used for this answer.")
        for src in sources:
            st.markdown(f"- {src}")
        # ── NEW: show agent tool trace ──
        if tool_calls:
            st.caption(f"🤖 Agent tools used: {', '.join(tool_calls)}")


# ─────────────────────────────────────────────
# ── CHANGED: _handle_question now calls run_agent()
# ─────────────────────────────────────────────

def _handle_question(question: str, mode: str) -> None:
    """
    Process a user question through the LangGraph ReAct agent.

    The agent autonomously decides whether to use:
      - document_retriever  (FAISS / local knowledge)
      - web_search          (Tavily or DuckDuckGo)
      - direct LLM reasoning

    Conversation memory is maintained across turns via thread_id.
    """
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    if len(st.session_state.chat_history) > MAX_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        source_placeholder = st.empty()

        with st.spinner("Agent is thinking…"):
            try:
                # ── SINGLE CALL replaces retrieve_context() + web_search() + generate_answer()
                result = run_agent(
                    question=question,
                    rag_pipeline=st.session_state.rag,
                    thread_id=st.session_state.thread_id,
                    mode=mode,
                )
                answer = result.answer
                sources = result.sources
                used_web = result.used_web
                tool_calls = result.tool_calls

            except Exception as e:
                answer = f"❌ Agent error: {e}"
                sources = []
                used_web = False
                tool_calls = []
                logger.error("Agent error: %s", e)

        answer_placeholder.markdown(answer)

        if sources or tool_calls:
            with source_placeholder.expander(
                "🌐 Web search used" if used_web else "📎 Sources from knowledge base",
                expanded=False,
            ):
                if used_web:
                    st.info("Tavily / DuckDuckGo web search was used for this answer.")
                for src in sources:
                    st.markdown(f"- {src}")
                if tool_calls:
                    st.caption(f"🤖 Agent tools used: {', '.join(tool_calls)}")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "used_web": used_web,
        "tool_calls": tool_calls,
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    settings = render_sidebar()
    render_chat(settings)


if __name__ == "__main__":
    main()
