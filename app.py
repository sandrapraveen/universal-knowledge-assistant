"""
app.py
------
Universal Knowledge Assistant — Streamlit UI.
Orchestrates document ingestion, YouTube loading, RAG retrieval,
web search fallback, and LLM answer generation.
"""

import logging
import sys
import os

import streamlit as st

# Ensure project root is on the path (needed when running via streamlit run)
sys.path.insert(0, os.path.dirname(__file__))

from config.config import APP_TITLE, APP_ICON, MAX_HISTORY
from models.llm import generate_answer
from utils.document_loader import load_file
from utils.youtube_loader import load_youtube
from utils.web_search import web_search
from utils.rag_pipeline import RAGPipeline, format_sources

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
        "chat_history": [],        # [{"role": "user"|"assistant", "content": str, "sources": list}]
        "indexed_sources": [],     # Display names of all indexed sources
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _add_to_index(chunks: list, label: str) -> None:
    """Index chunks and record the source label."""
    st.session_state.rag.add_documents(chunks)
    if label not in st.session_state.indexed_sources:
        st.session_state.indexed_sources.append(label)


def _reset_knowledge_base() -> None:
    """Clear the vector store and all session state."""
    st.session_state.rag.clear()
    st.session_state.indexed_sources = []
    st.session_state.chat_history = []
    st.rerun()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the sidebar and return UI settings."""
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption("Turn your documents and videos into a queryable knowledge base.")
        st.divider()

        # ── Response Mode ─────────────────────────────
        st.subheader("💬 Response Mode")
        mode = st.radio(
            "Select mode",
            options=["Concise", "Detailed"],
            horizontal=True,
            help="Concise: 2–4 sentence answers. Detailed: full explanations.",
        )

        st.divider()

        # ── Document Upload ───────────────────────────
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

        # ── YouTube URLs ──────────────────────────────
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

        # ── Knowledge base status ─────────────────────
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

        st.caption("Powered by Groq · FAISS · all-MiniLM-L6-v2 · DuckDuckGo")

    return {"mode": mode.lower()}


def _process_uploaded_files(uploaded_files) -> None:
    """Load and index a batch of uploaded files."""
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
    """Parse and index YouTube URLs from the text area."""
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
# Chat interface
# ─────────────────────────────────────────────

def render_chat(settings: dict) -> None:
    """Render the main chat area."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Upload documents or add YouTube URLs in the sidebar, then ask questions below."
    )
    st.divider()

    # ── Chat history ──────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            # Show sources only for assistant turns
            if entry["role"] == "assistant" and entry.get("sources"):
                _render_sources(entry["sources"], entry.get("used_web", False))

    # ── Input ─────────────────────────────────
    question = st.chat_input("Ask a question about your knowledge base…")
    if question:
        _handle_question(question, settings["mode"])


def _render_sources(sources: list, used_web: bool) -> None:
    """Render the source citations in an expander."""
    label = "🌐 Web search results used" if used_web else "📎 Sources from knowledge base"
    with st.expander(label, expanded=False):
        if used_web:
            st.info("No relevant content was found in your documents. Web search was used.")
        else:
            for src in sources:
                st.markdown(f"- {src}")


def _handle_question(question: str, mode: str) -> None:
    """
    Process a user question through the RAG pipeline.

    Pipeline:
    1. Embed question → search FAISS
    2. If good match → use local context
    3. If no match → fall back to web search
    4. Send context to Groq LLM
    5. Display answer + sources
    """
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Trim history
    if len(st.session_state.chat_history) > MAX_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        source_placeholder = st.empty()

        with st.spinner("Thinking…"):

            # ── Step 1: RAG retrieval ──────────────────
            context, retrieved_chunks, needs_web = st.session_state.rag.retrieve_context(question)
            used_web = False
            sources = []

            # ── Step 2: Web search fallback ───────────
            if needs_web:
                with st.spinner("🔍 Searching the web…"):
                    try:
                        context = web_search(question)
                        used_web = True
                        sources = ["Web search (DuckDuckGo)"]
                    except Exception as e:
                        context = ""
                        logger.error("Web search failed: %s", e)
            else:
                sources = format_sources(retrieved_chunks)

            # ── Step 3: Generate answer ────────────────
            if not context:
                answer = (
                    "⚠️ I could not find any relevant information in your knowledge base "
                    "or via web search. Please try rephrasing your question or uploading "
                    "more relevant documents."
                )
            else:
                try:
                    answer = generate_answer(
                        question=question,
                        context=context,
                        mode=mode,
                    )
                except Exception as e:
                    answer = f"❌ LLM error: {e}"
                    logger.error("LLM generation error: %s", e)

        # ── Display ────────────────────────────────
        answer_placeholder.markdown(answer)
        if sources:
            with source_placeholder.expander(
                "🌐 Web search results used" if used_web else "📎 Sources from knowledge base",
                expanded=False,
            ):
                if used_web:
                    st.info("No relevant content was found in your documents. Web search was used.")
                else:
                    for src in sources:
                        st.markdown(f"- {src}")

    # Append assistant turn to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "used_web": used_web,
    })


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    settings = render_sidebar()
    render_chat(settings)


if __name__ == "__main__":
    main()
