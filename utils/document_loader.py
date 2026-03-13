"""
utils/document_loader.py
------------------------
Load, extract text from, and chunk multiple document types:
PDF, DOCX, TXT, Markdown, CSV, JSON.

Each function returns a list of chunk dicts:
    {"text": str, "metadata": {"source": str, "type": str, ...}}
"""

import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, List

from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────
Chunk = Dict[str, Any]


# ─────────────────────────────────────────────
# Text splitter (no external dependency)
# ─────────────────────────────────────────────

def _split_text(
    text: str,
    source: str,
    doc_type: str,
    extra_meta: Dict[str, Any] = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split text into overlapping chunks and attach metadata.

    Args:
        text:       Full document text.
        source:     Filename / identifier shown to the user.
        doc_type:   e.g. "pdf", "docx", "txt", "csv", "json", "youtube"
        extra_meta: Additional metadata fields merged into each chunk.
        chunk_size: Max characters per chunk.
        overlap:    Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with keys "text" and "metadata".
    """
    text = text.strip()
    if not text:
        return []

    extra_meta = extra_meta or {}
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            meta = {"source": source, "type": doc_type, "chunk": chunk_index}
            meta.update(extra_meta)
            chunks.append({"text": chunk_text, "metadata": meta})
            chunk_index += 1
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────

def load_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Extract text from a PDF file page-by-page and return chunks.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks with page metadata.
    """
    try:
        import pypdf
    except ImportError as e:
        raise ImportError("pypdf is required. Run: pip install pypdf") from e

    chunks: List[Chunk] = []
    try:
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_chunks = _split_text(
                text=page_text,
                source=filename,
                doc_type="pdf",
                extra_meta={"page": page_num},
            )
            chunks.extend(page_chunks)
        logger.info("PDF '%s': extracted %d chunks from %d pages.", filename, len(chunks), len(reader.pages))
    except Exception as e:
        logger.error("Error loading PDF '%s': %s", filename, e)
        raise
    return chunks


# ─────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────

def load_docx(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Extract text from a Word document and return chunks.

    Args:
        file_bytes: Raw bytes of the uploaded DOCX.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks.
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx is required. Run: pip install python-docx") from e

    try:
        doc = Document(BytesIO(file_bytes))
        # Group paragraphs into sections separated by headings
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        chunks = _split_text(full_text, filename, "docx")
        logger.info("DOCX '%s': extracted %d chunks.", filename, len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error loading DOCX '%s': %s", filename, e)
        raise


# ─────────────────────────────────────────────
# TXT
# ─────────────────────────────────────────────

def load_txt(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Load a plain-text file and return chunks.

    Args:
        file_bytes: Raw bytes of the TXT file.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        chunks = _split_text(text, filename, "txt")
        logger.info("TXT '%s': extracted %d chunks.", filename, len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error loading TXT '%s': %s", filename, e)
        raise


# ─────────────────────────────────────────────
# Markdown
# ─────────────────────────────────────────────

def load_md(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Load a Markdown file and return chunks (treated as plain text).

    Args:
        file_bytes: Raw bytes of the Markdown file.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        chunks = _split_text(text, filename, "markdown")
        logger.info("MD '%s': extracted %d chunks.", filename, len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error loading MD '%s': %s", filename, e)
        raise


# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────

def load_csv(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Load a CSV file, convert each row to a readable text string, and chunk.

    Row format: "Column1: value1 | Column2: value2 | ..."

    Args:
        file_bytes: Raw bytes of the CSV file.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks, each representing one or more CSV rows.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required. Run: pip install pandas") from e

    try:
        df = pd.read_csv(BytesIO(file_bytes))
        row_texts = []
        for idx, row in df.iterrows():
            row_str = " | ".join(f"{col}: {val}" for col, val in row.items())
            row_texts.append(f"Row {idx + 1}: {row_str}")

        full_text = "\n".join(row_texts)
        chunks = _split_text(full_text, filename, "csv")
        logger.info("CSV '%s': %d rows → %d chunks.", filename, len(df), len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error loading CSV '%s': %s", filename, e)
        raise


# ─────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────

def load_json(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Load a JSON file and convert its contents to readable text, then chunk.

    Supports both JSON objects and JSON arrays.

    Args:
        file_bytes: Raw bytes of the JSON file.
        filename:   Original filename for metadata.

    Returns:
        List of text chunks.
    """
    try:
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))

        # Normalise to a list of items
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            items = [{"value": str(data)}]

        row_texts = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                row_str = " | ".join(f"{k}: {v}" for k, v in item.items())
            else:
                row_str = str(item)
            row_texts.append(f"Item {i + 1}: {row_str}")

        full_text = "\n".join(row_texts)
        chunks = _split_text(full_text, filename, "json")
        logger.info("JSON '%s': %d items → %d chunks.", filename, len(items), len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error loading JSON '%s': %s", filename, e)
        raise


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def load_file(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Auto-detect file type by extension and dispatch to the correct loader.

    Supported: .pdf, .docx, .txt, .md, .markdown, .csv, .json

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used for type detection and metadata).

    Returns:
        List of text chunks with metadata.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(filename)[-1].lower()
    dispatch = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_md,
        ".markdown": load_md,
        ".csv": load_csv,
        ".json": load_json,
    }
    if ext not in dispatch:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            "Supported: .pdf, .docx, .txt, .md, .csv, .json"
        )
    return dispatch[ext](file_bytes, filename)
