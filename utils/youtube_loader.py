"""
utils/youtube_loader.py
-----------------------
Fetch YouTube video transcripts and convert them into text chunks
with timestamp metadata.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

Chunk = Dict[str, Any]


# ─────────────────────────────────────────────
# Video ID extraction
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from a variety of URL formats.

    Supports:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtu.be/VIDEO_ID
        https://youtube.com/shorts/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID

    Args:
        url: Any YouTube URL string.

    Returns:
        11-character video ID string, or None if not found.
    """
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ─────────────────────────────────────────────
# Timestamp formatter
# ─────────────────────────────────────────────

def _seconds_to_timestamp(seconds: float) -> str:
    """Convert a seconds float to a human-readable MM:SS or HH:MM:SS string."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ─────────────────────────────────────────────
# Transcript fetcher
# ─────────────────────────────────────────────

def _fetch_transcript(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a YouTube video.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        List of transcript segment dicts: {"text": str, "start": float, "duration": float}

    Raises:
        Exception: If the transcript cannot be fetched.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise ImportError(
            "youtube-transcript-api is required. Run: pip install youtube-transcript-api"
        ) from e

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    segments = [
        {
            "text": seg.text,
            "start": seg.start,
            "duration": seg.duration,
        }
        for seg in transcript
    ]

    return segments

# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def _chunk_transcript(
    segments: List[Dict],
    video_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Group transcript segments into text chunks of approximately chunk_size characters,
    recording the start timestamp of each chunk.

    Args:
        segments:   Raw transcript segments from YouTubeTranscriptApi.
        video_id:   Video ID used for metadata sourcing.
        chunk_size: Target characters per chunk.
        overlap:    Overlap by re-including recent segments in the next chunk.

    Returns:
        List of chunk dicts with "text" and "metadata" keys.
    """
    chunks: List[Chunk] = []
    current_text = ""
    current_start: Optional[float] = None
    chunk_index = 0

    for segment in segments:
        seg_text = segment.get("text", "").strip()
        seg_start = segment.get("start", 0.0)

        if current_start is None:
            current_start = seg_start

        current_text += " " + seg_text

        if len(current_text) >= chunk_size:
            timestamp = _seconds_to_timestamp(current_start)
            chunks.append({
                "text": current_text.strip(),
                "metadata": {
                    "source": f"YouTube: {video_id}",
                    "type": "youtube",
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "chunk": chunk_index,
                },
            })
            chunk_index += 1
            # Overlap: retain the tail of the current chunk
            overlap_text = current_text[-overlap:] if overlap < len(current_text) else current_text
            current_text = overlap_text
            current_start = seg_start  # reset window start

    # Flush remaining text
    if current_text.strip():
        timestamp = _seconds_to_timestamp(current_start or 0)
        chunks.append({
            "text": current_text.strip(),
            "metadata": {
                "source": f"YouTube: {video_id}",
                "type": "youtube",
                "video_id": video_id,
                "timestamp": timestamp,
                "chunk": chunk_index,
            },
        })

    return chunks


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_youtube(url: str) -> List[Chunk]:
    """
    Load a YouTube video transcript from a URL and return text chunks.

    Args:
        url: Full YouTube video URL.

    Returns:
        List of text chunks with timestamp metadata.

    Raises:
        ValueError: If the video ID cannot be extracted from the URL.
        Exception:  If the transcript is unavailable or the API call fails.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(
            f"Could not extract a video ID from URL: {url!r}. "
            "Please provide a valid YouTube URL."
        )

    logger.info("Fetching transcript for video ID: %s", video_id)
    try:
        segments = _fetch_transcript(video_id)
        chunks = _chunk_transcript(segments, video_id)
        logger.info(
            "YouTube '%s': %d segments → %d chunks.", video_id, len(segments), len(chunks)
        )
        return chunks
    except Exception as e:
        logger.error("Failed to load YouTube transcript for '%s': %s", video_id, e)
        raise
