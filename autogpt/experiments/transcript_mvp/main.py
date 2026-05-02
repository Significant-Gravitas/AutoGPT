"""
Transcript MVP - Sandbox experiment (not integrated with AutoGPT core).

Pipeline:
    YouTube URL -> fetch transcript -> process by LLM -> print output

Usage:
    python autogpt/experiments/transcript_mvp/main.py <youtube_url> [prompt]

Example:
    python autogpt/experiments/transcript_mvp/main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

Requires:
    pip install youtube-transcript-api openai
    OPENAI_API_KEY environment variable set
"""

import os
import re
import sys

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from openai import OpenAI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract the 11-character video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def fetch_transcript(video_id: str) -> str:
    """Download and concatenate the transcript for a YouTube video."""
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        return " ".join(snippet.text for snippet in fetched)
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
    except VideoUnavailable:
        raise RuntimeError("Video is unavailable or private.")
    except CouldNotRetrieveTranscript as e:
        raise RuntimeError(f"Could not retrieve transcript: {e}")


def process_with_llm(transcript: str, instruction: str | None = None) -> str:
    """Send transcript to OpenAI and return the processed result."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "  $env:OPENAI_API_KEY = 'sk-...'"
        )

    client = OpenAI(api_key=api_key)

    system_prompt = instruction or (
        "You are a helpful assistant. Read the YouTube video transcript below "
        "and produce a concise summary with clear bullet points covering the "
        "main ideas, key takeaways, and any action items mentioned."
    )

    # Limit to ~12 000 chars to stay within token budget on gpt-3.5-turbo
    truncated = transcript[:12_000]
    if len(transcript) > 12_000:
        truncated += "\n\n[Transcript truncated for length]"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n\n{truncated}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(youtube_url: str, instruction: str | None = None) -> str:
    print(f"\n[1/3] Parsing URL: {youtube_url}")
    video_id = extract_video_id(youtube_url)
    print(f"      Video ID: {video_id}")

    print("[2/3] Fetching transcript...")
    transcript = fetch_transcript(video_id)
    print(f"      Transcript length: {len(transcript):,} characters")

    print("[3/3] Processing with LLM...")
    result = process_with_llm(transcript, instruction)

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(result)
    print("=" * 60 + "\n")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]
    custom_instruction = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(url, custom_instruction)
