#!/usr/bin/env python3
"""Download CoPilot transcripts from GCS for debugging.

Usage:
    poetry run python scripts/download_transcripts.py <session_id> [<session_id> ...]

Requires GCS credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth).
If user_id is unknown, set USER_ID env var or the script will try common paths.

Each transcript is saved to transcripts/<session_id>.jsonl
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Only allow alphanumeric, dash, underscore in session IDs to prevent path traversal
_SAFE_SESSION_RE = re.compile(r"[^0-9A-Za-z_-]")


def _safe_session_filename(session_id: str) -> str:
    """Sanitize session_id for use as a filename, preventing path traversal."""
    cleaned = _SAFE_SESSION_RE.sub("", session_id or "")
    if not cleaned:
        raise ValueError(f"Invalid session_id after sanitization: {session_id!r}")
    return cleaned


async def try_download(user_id: str, session_id: str) -> tuple[str, int] | None:
    """Try downloading a transcript for a given user_id + session_id."""
    from backend.copilot.sdk.transcript import download_transcript

    dl = await download_transcript(user_id, session_id)
    if dl and dl.content:
        return dl.content, dl.message_count
    return None


async def download_for_session(session_id: str, output_dir: str) -> None:
    """Download transcript for a session, trying multiple user_id strategies."""
    import json

    user_id = os.environ.get("USER_ID", "")

    if user_id:
        print(f"[{session_id[:12]}] Trying user_id={user_id[:12]}...")
        result = await try_download(user_id, session_id)
        if result:
            content, msg_count = result
            safe_sid = _safe_session_filename(session_id)
            out_path = os.path.join(output_dir, f"{safe_sid}.jsonl")
            with open(out_path, "w") as f:
                f.write(content)
            lines = len(content.strip().split("\n"))
            print(
                f"[{session_id[:12]}] Saved {len(content)} bytes, "
                f"{lines} entries, msg_count={msg_count}"
            )

            meta_path = os.path.join(output_dir, f"{safe_sid}.meta.json")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "message_count": msg_count,
                        "transcript_bytes": len(content),
                        "transcript_lines": lines,
                    },
                    f,
                    indent=2,
                )
            return

    print(f"[{session_id[:12]}] No transcript found. Set USER_ID env var.")


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_ids = sys.argv[1:]
    output_dir = os.path.join(os.path.dirname(__file__), "..", "transcripts")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {len(session_ids)} transcript(s) to {output_dir}/\n")
    for sid in session_ids:
        await download_for_session(sid, output_dir)
        print()


if __name__ == "__main__":
    asyncio.run(main())
