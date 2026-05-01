#!/usr/bin/env python3
"""Download CoPilot transcripts from prod GCS and load into local dev environment.

Usage:
    # Step 1: Download from prod GCS (needs MEDIA_GCS_BUCKET_NAME + gcloud auth)
    MEDIA_GCS_BUCKET_NAME=<prod-bucket> USER_ID=<user-uuid> \
        poetry run python scripts/download_transcripts.py download <session_id> ...

    # Step 2: Load downloaded transcripts into local storage + DB
    poetry run python scripts/download_transcripts.py load <session_id> ...

    # Or do both in one step (if you have GCS access):
    MEDIA_GCS_BUCKET_NAME=<prod-bucket> USER_ID=<user-uuid> \
        poetry run python scripts/download_transcripts.py both <session_id> ...

The "download" step saves transcripts to transcripts/<session_id>.jsonl.
The "load" step reads those files and:
  1. Creates a ChatSession in local DB (or reuses existing)
  2. Populates messages from the transcript
  3. Stores transcript in local workspace storage
  4. Creates metadata so --resume works on the next turn

After "load", you can send a message to the session via the CoPilot UI
and it will use --resume with the loaded transcript.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_SAFE_RE = re.compile(r"[^0-9a-fA-F-]")
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "transcripts")


def _sanitize(raw: str) -> str:
    cleaned = _SAFE_RE.sub("", raw or "")[:36]
    if not cleaned:
        raise ValueError(f"Invalid ID: {raw!r}")
    return cleaned


def _transcript_path(session_id: str) -> str:
    return os.path.join(TRANSCRIPTS_DIR, f"{_sanitize(session_id)}.jsonl")


def _meta_path(session_id: str) -> str:
    return os.path.join(TRANSCRIPTS_DIR, f"{_sanitize(session_id)}.meta.json")


# ── Download from GCS ─────────────────────────────────────────────────────


async def cmd_download(session_ids: list[str]) -> None:
    """Download transcripts from prod GCS to transcripts/ directory."""
    from backend.copilot.sdk.transcript import download_transcript

    user_id = os.environ.get("USER_ID", "")
    if not user_id:
        print("ERROR: Set USER_ID env var to the session owner's user ID.")
        print("  You can find it in Sentry breadcrumbs or the DB.")
        sys.exit(1)

    bucket = os.environ.get("MEDIA_GCS_BUCKET_NAME", "")
    if not bucket:
        print("ERROR: Set MEDIA_GCS_BUCKET_NAME to the prod GCS bucket.")
        sys.exit(1)

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    print(f"Downloading from GCS bucket: {bucket}")
    print(f"User ID: {user_id}\n")

    for sid in session_ids:
        print(f"[{sid[:12]}] Downloading...")
        try:
            dl = await download_transcript(user_id, sid)
        except Exception as e:
            print(f"[{sid[:12]}] Failed: {e}")
            continue

        if not dl or not dl.content:
            print(f"[{sid[:12]}] Not found in GCS")
            continue

        content_str = (
            dl.content.decode("utf-8") if isinstance(dl.content, bytes) else dl.content
        )
        out = _transcript_path(sid)
        with open(out, "w") as f:
            f.write(content_str)

        lines = len(content_str.strip().split("\n"))
        meta = {
            "session_id": sid,
            "user_id": user_id,
            "message_count": dl.message_count,
            "transcript_bytes": len(content_str),
            "transcript_lines": lines,
        }
        with open(_meta_path(sid), "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[{sid[:12]}] Saved: {lines} entries, "
            f"{len(content_str)} bytes, msg_count={dl.message_count}"
        )
    print("\nDone. Run 'load' command to import into local dev environment.")


# ── Load into local dev ───────────────────────────────────────────────────


def _parse_messages_from_transcript(content: str) -> list[dict]:
    """Extract user/assistant messages from JSONL transcript for DB seeding."""
    messages: list[dict] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg = entry.get("message", {})
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue

        content_blocks = msg.get("content", "")
        if isinstance(content_blocks, list):
            # Flatten content blocks to text
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            text = "\n".join(text_parts)
        elif isinstance(content_blocks, str):
            text = content_blocks
        else:
            text = ""

        if text:
            messages.append({"role": role, "content": text})

    return messages


async def cmd_load(session_ids: list[str]) -> None:
    """Load downloaded transcripts into local workspace storage + DB."""
    from backend.copilot.sdk.transcript import upload_transcript

    # Use the user_id from meta file or env var
    default_user_id = os.environ.get("USER_ID", "")

    for sid in session_ids:
        transcript_file = _transcript_path(sid)
        meta_file = _meta_path(sid)

        if not os.path.exists(transcript_file):
            print(f"[{sid[:12]}] No transcript file at {transcript_file}")
            print("  Run 'download' first, or place the file manually.")
            continue

        with open(transcript_file) as f:
            content = f.read()

        # Load meta if available
        user_id = default_user_id
        msg_count = 0
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            user_id = meta.get("user_id", user_id)
            msg_count = meta.get("message_count", 0)

        if not user_id:
            print(f"[{sid[:12]}] No user_id — set USER_ID env var or download first")
            continue

        lines = len(content.strip().split("\n"))
        print(f"[{sid[:12]}] Loading transcript: {lines} entries, {len(content)} bytes")

        # Parse messages from transcript for DB
        messages = _parse_messages_from_transcript(content)
        if not msg_count:
            msg_count = len(messages)
        print(f"[{sid[:12]}] Parsed {len(messages)} messages for DB")

        # Create chat session in DB
        try:
            from backend.copilot.db import create_chat_session, get_chat_session

            existing = await get_chat_session(sid)
            if existing:
                print(f"[{sid[:12]}] Session already exists in DB, skipping creation")
            else:
                await create_chat_session(sid, user_id)
                print(f"[{sid[:12]}] Created ChatSession in DB")
        except Exception as e:
            print(f"[{sid[:12]}] DB session creation failed: {e}")
            print("  You may need to create it manually or run with DB access.")

        # Add messages to DB
        if messages:
            try:
                from backend.copilot.db import add_chat_messages_batch

                msg_dicts = [
                    {"role": m["role"], "content": m["content"]} for m in messages
                ]
                await add_chat_messages_batch(sid, msg_dicts, start_sequence=0)
                print(f"[{sid[:12]}] Added {len(messages)} messages to DB")
            except Exception as e:
                print(f"[{sid[:12]}] Message insertion failed: {e}")
                print("  (Session may already have messages)")

        # Store transcript in local workspace storage
        try:
            await upload_transcript(
                user_id=user_id,
                session_id=sid,
                content=content.encode("utf-8"),
                message_count=msg_count,
            )
            print(f"[{sid[:12]}] Stored transcript in local workspace storage")
        except Exception as e:
            print(f"[{sid[:12]}] Transcript storage failed: {e}")

        # Also store directly to filesystem as fallback
        try:
            from backend.util.settings import Settings

            settings = Settings()
            storage_dir = settings.config.workspace_storage_dir or os.path.join(
                os.path.expanduser("~"), ".autogpt", "workspaces"
            )
            ts_dir = os.path.join(storage_dir, "chat-transcripts", _sanitize(user_id))
            os.makedirs(ts_dir, exist_ok=True)

            ts_path = os.path.join(ts_dir, f"{_sanitize(sid)}.jsonl")
            with open(ts_path, "w") as f:
                f.write(content)

            meta_storage = {
                "message_count": msg_count,
                "uploaded_at": time.time(),
            }
            meta_storage_path = os.path.join(ts_dir, f"{_sanitize(sid)}.meta.json")
            with open(meta_storage_path, "w") as f:
                json.dump(meta_storage, f)

            print(f"[{sid[:12]}] Also wrote to: {ts_path}")
        except Exception as e:
            print(f"[{sid[:12]}] Direct file write failed: {e}")

        print(f"[{sid[:12]}] Ready — send a message to this session to test")
        print()

    print("Done. Start the backend and send a message to the session(s).")
    print("The CoPilot will use --resume with the loaded transcript.")


# ── Main ──────────────────────────────────────────────────────────────────


async def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    session_ids = sys.argv[2:]

    if command == "download":
        await cmd_download(session_ids)
    elif command == "load":
        await cmd_load(session_ids)
    elif command == "both":
        await cmd_download(session_ids)
        print("\n" + "=" * 60 + "\n")
        await cmd_load(session_ids)
    else:
        print(f"Unknown command: {command}")
        print("Usage: download | load | both")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
