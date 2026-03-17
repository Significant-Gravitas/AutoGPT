"""Shared test fixtures for copilot SDK tests."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from backend.util import json


@pytest.fixture()
def mock_chat_config():
    """Mock ChatConfig so compact_transcript tests skip real config lookup."""
    with patch(
        "backend.copilot.config.ChatConfig",
        return_value=type("Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"})(),
    ):
        yield


def build_test_transcript(pairs: list[tuple[str, str]]) -> str:
    """Build a minimal valid JSONL transcript from (role, content) pairs.

    Use this helper in any copilot SDK test that needs a well-formed
    transcript without hitting the real storage layer.
    """
    lines: list[str] = []
    last_uuid: str | None = None
    for role, content in pairs:
        uid = str(uuid4())
        entry_type = "assistant" if role == "assistant" else "user"
        msg: dict = {"role": role, "content": content}
        if role == "assistant":
            msg.update(
                {
                    "model": "",
                    "id": f"msg_{uid[:8]}",
                    "type": "message",
                    "content": [{"type": "text", "text": content}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                }
            )
        entry = {
            "type": entry_type,
            "uuid": uid,
            "parentUuid": last_uuid,
            "message": msg,
        }
        lines.append(json.dumps(entry, separators=(",", ":")))
        last_uuid = uid
    return "\n".join(lines) + "\n"
