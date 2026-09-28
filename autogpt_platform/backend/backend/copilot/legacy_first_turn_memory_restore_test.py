"""Tests for the restore of a CLI session file an older session uploaded:
``legacy_first_turn_memory.strip_first_turn_memory_from_session`` on the
file's first user entry, and ``transcript.download_transcript``, which every
restore goes through."""

import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import transcript as transcript_module
from backend.copilot.legacy_first_turn_memory import (
    strip_first_turn_memory_from_session,
)
from backend.copilot.legacy_first_turn_memory_test_data import (
    ALICE,
    BUDGET_BLOCK,
    BUILDER_BLOCK,
    REST,
    SKILLS_BLOCK,
    bucket_storage,
    built_to_backtrack,
    legacy_first_message,
    session_file,
)
from backend.copilot.transcript import download_transcript


def _entries(content: bytes) -> list[dict]:
    return [json.loads(line) for line in content.splitlines()]


def _text(entry: dict) -> str:
    """The text of a user entry, a bare string or its text blocks."""
    message = entry["message"]["content"]
    if type(message) is str:
        return message
    return "".join(block.get("text", "") for block in message)


_OLD_QUERY = BUILDER_BLOCK + BUDGET_BLOCK + legacy_first_message(ALICE)


class TestStripFirstTurnMemoryFromSession:
    def test_an_old_upload_loses_the_block_and_nothing_else(self):
        """The first user entry recorded the first turn's query as sent, the
        engine's query-only blocks first. The block goes; those blocks, the
        stored message's other blocks, the user's words, every other field
        of the entry and every other line stay."""
        old = session_file(
            ("user", _OLD_QUERY),
            ("assistant", "done"),
            ("user", "and Bob?"),
            ("assistant", "Bob leads Atlas"),
        )

        restored = strip_first_turn_memory_from_session(old)

        before, after = _entries(old), _entries(restored)
        assert _text(after[0]) == BUILDER_BLOCK + BUDGET_BLOCK + SKILLS_BLOCK + REST
        assert "Alice works on Atlas" not in restored.decode()
        assert {k: v for k, v in after[0].items() if k != "message"} == {
            k: v for k, v in before[0].items() if k != "message"
        }
        assert restored.splitlines()[1:] == old.splitlines()[1:]

    def test_a_text_block_beside_an_image_is_stripped_too(self):
        image = {"type": "image", "source": {"type": "base64", "data": "AAA"}}
        old = session_file(
            ("user", [image, {"type": "text", "text": _OLD_QUERY}]),
            ("assistant", "done"),
        )

        first = _entries(strip_first_turn_memory_from_session(old))[0]

        assert first["message"]["content"][0] == image
        assert "memory_context" not in _text(first)
        assert _text(first).endswith("what is Alice working on")

    def test_a_tag_the_user_typed_in_the_first_entry_is_left(self):
        typed = session_file(
            ("user", f"see <memory_context>\n{ALICE}\n</memory_context>\n\nok"),
            ("assistant", "done"),
        )

        assert strip_first_turn_memory_from_session(typed) == typed

    def test_a_copy_pasted_into_a_later_entry_is_left(self):
        """Only the first user entry is the first turn's query; the same text
        in a later message is the user's."""
        pasted = session_file(
            ("user", "hello"),
            ("assistant", "hi"),
            ("user", legacy_first_message(ALICE)),
            ("assistant", "noted"),
        )

        assert strip_first_turn_memory_from_session(pasted) == pasted

    def test_only_the_first_entry_changes_when_both_hold_the_block(self):
        old = session_file(
            ("user", _OLD_QUERY),
            ("assistant", "done"),
            ("user", legacy_first_message(ALICE)),
            ("assistant", "noted"),
        )

        restored = strip_first_turn_memory_from_session(old)

        assert "memory_context" not in _text(_entries(restored)[0])
        assert restored.splitlines()[1:] == old.splitlines()[1:]

    def test_a_new_session_file_comes_back_as_it_was(self):
        new = session_file(
            ("user", SKILLS_BLOCK + REST),
            ("assistant", "done"),
        )

        assert strip_first_turn_memory_from_session(new) is new

    def test_lines_before_the_first_user_entry_are_skipped(self):
        old = session_file(("user", _OLD_QUERY), ("assistant", "done"))
        prefixed = b'{"type":"summary","summary":"t"}\nnot json\n' + old

        restored = strip_first_turn_memory_from_session(prefixed)

        assert restored.startswith(b'{"type":"summary","summary":"t"}\nnot json\n')
        assert b"Alice works on Atlas" not in restored

    def test_a_first_user_entry_of_another_shape_ends_the_search(self):
        """An entry of type user the models cannot read is still the first
        user entry: nothing after it is taken for the first turn's query."""
        odd = b'{"type":"user","message":{"role":"user"}}\n'
        later = session_file(("user", legacy_first_message(ALICE)), ("assistant", "x"))

        assert strip_first_turn_memory_from_session(odd + later) == odd + later

    def test_a_file_that_is_not_utf8_is_returned_as_is(self):
        broken = b"\xff\xfe<memory_context>\n"

        assert strip_first_turn_memory_from_session(broken) == broken

    def test_a_first_entry_built_to_backtrack_is_left_at_once(self):
        """A reply that mentions ``<memory_context>`` makes any new session's
        file worth a look; its first entry, built to split into leading
        blocks in 2^25 ways, is still read once and left."""
        built = session_file(
            ("user", built_to_backtrack(25)),
            ("assistant", "the tag is <memory_context>"),
        )
        started = time.perf_counter()

        assert strip_first_turn_memory_from_session(built) is built
        assert time.perf_counter() - started < 1.0


class TestDownloadTranscript:
    """Every restore, on either engine, goes through ``download_transcript``."""

    @pytest.mark.asyncio
    async def test_a_restore_takes_the_block_out(self, caplog):
        old = session_file(("user", _OLD_QUERY), ("assistant", "done"))
        caplog.set_level(logging.INFO, logger=transcript_module.__name__)

        with patch.object(
            transcript_module,
            "get_workspace_storage",
            new=AsyncMock(return_value=bucket_storage(old)),
        ):
            download = await download_transcript("user-1", "session-1")

        assert download is not None
        assert download.content == strip_first_turn_memory_from_session(old)
        assert download.content != old
        assert download.message_count == 2 and download.mode == "sdk"
        assert "Removed the first-turn memory block" in caplog.text

    @pytest.mark.asyncio
    async def test_a_new_session_restores_byte_for_byte(self, caplog):
        new = session_file(("user", SKILLS_BLOCK + REST), ("assistant", "done"))
        caplog.set_level(logging.INFO, logger=transcript_module.__name__)

        with patch.object(
            transcript_module,
            "get_workspace_storage",
            new=AsyncMock(return_value=bucket_storage(new)),
        ):
            download = await download_transcript("user-1", "session-1")

        assert download is not None and download.content == new
        assert "Removed the first-turn memory block" not in caplog.text
