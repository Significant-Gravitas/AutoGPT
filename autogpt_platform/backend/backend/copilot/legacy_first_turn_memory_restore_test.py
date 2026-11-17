"""Tests for the restore of a CLI session file an older session uploaded:
``legacy_session_file.restore_session_file``, and
``transcript.download_transcript``, which every restore goes through."""

import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import transcript as transcript_module
from backend.copilot.legacy_first_turn_memory_test_data import (
    ALICE,
    BUDGET_BLOCK,
    BUILDER_BLOCK,
    RENDERER_IMPOSSIBLE,
    REST,
    SKILLS_BLOCK,
    USER_AUTHORED_BLOCK,
    bucket_storage,
    built_to_backtrack,
    folded_query,
    history_query,
    legacy_first_message,
    master_warm,
    session_file,
)
from backend.copilot.legacy_session_file import restore_session_file
from backend.copilot.transcript import download_transcript


def _entries(content: bytes) -> list[dict]:
    return [json.loads(line) for line in content.splitlines()]


def _text(entry: dict) -> str:
    """The text of a user entry, a bare string or its text blocks."""
    message = entry["message"]["content"]
    if type(message) is str:
        return message
    return "".join(block.get("text", "") for block in message)


_FIRST = legacy_first_message(master_warm(("Alice works on Atlas",)))
_OLD_QUERY = BUILDER_BLOCK + BUDGET_BLOCK + _FIRST
# First queries whose block no renderer wrote, where the platform put its
# own: the review's counterexample, then each body of ``RENDERER_IMPOSSIBLE``
# behind the engine's query-only blocks, as a CLI session file records them.
_UNPROVEN = {
    "review-counterexample": USER_AUTHORED_BLOCK,
    **{
        name: BUILDER_BLOCK
        + BUDGET_BLOCK
        + legacy_first_message(f"<temporal_context>\n{body}\n</temporal_context>")
        for name, body in RENDERER_IMPOSSIBLE.items()
    },
}


class TestTheFirstQuery:
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

        restored = restore_session_file(old)

        assert restored.stripped and restored.content is not None
        before, after = _entries(old), _entries(restored.content)
        assert _text(after[0]) == BUILDER_BLOCK + BUDGET_BLOCK + SKILLS_BLOCK + REST
        assert b"Alice works on Atlas" not in restored.content
        assert {k: v for k, v in after[0].items() if k != "message"} == {
            k: v for k, v in before[0].items() if k != "message"
        }
        assert restored.content.splitlines()[1:] == old.splitlines()[1:]

    def test_a_crlf_file_keeps_its_line_endings(self):
        old = session_file(("user", _OLD_QUERY), ("assistant", "done")).replace(
            b"\n", b"\r\n"
        )

        content = restore_session_file(old).content

        assert content is not None and b"Alice works" not in content
        assert content.count(b"\r\n") == old.count(b"\r\n")
        assert b"\n" not in content.replace(b"\r\n", b"")

    def test_a_text_block_beside_an_image_is_stripped_too(self):
        image = {"type": "image", "source": {"type": "base64", "data": "AAA"}}
        old = session_file(
            ("user", [image, {"type": "text", "text": _OLD_QUERY}]),
            ("assistant", "done"),
        )

        content = restore_session_file(old).content

        assert content is not None
        first = _entries(content)[0]
        assert first["message"]["content"][0] == image
        assert "memory_context" not in _text(first)

    @pytest.mark.parametrize("first", _UNPROVEN.values(), ids=_UNPROVEN)
    def test_a_block_the_renderer_never_wrote_is_left_and_resumed(self, first):
        """Not provably the platform's, so not stripped; where the platform
        would have put it, so not a reason to drop the file either."""
        typed = session_file(("user", first), ("assistant", "ok"))

        restored = restore_session_file(typed)

        assert restored.content is typed and not restored.stripped

    def test_only_the_first_entry_changes_when_a_later_one_holds_a_paste(self):
        old = session_file(
            ("user", _OLD_QUERY),
            ("assistant", "done"),
            ("user", _FIRST),
            ("assistant", "noted"),
        )

        content = restore_session_file(old).content

        assert content is not None
        assert "memory_context" not in _text(_entries(content)[0])
        assert content.splitlines()[1:] == old.splitlines()[1:]


class TestCopiesItCannotStrip:
    """The file is not resumed from; the turn rebuilds from the database."""

    def test_a_pending_message_folded_in_front_of_the_block(self):
        old = session_file(("user", folded_query(_FIRST)), ("assistant", "ok"))

        restored = restore_session_file(old)

        assert restored.content is None
        assert "pending messages" in restored.reason

    @pytest.mark.parametrize("later", [False, True], ids=["first-entry", "later-entry"])
    def test_a_history_rebuilt_from_the_database(self, later):
        """A turn without ``--resume`` sent the stored first message inside
        its ``<conversation_history>``; its CLI session starts from that
        query, or a stale resume sent it later."""
        entries = [("user", history_query(_FIRST)), ("assistant", "ok")]
        if later:
            entries = [("user", "hi"), ("assistant", "hello"), *entries]

        restored = restore_session_file(session_file(*entries))

        assert restored.content is None
        assert "history" in restored.reason

    def test_a_history_line_without_its_prefix(self):
        """The copy right after the history's opening tag, with no ``User:``
        (the shape the review's probe used) is dropped all the same."""
        query = f"<conversation_history>\n{_FIRST}\n</conversation_history>"
        old = session_file(("user", _FIRST), ("assistant", "ok"), ("user", query))

        assert restore_session_file(old).content is None

    @pytest.mark.parametrize(
        "cut",
        [
            lambda text: text[: text.index("Alice works") + 5] + " … what is",
            lambda text: "<available_skills>\nSkil … " + text[text.index("Alice") :],
        ],
        ids=["head-kept", "tail-kept"],
    )
    def test_a_history_the_old_compression_cut_short(self, cut):
        """Compression kept the head and the tail of a long first message;
        either end of the block still marks the copy."""
        query = history_query(cut(_FIRST))

        assert restore_session_file(session_file(("user", query))).content is None


class TestTheUsersOwnText:
    """Typed or pasted, anywhere the platform did not write the block: never
    stripped and never a reason to drop the file."""

    @pytest.mark.parametrize(
        "entries",
        [
            [("user", f"see <memory_context>\n{ALICE}\n</memory_context>\n\nok")],
            [("user", "hello"), ("assistant", "hi"), ("user", _FIRST)],
            [("user", "hello"), ("assistant", "hi"), ("user", "look:\n\n" + _FIRST)],
            [
                ("user", "hi"),
                ("assistant", "yo"),
                ("user", "x " + history_query(_FIRST)),
            ],
        ],
        ids=[
            "typed-mid-message",
            "pasted-later",
            "pasted-later-paragraph",
            "typed-history-later",
        ],
    )
    def test_is_kept_as_it_is(self, entries):
        typed = session_file(*entries, ("assistant", "noted"))

        restored = restore_session_file(typed)

        assert restored.content is typed and not restored.stripped

    def test_a_new_session_file_comes_back_as_it_was(self):
        new = session_file(("user", SKILLS_BLOCK + REST), ("assistant", "done"))

        assert restore_session_file(new).content is new


class TestReading:
    def test_lines_before_the_first_user_entry_are_skipped(self):
        old = session_file(("user", _OLD_QUERY), ("assistant", "done"))
        prefixed = b'{"type":"summary","summary":"t"}\nnot json\n' + old

        content = restore_session_file(prefixed).content

        assert content is not None
        assert content.startswith(b'{"type":"summary","summary":"t"}\nnot json\n')
        assert b"Alice works on Atlas" not in content

    def test_a_first_user_entry_of_another_shape_ends_the_search(self):
        """An entry of type user the models cannot read is still the first
        user entry: nothing after it is taken for the first turn's query."""
        odd = b'{"type":"user","message":{"role":"user"}}\n'
        later = session_file(("user", _FIRST), ("assistant", "x"))

        assert restore_session_file(odd + later).content == odd + later

    def test_a_file_that_is_not_utf8_is_returned_as_is(self):
        broken = b"\xff\xfe<memory_context>\n"

        assert restore_session_file(broken).content == broken

    def test_a_first_entry_built_to_backtrack_is_read_at_once(self):
        """A reply that mentions ``<memory_context>`` makes any new session's
        file worth a look; its first entry, built to split into leading
        blocks in 2^25 ways, is still read once and left."""
        built = session_file(
            ("user", built_to_backtrack(25)),
            ("assistant", "the tag is <memory_context>"),
        )
        started = time.perf_counter()

        assert restore_session_file(built).content is built
        assert time.perf_counter() - started < 1.0


class TestDownloadTranscript:
    """Every restore, on either engine, goes through ``download_transcript``."""

    async def _download(self, content: bytes, caplog):
        caplog.set_level(logging.INFO, logger=transcript_module.__name__)
        with patch.object(
            transcript_module,
            "get_workspace_storage",
            new=AsyncMock(return_value=bucket_storage(content)),
        ):
            return await download_transcript("user-1", "session-1")

    @pytest.mark.asyncio
    async def test_a_restore_takes_the_block_out(self, caplog):
        old = session_file(("user", _OLD_QUERY), ("assistant", "done"))

        download = await self._download(old, caplog)

        assert download is not None
        assert download.content == restore_session_file(old).content
        assert download.message_count == 2 and download.mode == "sdk"
        assert "Removed the first-turn memory block" in caplog.text

    @pytest.mark.asyncio
    async def test_a_file_it_cannot_clean_is_not_restored(self, caplog):
        old = session_file(("user", folded_query(_FIRST)), ("assistant", "ok"))

        assert await self._download(old, caplog) is None
        lines = [r for r in caplog.records if "Not resuming" in r.getMessage()]
        assert len(lines) == 1 and lines[0].levelno == logging.WARNING

    @pytest.mark.asyncio
    @pytest.mark.parametrize("first", _UNPROVEN.values(), ids=_UNPROVEN)
    async def test_a_block_no_renderer_wrote_restores_byte_for_byte(
        self, first, caplog
    ):
        typed = session_file(("user", first), ("assistant", "ok"))

        download = await self._download(typed, caplog)

        assert download is not None and download.content == typed
        assert "first-turn memory" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_new_session_restores_byte_for_byte(self, caplog):
        new = session_file(("user", SKILLS_BLOCK + REST), ("assistant", "done"))

        download = await self._download(new, caplog)

        assert download is not None and download.content == new
        assert "first-turn memory" not in caplog.text
