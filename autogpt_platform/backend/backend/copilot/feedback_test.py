"""Tests for reading the chat UI's message ids in ``copilot.feedback``."""

import pytest

from backend.copilot.feedback import sequence_from_ui_message_id

SESSION_ID = "8d2f5c1e-2b7a-4c55-9d0e-3f6a1b2c4d5e"


@pytest.mark.parametrize(
    ("message_id", "expected"),
    [
        (f"{SESSION_ID}-seq-0", 0),
        (f"{SESSION_ID}-seq-42", 42),
        (f"{SESSION_ID}-seq-999999999", 999_999_999),
        # A reply the UI has not re-keyed from the stream yet, or a row id.
        ("5b0c2a8e-4f1d-4c3b-9a7e-6d5f4e3c2b1a", None),
        # The idx fallback names a render position, not a row.
        (f"{SESSION_ID}-idx-3", None),
        # Another chat's reply never resolves against this one.
        ("11111111-2222-4333-8444-555555555555-seq-3", None),
        (f"{SESSION_ID}-seq-", None),
        (f"{SESSION_ID}-seq-3a", None),
        (f"{SESSION_ID}-seq--3", None),
        (f"{SESSION_ID}-seq-²", None),
        # Past int4: no chat is that long, and the lookup must not overflow.
        (f"{SESSION_ID}-seq-2147483648", None),
    ],
)
def test_sequence_from_ui_message_id(message_id: str, expected: int | None) -> None:
    assert sequence_from_ui_message_id(SESSION_ID, message_id) == expected
