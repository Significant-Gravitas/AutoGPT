"""An approved held Google call's late result stays out of the traces."""

import json
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.google.gmail import GmailReadBlock
from backend.blocks.text import MatchTextPatternBlock
from backend.copilot.gate import held
from backend.copilot.google_data_redaction import (
    REDACTED,
    redact_google_data,
    remember_google_output,
)
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import combine_pending_with_current
from backend.copilot.pending_messages import (
    PendingMessage,
    format_pending_as_user_message,
)
from backend.copilot.trial_cost_context import TrialCostContext

EMAIL = "Subject: payroll -- the new salaries are attached"
# Longer than the trimmed fallback keeps, so no cut of it parses.
OUTPUT = json.dumps(
    {"type": "block_output", "provider": "google", "outputs": EMAIL + "." * 70_000}
)


def _cut(text: str) -> str:
    """A cap that leaves a preview which no longer parses."""
    return f"<tool-output-truncated total_chars={len(text)}>\n{text[:200]}"


def _failing(text: str) -> str:
    raise RuntimeError("cap failed")


async def _deliver(
    monkeypatch: pytest.MonkeyPatch,
    args: dict,
    output: str,
    cap: Callable[[str], str],
) -> PendingMessage:
    call = held.HeldCall(
        review_id="review-1",
        tool_name="run_block",
        tool_call_id="call-1",
        args=args,
        cost_context=TrialCostContext(user_id="user-1", trial_id=None),
    )
    monkeypatch.setattr(held, "answered", AsyncMock(return_value=[call]))
    monkeypatch.setattr(held, "_claim", AsyncMock(return_value=True))
    monkeypatch.setattr(
        held,
        "review_db",
        lambda: MagicMock(get_reviews_by_node_exec_ids=AsyncMock(return_value={})),
    )
    monkeypatch.setattr(held, "_due", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(held, "_run", AsyncMock(return_value=("approved", output)))
    delivered: list[PendingMessage] = []
    async for _status in held.resolve_answered(
        "user-1",
        ChatSession.new(user_id="user-1", dry_run=False),
        delivered.append,
        cap=cap,
    ):
        pass
    [row] = delivered
    return row


def _traced(row: PendingMessage) -> str:
    """The row as each engine hands it to the model, through the redaction:
    a user row of its own (baseline) and folded into the turn's message (SDK)."""
    folded = combine_pending_with_current([row], "go on", request_arrival_at=0)
    return json.dumps(
        redact_google_data(
            [
                format_pending_as_user_message(row),
                {"role": "user", "content": [{"type": "text", "text": folded}]},
            ]
        )
    )


@pytest.mark.parametrize("cap", [_cut, _failing])
async def test_a_held_google_call_result_is_redacted_however_it_was_cut(
    monkeypatch: pytest.MonkeyPatch, cap: Callable[[str], str]
):
    args = {"block_id": GmailReadBlock().id, "input_data": {}}
    row = await _deliver(monkeypatch, args, OUTPUT, cap)

    assert "Subject: payroll" in row.content
    traced = _traced(row)
    assert "Subject: payroll" not in traced
    assert REDACTED in traced


async def test_a_held_call_whose_output_was_google_data_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
):
    """``continue_run_block`` names a review, not the block; ``BaseTool``
    remembered the output the run produced."""
    output = f"a preview of an email: {EMAIL}"
    remember_google_output(output)
    row = await _deliver(monkeypatch, {"review_id": "rev-0"}, output, _cut)

    assert "Subject: payroll" not in _traced(row)


async def test_a_held_call_of_another_block_stays_readable(
    monkeypatch: pytest.MonkeyPatch,
):
    args = {"block_id": MatchTextPatternBlock().id, "input_data": {}}
    row = await _deliver(monkeypatch, args, "a harmless match", _cut)

    assert "a harmless match" in _traced(row)
