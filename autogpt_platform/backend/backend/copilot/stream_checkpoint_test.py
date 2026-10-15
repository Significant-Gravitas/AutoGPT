from backend.copilot.model import ChatMessage
from backend.copilot.stream_checkpoint import rows_digest, turn_checkpoint


def test_a_checkpoint_waits_for_the_tool_call_back_fill():
    rows = [
        ChatMessage(role="user", content="Go", sequence=0),
        ChatMessage(
            role="assistant",
            content="",
            sequence=1,
            tool_calls=[{"id": "c1", "function": {"name": "bash_exec"}}],
            tool_calls_pending_save=True,
        ),
        ChatMessage(role="tool", content="ok", tool_call_id="c1", sequence=2),
    ]

    held = turn_checkpoint(rows, 1)
    rows[1].tool_calls_pending_save = False
    landed = turn_checkpoint(rows, 1)

    assert held is None
    assert landed is not None
    assert (landed.rows, landed.sequence) == (2, 1)
    assert landed.digest == rows_digest(rows[1:])
