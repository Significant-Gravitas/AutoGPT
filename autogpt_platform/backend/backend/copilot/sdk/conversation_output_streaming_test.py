import pytest
from langsmith import tracing_context

from . import conversation_output_test as fixtures
from .conversation_output_test import assistant, consume, event, result

blocked = fixtures.blocked
client = fixtures.client
script = fixtures.script
exported = fixtures.exported

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("prefix", ["", "hello "])
@pytest.mark.parametrize(
    "terminal,reason",
    [
        (result(), "end_turn"),
        (result(is_error=True, subtype="error_max_turns"), "error_max_turns"),
        (result(terminal_reason="aborted_streaming"), "cancelled"),
    ],
)
async def test_summary_completes_partial_text(
    client, script, exported, prefix, terminal, reason
):
    script.extend(
        [
            event("message_start", message={"id": "msg-1"}),
            event(
                "content_block_start",
                index=0,
                content_block={"type": "text", "text": ""},
            ),
            event(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": prefix},
            ),
            assistant("hello world"),
            terminal,
        ]
    )
    with tracing_context(enabled=True):
        await consume(client)
    assert exported[0]["outputs"] == {"output": "hello world", "stop_reason": reason}


@pytest.mark.parametrize("combined_partial", [False, True])
async def test_separate_block_summaries_reconcile_in_order(
    client, script, exported, combined_partial
):
    script.extend(
        [
            event("message_start", message={"id": "msg-1"}),
            event(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "hello "},
            ),
        ]
    )
    if combined_partial:
        script.append(
            event(
                "content_block_delta",
                index=1,
                delta={"type": "text_delta", "text": "world"},
            )
        )
    script.extend([assistant("hello "), assistant("world"), result()])
    with tracing_context(enabled=True):
        await consume(client)
    assert exported[0]["outputs"] == {
        "output": "hello world",
        "stop_reason": "end_turn",
    }


async def test_summary_replaces_divergent_partial_text(client, script, exported):
    script.extend(
        [
            event("message_start", message={"id": "msg-1"}),
            assistant("first block "),
            event(
                "content_block_delta",
                index=1,
                delta={"type": "text_delta", "text": "draft"},
            ),
            assistant("final block"),
            result(),
        ]
    )
    with tracing_context(enabled=True):
        await consume(client)
    assert exported[0]["outputs"] == {
        "output": "first block final block",
        "stop_reason": "end_turn",
    }
