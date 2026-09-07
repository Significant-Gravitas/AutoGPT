"""Tool display metadata stays scoped to one concurrent execution."""

import asyncio

import pytest

from backend.copilot.tool_display import (
    emit_tool_display_name,
    tool_calls_for_provider,
    tool_display_context,
)


@pytest.mark.asyncio
async def test_display_names_are_isolated_between_concurrent_calls():
    received: dict[str, list[str]] = {"first": [], "second": []}
    both_started = asyncio.Barrier(2)

    async def execute(call_id: str, name: str):
        with tool_display_context(received[call_id].append):
            await both_started.wait()
            emit_tool_display_name(name)

    await asyncio.gather(execute("first", "Daily report"), execute("second", "Inbox"))
    emit_tool_display_name("Outside any tool")
    assert received == {"first": ["Daily report"], "second": ["Inbox"]}


def test_display_context_restores_parent_after_failure_and_ignores_blank_names():
    outer: list[str] = []
    inner: list[str] = []
    with tool_display_context(outer.append):
        with pytest.raises(RuntimeError), tool_display_context(inner.append):
            emit_tool_display_name("  Inner  ")
            raise RuntimeError("execution failed")
        emit_tool_display_name("   ")
        emit_tool_display_name("Outer")
    emit_tool_display_name("Outside any tool")
    assert outer == ["Outer"]
    assert inner == ["Inner"]


def test_provider_tool_calls_omit_display_metadata_without_mutating_saved_calls():
    saved = [
        {
            "id": "call-1",
            "display_name": "Daily report",
            "function": {"name": "run_agent", "arguments": '{"display_name":"input"}'},
        }
    ]
    provider = tool_calls_for_provider(saved)
    assert "display_name" not in provider[0]
    assert provider[0]["function"]["arguments"] == '{"display_name":"input"}'
    assert saved[0]["display_name"] == "Daily report"
