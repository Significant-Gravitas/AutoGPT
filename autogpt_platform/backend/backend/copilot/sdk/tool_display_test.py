import asyncio

import pytest

from backend.copilot.sdk.tool_display import SDKToolDisplayBridge
from backend.copilot.tool_display import emit_tool_display_name, tool_display_context


def test_display_context_uses_provider_id_and_keeps_arguments_clean():
    bridge = SDKToolDisplayBridge()
    original = {"library_agent_id": "agent-a"}
    tagged = bridge.prepare_call("run_agent", original, "toolu-a")
    with bridge.execution_context("run_agent", tagged) as arguments:
        assert arguments == original
        emit_tool_display_name("Daily Digest")
    [event] = bridge.drain()
    assert event.id == "toolu-a"
    assert event.data.toolCallId == "toolu-a"
    assert event.data.displayName == "Daily Digest"
    assert original == {"library_agent_id": "agent-a"}


@pytest.mark.asyncio
async def test_parallel_identical_calls_keep_exact_ids_when_started_in_reverse():
    bridge = SDKToolDisplayBridge()
    tagged_a = bridge.prepare_call("run_agent", {"preset_id": "p"}, "toolu-a")
    tagged_b = bridge.prepare_call("run_agent", {"preset_id": "p"}, "toolu-b")

    async def execute(tagged, name):
        with bridge.execution_context("run_agent", tagged):
            await asyncio.sleep(0)
            emit_tool_display_name(name)

    await asyncio.gather(execute(tagged_b, "Name B"), execute(tagged_a, "Name A"))
    assert {event.id: event.data.displayName for event in bridge.drain()} == {
        "toolu-a": "Name A",
        "toolu-b": "Name B",
    }


def test_reset_invalidates_old_tokens_and_callbacks():
    bridge = SDKToolDisplayBridge()
    tagged = bridge.prepare_call("run_agent", {}, "old")
    with bridge.execution_context("run_agent", tagged):
        bridge.reset()
        emit_tool_display_name("Late old task")
    with bridge.execution_context("run_agent", tagged):
        emit_tool_display_name("Replayed old token")
    assert bridge.drain() == []
    assert not bridge.ready.is_set()


def test_untrusted_or_wrong_tool_tokens_cannot_label_another_call():
    bridge = SDKToolDisplayBridge()
    tagged = bridge.prepare_call("run_agent", {}, "real")
    with bridge.execution_context("different_tool", tagged):
        emit_tool_display_name("Wrong")
    with bridge.execution_context("run_agent", {"__agpt_display_token": "forged"}):
        emit_tool_display_name("Forged")
    assert bridge.drain() == []


def test_exception_resets_context_and_token_is_single_use():
    bridge = SDKToolDisplayBridge()
    tagged = bridge.prepare_call("run_agent", {}, "real")
    with pytest.raises(RuntimeError):
        with bridge.execution_context("run_agent", tagged):
            raise RuntimeError("test")
    emit_tool_display_name("Outside failed execution")
    with bridge.execution_context("run_agent", tagged):
        emit_tool_display_name("Duplicate")
    assert bridge.drain() == []


def test_sdk_scopes_do_not_inherit_outer_tool_display_callback():
    names = []
    bridge = SDKToolDisplayBridge()
    with tool_display_context(names.append):
        with bridge.execution_context("run_agent", {}):
            emit_tool_display_name("Uncorrelated")
        emit_tool_display_name("Outer")
    assert names == ["Outer"]
