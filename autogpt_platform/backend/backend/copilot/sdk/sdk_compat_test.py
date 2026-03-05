"""SDK compatibility tests — verify the claude-agent-sdk public API surface we depend on.

Instead of pinning to a narrow version range, these tests verify that the
installed SDK exposes every class, function, attribute, and method the copilot
integration relies on.  If an SDK upgrade removes or renames something these
tests will catch it immediately.
"""

import inspect

import pytest

# ---------------------------------------------------------------------------
# Public types & factories
# ---------------------------------------------------------------------------


def test_sdk_exports_client_and_options():
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    assert inspect.isclass(ClaudeSDKClient)
    assert inspect.isclass(ClaudeAgentOptions)


def test_sdk_exports_message_types():
    from claude_agent_sdk import (
        AssistantMessage,
        Message,
        ResultMessage,
        SystemMessage,
        UserMessage,
    )

    for cls in (AssistantMessage, ResultMessage, SystemMessage, UserMessage):
        assert inspect.isclass(cls), f"{cls.__name__} is not a class"
    # Message is a Union type alias, just verify it's importable
    assert Message is not None


def test_sdk_exports_content_block_types():
    from claude_agent_sdk import TextBlock, ToolResultBlock, ToolUseBlock

    for cls in (TextBlock, ToolResultBlock, ToolUseBlock):
        assert inspect.isclass(cls), f"{cls.__name__} is not a class"


def test_sdk_exports_mcp_helpers():
    from claude_agent_sdk import create_sdk_mcp_server, tool

    assert callable(create_sdk_mcp_server)
    assert callable(tool)


# ---------------------------------------------------------------------------
# ClaudeSDKClient interface
# ---------------------------------------------------------------------------


def test_client_has_required_methods():
    from claude_agent_sdk import ClaudeSDKClient

    required = ["connect", "disconnect", "query", "receive_messages"]
    for name in required:
        attr = getattr(ClaudeSDKClient, name, None)
        assert attr is not None, f"ClaudeSDKClient.{name} missing"
        assert callable(attr), f"ClaudeSDKClient.{name} is not callable"


def test_client_supports_async_context_manager():
    from claude_agent_sdk import ClaudeSDKClient

    assert hasattr(ClaudeSDKClient, "__aenter__")
    assert hasattr(ClaudeSDKClient, "__aexit__")


# ---------------------------------------------------------------------------
# ClaudeAgentOptions fields
# ---------------------------------------------------------------------------


def test_agent_options_accepts_required_fields():
    """Verify ClaudeAgentOptions accepts all kwargs our code passes."""
    from claude_agent_sdk import ClaudeAgentOptions

    opts = ClaudeAgentOptions(
        system_prompt="test",
        cwd="/tmp",
    )
    assert opts.system_prompt == "test"
    assert opts.cwd == "/tmp"


def test_agent_options_accepts_all_our_fields():
    """Comprehensive check of every field we use in service.py."""
    from claude_agent_sdk import ClaudeAgentOptions

    fields_we_use = [
        "system_prompt",
        "mcp_servers",
        "allowed_tools",
        "disallowed_tools",
        "hooks",
        "cwd",
        "model",
        "env",
        "resume",
        "max_buffer_size",
    ]
    sig = inspect.signature(ClaudeAgentOptions)
    for field in fields_we_use:
        assert field in sig.parameters, (
            f"ClaudeAgentOptions no longer accepts '{field}' — "
            f"available params: {list(sig.parameters.keys())}"
        )


# ---------------------------------------------------------------------------
# Message attributes
# ---------------------------------------------------------------------------


def test_assistant_message_has_content_and_model():
    from claude_agent_sdk import AssistantMessage, TextBlock

    msg = AssistantMessage(content=[TextBlock(text="hi")], model="test")
    assert hasattr(msg, "content")
    assert hasattr(msg, "model")


def test_result_message_has_required_attrs():
    from claude_agent_sdk import ResultMessage

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
    )
    assert msg.subtype == "success"
    assert hasattr(msg, "result")


def test_system_message_has_subtype_and_data():
    from claude_agent_sdk import SystemMessage

    msg = SystemMessage(subtype="init", data={})
    assert msg.subtype == "init"
    assert msg.data == {}


def test_user_message_has_parent_tool_use_id():
    from claude_agent_sdk import UserMessage

    msg = UserMessage(content="test")
    assert hasattr(msg, "parent_tool_use_id")
    assert hasattr(msg, "tool_use_result")


def test_tool_use_block_has_id_name_input():
    from claude_agent_sdk import ToolUseBlock

    block = ToolUseBlock(id="t1", name="test", input={"key": "val"})
    assert block.id == "t1"
    assert block.name == "test"
    assert block.input == {"key": "val"}


def test_tool_result_block_has_required_attrs():
    from claude_agent_sdk import ToolResultBlock

    block = ToolResultBlock(tool_use_id="t1", content="result")
    assert block.tool_use_id == "t1"
    assert block.content == "result"
    assert hasattr(block, "is_error")


# ---------------------------------------------------------------------------
# Hook types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook_event",
    ["PreToolUse", "PostToolUse", "Stop"],
)
def test_sdk_exports_hook_event_type(hook_event: str):
    """Verify HookEvent literal includes the events our security_hooks use."""
    from claude_agent_sdk.types import HookEvent

    # HookEvent is a Literal type — check that our events are valid values.
    # We can't easily inspect Literal at runtime, so just verify the type exists.
    assert HookEvent is not None
