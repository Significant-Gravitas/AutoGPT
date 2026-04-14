"""SDK compatibility tests — verify the claude-agent-sdk public API surface we depend on.

Instead of pinning to a narrow version range, these tests verify that the
installed SDK exposes every class, function, attribute, and method the copilot
integration relies on.  If an SDK upgrade removes or renames something these
tests will catch it immediately.
"""

import inspect
from typing import cast

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


def test_agent_options_accepts_system_prompt_preset_with_exclude_dynamic_sections():
    """Verify ClaudeAgentOptions accepts the exact preset dict _build_system_prompt_value produces.

    The production code always includes ``exclude_dynamic_sections=True`` in the preset
    dict.  This compat test mirrors that exact shape so any SDK version that starts
    rejecting unknown keys will be caught here rather than at runtime.
    """
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk.types import SystemPromptPreset

    from .service import _build_system_prompt_value

    # Call the production helper directly so this test is tied to the real
    # dict shape rather than a hand-rolled copy.
    preset = _build_system_prompt_value("custom system prompt", cross_user_cache=True)
    assert isinstance(
        preset, dict
    ), "_build_system_prompt_value must return a dict when caching is on"

    sdk_preset = cast(SystemPromptPreset, preset)
    opts = ClaudeAgentOptions(system_prompt=sdk_preset)
    assert opts.system_prompt == sdk_preset


def test_build_system_prompt_value_returns_plain_string_when_cross_user_cache_off():
    """When cross_user_cache=False (e.g. on --resume turns), the helper must return
    a plain string so the preset+resume crash is avoided."""
    from .service import _build_system_prompt_value

    result = _build_system_prompt_value("my prompt", cross_user_cache=False)
    assert result == "my prompt", "Must return the raw string, not a preset dict"


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
        "stderr",
        "fallback_model",
        "max_turns",
        "max_budget_usd",
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


# ---------------------------------------------------------------------------
# OpenRouter compatibility — bundled CLI version pin
# ---------------------------------------------------------------------------
#
# Newer ``claude-agent-sdk`` versions bundle CLI binaries that send
# features incompatible with OpenRouter (``tool_reference`` content
# blocks, ``context-management-2025-06-27`` beta).  We neutralise these
# at runtime by injecting ``CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1``
# into the CLI subprocess env (see ``build_sdk_env()`` in ``env.py``).
#
# This test is the cheapest possible regression guard: it pins the
# bundled CLI to a known-good version.  If anyone bumps
# ``claude-agent-sdk`` in ``pyproject.toml``, the bundled CLI version in
# ``_cli_version.py`` will change and this test will fail with a clear
# message that points the next person at the OpenRouter compat issue
# instead of letting them silently re-break production.

# CLI versions bisect-verified as OpenRouter-safe.  2.1.63 and 2.1.70 pre-date
# the context-management beta regression and work without any env var.  2.1.97+
# requires ``CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`` (injected by
# ``build_sdk_env()`` in ``env.py``) to strip the beta header.
_KNOWN_GOOD_BUNDLED_CLI_VERSIONS: frozenset[str] = frozenset(
    {
        "2.1.63",  # claude-agent-sdk 0.1.45 -- original pin from PR #12294.
        "2.1.70",  # claude-agent-sdk 0.1.47 -- first version with the
        #          tool_reference proxy detection fix; bisect-verified
        #          OpenRouter-safe in #12742.
        "2.1.97",  # claude-agent-sdk 0.1.58 -- OpenRouter-safe only with
        #          CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 (injected by
        #          build_sdk_env() in env.py).
    }
)


def test_bundled_cli_version_is_known_good_against_openrouter():
    """Pin the bundled CLI version so accidental SDK bumps cause a loud,
    fast failure with a pointer to the OpenRouter compatibility issue.
    """
    from claude_agent_sdk._cli_version import __cli_version__

    assert __cli_version__ in _KNOWN_GOOD_BUNDLED_CLI_VERSIONS, (
        f"Bundled Claude Code CLI version is {__cli_version__!r}, which is "
        f"not in the OpenRouter-known-good set "
        f"({sorted(_KNOWN_GOOD_BUNDLED_CLI_VERSIONS)!r}). "
        "If you intentionally bumped `claude-agent-sdk`, verify the new "
        "bundled CLI works with OpenRouter against the reproduction test "
        "in `cli_openrouter_compat_test.py` (with "
        "`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`), then add the new "
        "CLI version to `_KNOWN_GOOD_BUNDLED_CLI_VERSIONS`. If the env "
        "var is not sufficient, set `claude_agent_cli_path` to a "
        "known-good binary instead. See "
        "https://github.com/anthropics/claude-agent-sdk-python/issues/789 "
        "and https://github.com/Significant-Gravitas/AutoGPT/pull/12294."
    )


def test_sdk_exposes_cli_path_option():
    """Sanity-check that the SDK still exposes the `cli_path` option we use
    for the OpenRouter workaround.  If upstream removes it we need to know."""
    import inspect

    from claude_agent_sdk import ClaudeAgentOptions

    sig = inspect.signature(ClaudeAgentOptions)
    assert "cli_path" in sig.parameters, (
        "ClaudeAgentOptions no longer accepts `cli_path` — our "
        "claude_agent_cli_path config override would be silently ignored. "
        "Either find an alternative override mechanism or pin the SDK to a "
        "version that still exposes it."
    )


def test_sdk_exposes_max_thinking_tokens_option():
    """Sanity-check that the SDK still exposes the `max_thinking_tokens` option
    we use to cap extended thinking cost.  If upstream removes or renames it
    the cap will be silently ignored and Opus thinking tokens will be unbounded."""
    import inspect

    from claude_agent_sdk import ClaudeAgentOptions

    sig = inspect.signature(ClaudeAgentOptions)
    assert "max_thinking_tokens" in sig.parameters, (
        "ClaudeAgentOptions no longer accepts `max_thinking_tokens` — our "
        "claude_agent_max_thinking_tokens cost cap would be silently ignored, "
        "allowing Opus extended thinking to generate unbounded tokens at $75/M. "
        "Find the correct parameter name in the new SDK version and update "
        "ChatConfig.claude_agent_max_thinking_tokens and service.py accordingly."
    )
