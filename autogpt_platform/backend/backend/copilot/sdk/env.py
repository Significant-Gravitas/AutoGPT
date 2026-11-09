"""The CLI's environment for a launch, from the route profile it runs on.

Kept out of ``service.py`` so that ``backend.blocks.orchestrator`` can launch
through the same function without pulling in the full copilot service module
(which would create a circular import through ``executor`` → ``credit`` →
``block_cost_config``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from backend.copilot.config import ChatConfig
from backend.copilot.llm_attribution import openrouter_attribution
from backend.copilot.route_profile import RouteProfile
from backend.copilot.route_table import is_openrouter_url
from backend.copilot.sdk.route_credentials import (
    CLAUDE_OAUTH_BLANK,
    LOOPBACK_HOSTS,
    direct_anthropic_env,
    require_local_oauth_base_url,
    require_loopback_http,
    subscription_env,
)
from backend.copilot.sdk.subscription import validate_subscription

# ChatConfig is stateless (reads env vars) — a separate instance is fine.
# A singleton would require importing service.py which causes the circular dep
# this module was created to avoid.
config = ChatConfig()

# The window the CLI assumes for any Claude model once 1M is gated off.
CLI_DEFAULT_CONTEXT_WINDOW = 200_000

# Both launch paths check the CLI binary themselves (``cli_version``), from the
# allowlisted environment. The SDK's own check takes no environment, so before
# each launch it would start the CLI with the backend's whole one; an empty
# value would leave it on.
os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"

# RFC 7230 §3.2.6 — keep only printable ASCII; strip control chars and non-ASCII.
_HEADER_SAFE_RE = re.compile(r"[^\x20-\x7e]")
_MAX_HEADER_VALUE_LEN = 128


class GatewayEndpoint(BaseModel):
    """The loopback gateway a launch talks to instead of the provider."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    auth_token: str


def common_launch_options() -> dict[str, Any]:
    """``ClaudeAgentOptions`` every launch takes: the chat's and the
    orchestrator block's alike.

    Never load the host machine's ~/.claude settings (hooks, skills,
    plugins) or filesystem MCP servers into the subprocess: on a developer
    machine a personal Claude Code setup can inject hundreds of MCP tool
    schemas, inflating the static prompt past the autocompact threshold.
    ``max_turns`` caps agentic tool-use loops per query so a runaway one
    cannot burn the budget; ``cli_path`` pins the CLI binary when set.
    """
    options: dict[str, Any] = {
        "setting_sources": [],
        "verbatim_prompts": True,
        "extra_args": {"strict-mcp-config": None, DISABLE_SLASH_COMMANDS_FLAG: None},
        "max_buffer_size": config.claude_agent_max_buffer_size,
        "max_turns": config.agent_max_turns,
    }
    if config.claude_agent_cli_path:
        options["cli_path"] = config.claude_agent_cli_path
    return options


def route_launch_env(
    profile: RouteProfile,
    *,
    sdk_cwd: str | None = None,
    gateway: GatewayEndpoint | None = None,
    stand_in_model_url: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, str]:
    """The CLI's environment for one launch on *profile*.

    The chat and the orchestrator block both launch through here, so a launch
    on the same route gets the same environment; only the chat passes the
    session and user, which OpenRouter forwards to tracing and to Broadcast's
    attribution.

    Every launch states its credential and base URL explicitly (see
    ``sdk/route_credentials.py``), gets the same workspace isolation and
    hardening, and is pinned to the profile's window, trigger and timeout.
    """
    if stand_in_model_url is not None:
        if gateway is not None:
            raise ValueError("A turn has one model endpoint")
        base, host = stand_in_base_url(stand_in_model_url)
        no_proxy = _no_proxy_value(host)
        env = {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": "load-test",
            "ANTHROPIC_API_KEY": "",
            **CLAUDE_OAUTH_BLANK,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
    else:
        env = _credential_env(profile, gateway)
    if profile.name == "openrouter" and stand_in_model_url is None:
        headers = _trace_headers(session_id, user_id)
        if headers:
            env["ANTHROPIC_CUSTOM_HEADERS"] = headers
        if (session_id or user_id) and is_openrouter_url(profile.base_url):
            # The chat's launches only: the fields OpenRouter Broadcast
            # attributes a generation by, merged into every request body.
            # Another server on this route rejects body fields it does not
            # know, so it gets none.
            env["CLAUDE_CODE_EXTRA_BODY"] = _broadcast_extra_body(session_id, user_id)
    if not profile.row.attribution_header:
        # The CLI's billing-header system block carries per-session values;
        # without it the system prefix is the same in every session.
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"

    # Route subagent temp files into the per-session workspace so output
    # files are accessible (fixes /tmp/claude-0/ permission errors in E2B).
    if sdk_cwd:
        from backend.copilot.host_fs import cli_tmpdir

        env["CLAUDE_CODE_TMPDIR"] = cli_tmpdir(sdk_cwd)
    env.update(_HARDENING)
    env["API_TIMEOUT_MS"] = str(profile.row.api_timeout_ms)
    env.update(_window_env(profile))

    # Disable gzip on API responses to prevent ZlibError decompression
    # failures (see oven-sh/bun#23149, anthropics/claude-code#18302).
    accept_encoding = "Accept-Encoding: identity"
    existing = env.get("ANTHROPIC_CUSTOM_HEADERS", "")
    env["ANTHROPIC_CUSTOM_HEADERS"] = (
        f"{existing}\n{accept_encoding}" if existing else accept_encoding
    )

    if profile.name == "subscription":
        require_local_oauth_base_url(env)
    return env


# Harden multi-tenant deployment: no untrusted workspace .claude.md files, no
# auto-memory writes, no non-essential telemetry traffic.
_HARDENING = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    # Keep TodoWrite, the checklist the UI renders; the CLI otherwise swaps in
    # its TaskCreate/TaskUpdate tools.
    "CLAUDE_CODE_ENABLE_TASKS": "0",
    # Run sub-agents in the foreground.  Since Claude Code 2.1.198 the
    # Agent/Task tool launches them in the background by default and returns
    # ``async_launched`` at once; their report only arrives as a new CLI turn
    # after the ``ResultMessage``, by which point the turn has ended and the
    # CLI has been closed, killing the sub-agent and losing its work.  This
    # also stops the CLI from moving long tool calls to the background.
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    # Strip Anthropic-specific beta headers that OpenRouter rejects.  This
    # disables ALL experimental betas on every route: against Anthropic itself
    # (direct and subscription) it also drops the thinking-token-count,
    # context-management-2025-06-27 and prompt-caching-scope beta headers and
    # the ``context_management`` body field the CLI would otherwise send.
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
}


def _credential_env(
    profile: RouteProfile, gateway: GatewayEndpoint | None
) -> dict[str, str]:
    """The credential and base URL, which no launch inherits."""
    if profile.gateway_adapter is not None:
        if gateway is None:
            raise RuntimeError(f"The {profile.name} route launches behind a gateway")
        if not gateway.auth_token:
            raise ValueError("A gateway capability must not be empty")
        require_loopback_http(gateway.base_url, "The model gateway")
        no_proxy = _no_proxy_value(*LOOPBACK_HOSTS)
        return {
            "ANTHROPIC_BASE_URL": gateway.base_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": gateway.auth_token,
            "ANTHROPIC_API_KEY": "",
            **CLAUDE_OAUTH_BLANK,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
    if profile.name == "subscription":
        validate_subscription(config.claude_agent_cli_path)
        return subscription_env()
    key = profile.api_key.get_secret_value() if profile.api_key else ""
    if profile.name == "direct_anthropic":
        return {**direct_anthropic_env(key), "ANTHROPIC_BASE_URL": profile.base_url}
    if not key:
        raise RuntimeError(
            "No API key configured: the openrouter route needs "
            "CHAT_OPENROUTER_API_KEY."
        )
    return {
        "ANTHROPIC_BASE_URL": profile.base_url,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
        **CLAUDE_OAUTH_BLANK,
    }


def _window_env(profile: RouteProfile) -> dict[str, str]:
    """Pin the CLI's window and trigger to the profile's.

    ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` is the highest-precedence input to the
    CLI's window resolution, so the compaction trigger is a deliberate value
    rather than whatever the bundled CLI's model table and experiments say.
    That pin is clamped to the window the CLI assumes for the model, so on a
    route whose model names it does not know (Codex) the assumption is raised
    too; elsewhere the CLI's model table is the better authority, and raising
    it would push its length guard past what the provider accepts.
    """
    env = {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(profile.window)}
    if profile.pins_max_context:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(profile.window)
    # The model's own window is held at 200K by this switch; keeping it past
    # that point would swallow a larger pin silently.
    if profile.window <= CLI_DEFAULT_CONTEXT_WINDOW:
        env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
    if profile.trigger_pct:
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(profile.trigger_pct)
    return env


def _trace_headers(session_id: str | None, user_id: str | None) -> str:
    """The session and user OpenRouter forwards to Langfuse."""

    def _safe(v: str) -> str:
        return _HEADER_SAFE_RE.sub("", v).strip()[:_MAX_HEADER_VALUE_LEN]

    parts = []
    if session_id:
        parts.append(f"x-session-id: {_safe(session_id)}")
    if user_id:
        parts.append(f"x-user-id: {_safe(user_id)}")
    return "\n".join(parts)


# A direct-Anthropic launch's credential and where it goes, carried over from
# the backend's environment: what the CLI used to inherit along with the rest.
_DIRECT_ANTHROPIC_NAMES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)

# CLI flag for ``ClaudeAgentOptions.extra_args``. Every prompt we send is user
# or graph text, never a CLI slash command.
DISABLE_SLASH_COMMANDS_FLAG = "disable-slash-commands"


def without_url_credentials(url: str) -> str:
    """*url* without a user name and password some deployments embed in it."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))


def cli_prompt_text(text: str) -> str:
    """*text* as the CLI must receive it to send it on to the model.

    Even with the flag above, the CLI answers a prompt that starts with "/"
    itself when it names a built-in command. It only looks at the first
    character, so a leading space keeps the text whole.
    """
    return f" {text}" if text.startswith("/") else text


def cli_user_message(content: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    """A stream-json user message the CLI hands to the model as written.

    ``ClaudeAgentOptions.verbatim_prompts`` marks every message the SDK writes
    ``client_composed``; a message written straight to the transport must
    carry the mark itself.
    """
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
        "session_id": session_id,
        "client_composed": True,
    }


def stand_in_base_url(url: str) -> tuple[str, str]:
    """A load-test stand-in model's base URL, without ``/v1``, and its host.

    The turn's profile and the CLI's environment both take the base from here,
    so the two always name the same endpoint.
    """
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The stand-in model URL must be an http(s) URL")
    return base, parsed.hostname


def _no_proxy_value(*hosts: str) -> str:
    """The backend's NO_PROXY with *hosts* added, for a model endpoint the
    CLI must reach directly rather than through a configured proxy."""
    entries: list[str] = []
    for value in (os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")):
        for entry in value.split(","):
            entry = entry.strip()
            if entry and entry.lower() not in {item.lower() for item in entries}:
                entries.append(entry)
    for host in hosts:
        if host.lower() not in {item.lower() for item in entries}:
            entries.append(host)
    return ",".join(entries)


def describe_sdk_context(
    *,
    route: str,
    model: str | None,
    sdk_env: dict[str, str],
    window_source: str | None = None,
) -> str:
    """One-line summary of the context the SDK subprocess was pinned to.

    Read back off the built env (rather than recomputed) so the line
    always states what the subprocess actually received. Never includes
    secret values — window, trigger, and flags only.
    """
    window = sdk_env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "<unset>")
    pct = sdk_env.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "<cli-default>")
    kill = "true" if "CLAUDE_CODE_DISABLE_1M_CONTEXT" in sdk_env else "false"
    source = f" window_source={window_source}" if window_source else ""
    return (
        f"route={route} model={model} window={window}{source} "
        f"trigger_pct={pct} disable_1m_context={kill}"
    )


def _broadcast_extra_body(session_id: str | None, user_id: str | None) -> str:
    """``CLAUDE_CODE_EXTRA_BODY``: JSON the CLI merges into every request body
    (main turn, sub-agents, compaction), carrying the fields OpenRouter
    Broadcast attributes a generation by.

    ``metadata.user_id`` replaces the CLI's own (a JSON blob of its device and
    session ids) so that every user field on the request names the same
    platform user.
    """
    body = openrouter_attribution(
        user_id=user_id, session_id=session_id, generation_name="copilot-sdk"
    )
    if user_id:
        body["metadata"] = {"user_id": body["user"]}
    return json.dumps(body)
