"""SDK environment variable builder — importable without circular deps.

Extracted from ``service.py`` so that ``backend.blocks.orchestrator``
can reuse the same subscription / OpenRouter / direct-Anthropic logic
without pulling in the full copilot service module (which would create a
circular import through ``executor`` → ``credit`` → ``block_cost_config``).
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from backend.copilot.config import CLI_DEFAULT_CONTEXT_WINDOW, ChatConfig
from backend.copilot.moonshot import is_moonshot_model
from backend.copilot.sdk.context_window import (
    CodexEngineWindow,
    autocompact_pct,
    pinned_context_window,
)
from backend.copilot.sdk.subscription import validate_subscription

# ChatConfig is stateless (reads env vars) — a separate instance is fine.
# A singleton would require importing service.py which causes the circular dep
# this module was created to avoid.
config = ChatConfig()

# RFC 7230 §3.2.6 — keep only printable ASCII; strip control chars and non-ASCII.
_HEADER_SAFE_RE = re.compile(r"[^\x20-\x7e]")
_MAX_HEADER_VALUE_LEN = 128
_LOOPBACK_NO_PROXY_HOSTS = ("127.0.0.1", "localhost", "::1")


def _loopback_no_proxy_value() -> str:
    entries: list[str] = []
    for value in (os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")):
        for entry in value.split(","):
            entry = entry.strip()
            if entry and entry.lower() not in {item.lower() for item in entries}:
                entries.append(entry)
    for host in _LOOPBACK_NO_PROXY_HOSTS:
        if host.lower() not in {item.lower() for item in entries}:
            entries.append(host)
    return ",".join(entries)


def build_sdk_env(
    session_id: str | None = None,
    user_id: str | None = None,
    sdk_cwd: str | None = None,
    model: str | None = None,
    codex_gateway_url: str | None = None,
    codex_gateway_token: str | None = None,
    codex_engine: CodexEngineWindow | None = None,
) -> dict[str, str]:
    """Build env vars for the SDK CLI subprocess.

    *codex_engine* is what the connected account advertises for the routed
    model on the Codex route; the pin and trigger follow it when given
    (see ``sdk/context_window.py``).  Callers on other routes leave it None.

    Four modes (checked in order):
    1. **Codex gateway** — request-scoped loopback Anthropic compatibility.
    2. **Subscription** — clears all keys; CLI uses ``claude login`` auth.
    3. **Direct Anthropic** — subprocess inherits ``ANTHROPIC_API_KEY``
       from the parent environment (no overrides needed).
    4. **OpenRouter** (default) — overrides base URL and auth token to
       route through the proxy, with Langfuse trace headers.

    All modes receive workspace isolation (``CLAUDE_CODE_TMPDIR``) and
    security hardening env vars to prevent .claude.md loading, prompt
    history persistence, auto-memory writes, and non-essential traffic.

    *model* is the resolved SDK model slug (e.g. ``"moonshotai/kimi-k2.6"``
    or ``"anthropic/claude-sonnet-4-6"``).  Used to gate model-specific env
    vars (currently: ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`` is skipped for
    Moonshot since the cache-cost rationale doesn't apply there).
    """
    if (codex_gateway_url is None) != (codex_gateway_token is None):
        raise ValueError(
            "codex_gateway_url and codex_gateway_token must be provided together"
        )
    if codex_gateway_token is not None and not codex_gateway_token:
        raise ValueError("Codex gateway token must not be empty")
    if codex_gateway_url is not None:
        parsed_gateway = urlparse(codex_gateway_url)
        if parsed_gateway.scheme != "http" or parsed_gateway.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("Codex gateway must use a loopback HTTP URL")

    # A connected Codex account is a request-scoped auth transport.  It must
    # win over the deployment-wide profile, including ``local``: the loopback
    # gateway speaks the Anthropic wire protocol expected by Claude Code even
    # when the configured baseline provider does not.
    codex_route = codex_gateway_url is not None
    if codex_route and codex_gateway_token is not None:
        no_proxy = _loopback_no_proxy_value()
        env: dict[str, str] = {
            "ANTHROPIC_BASE_URL": codex_gateway_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": codex_gateway_token,
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CODE_REFRESH_TOKEN": "",
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }

    # Transports that don't run the SDK at all (currently: ``local`` —
    # Ollama et al. don't implement Anthropic's wire protocol) must not
    # reach this builder. The processor downgrades extended_thinking →
    # fast for those transports, so an entry here indicates a bug
    # upstream.  Fail loudly rather than constructing a doomed env.
    elif not config.transport.supports_sdk:
        raise RuntimeError(
            f"build_sdk_env() called under transport "
            f"{config.transport.name!r}, which doesn't support the SDK. "
            "The request should have been downgraded to the baseline "
            "path — see executor.processor.resolve_use_sdk."
        )

    # --- Mode 1: Claude Code subscription auth ---
    elif config.use_claude_code_subscription:
        validate_subscription()
        env = {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        }

    # --- Mode 2: Direct Anthropic (no proxy hop) ---
    elif not config.openrouter_active:
        # Clear OAuth tokens so CLI uses ANTHROPIC_API_KEY from parent env
        # rather than subscription auth if the container has those tokens set.
        env = {
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CODE_REFRESH_TOKEN": "",
        }

    # --- Mode 3: OpenRouter proxy ---
    else:
        base = (config.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        env = {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": config.api_key or "",
            "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
            "CLAUDE_CODE_OAUTH_TOKEN": "",  # prevent OAuth override of ANTHROPIC_AUTH_TOKEN
            "CLAUDE_CODE_REFRESH_TOKEN": "",  # prevent token refresh via subscription
        }

        # Inject broadcast headers so OpenRouter forwards traces to Langfuse.
        def _safe(v: str) -> str:
            return _HEADER_SAFE_RE.sub("", v).strip()[:_MAX_HEADER_VALUE_LEN]

        parts = []
        if session_id:
            parts.append(f"x-session-id: {_safe(session_id)}")
        if user_id:
            parts.append(f"x-user-id: {_safe(user_id)}")
        if parts:
            env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(parts)

    # --- Common: workspace isolation + security hardening (all modes) ---
    # Route subagent temp files into the per-session workspace so output
    # files are accessible (fixes /tmp/claude-0/ permission errors in E2B).
    if sdk_cwd:
        env["CLAUDE_CODE_TMPDIR"] = sdk_cwd

    # Harden multi-tenant deployment: prevent loading untrusted workspace
    # .claude.md files, writing auto-memory, and sending non-essential
    # telemetry traffic.
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    # Strip Anthropic-specific beta headers that OpenRouter rejects.
    # NOTE: this disables ALL experimental betas including
    # context-management-2025-06-27.  This is intentional: OpenRouter
    # compatibility takes priority, and Anthropic direct mode ignores this
    # flag harmlessly (those betas are not enabled there either by default).
    env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"

    # Pin the CLI's perceived context window explicitly.  This env var is the
    # highest-precedence input to the CLI's window resolution, so it preempts
    # the model table, the 1M capability flags and the server-side experiment
    # branches that newer CLIs consult — without it the compaction trigger is
    # an emergent property of whichever bundled CLI we happen to ship.
    # Each route is held to its coding engine's window (see
    # ``sdk/context_window.py``): what the account advertises on Codex, the
    # engine default elsewhere; the platform route keeps the 200K default.
    window = pinned_context_window(
        config, model, codex_route=codex_route, codex_engine=codex_engine
    )
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(window)

    # ...but that pin is clamped to the window the CLI *assumes* for the
    # model, so on a route whose slug the CLI does not recognise (the Codex
    # models) every value past that assumption is dropped in silence.
    # ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` is the knob that moves the
    # assumption — the CLI's own unknown-model notice points operators at it —
    # and without it the line above is inert.  Measured against CLI 2.1.274:
    # pins of 200K, 272K and 1M produce byte-identical compaction schedules
    # until this is set, and 1M only takes effect once it is.
    #
    # Codex-only on purpose.  Everywhere else the CLI's model table is the
    # better authority, and raising this would push its client-side length
    # guard past what the provider actually accepts — trading a clean local
    # refusal for a provider 400 mid-turn.
    if codex_route:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)

    # The window above is clamped by the model's own window, which this
    # kill-switch holds at 200K; keeping it set past that point would swallow
    # the raise silently.  1M is GA (no beta header) on Sonnet 4.6+/5, so the
    # experimental-betas flag above does not cover it.  On the Codex route
    # the flag is inert (a GPT slug has no 1M gate) but harmless, so the one
    # rule — set iff the pin is at or below the CLI default — stands.
    if window <= CLI_DEFAULT_CONTEXT_WINDOW:
        env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"

    # Trigger threshold, as a percentage of the window pinned above (CLI
    # default: ~93%).  The override caps Anthropic cache-creation cost;
    # Moonshot routes skip it because their OpenRouter endpoint returns
    # ``cache_create=0`` (no cache writes happen, so there's no cost to cap)
    # and an aggressive trigger cascades into 3+ compactions per turn.
    # The Codex route instead mirrors the engine's own 90% trigger (see
    # ``sdk/context_window.py``) — including for a moonshot-shaped slug,
    # which still runs on Codex infra there, not the Moonshot endpoint.
    # Operators can also set the config to 0 to disable globally.
    if codex_route or not is_moonshot_model(model):
        pct = autocompact_pct(
            config, model, codex_route=codex_route, codex_engine=codex_engine
        )
        if pct > 0:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(pct)

    # Disable gzip on API responses to prevent ZlibError decompression
    # failures (see oven-sh/bun#23149, anthropics/claude-code#18302).
    # Appended to any existing ANTHROPIC_CUSTOM_HEADERS (OpenRouter mode
    # already sets trace headers above).
    accept_encoding = "Accept-Encoding: identity"
    existing = env.get("ANTHROPIC_CUSTOM_HEADERS", "")
    env["ANTHROPIC_CUSTOM_HEADERS"] = (
        f"{existing}\n{accept_encoding}" if existing else accept_encoding
    )

    return env


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
