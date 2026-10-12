"""SDK environment variable builder — importable without circular deps.

Kept out of ``service.py`` so that ``backend.blocks.orchestrator`` can take
its launch credentials from here (``build_user_key_sdk_env``) without
pulling in the full copilot service module (which would create a circular
import through ``executor`` → ``credit`` → ``block_cost_config``).
"""

from __future__ import annotations

import os
import re

from backend.copilot.config import CLI_DEFAULT_CONTEXT_WINDOW, ChatConfig
from backend.copilot.moonshot import is_moonshot_model
from backend.copilot.sdk.context_window import (
    CodexEngineWindow,
    autocompact_pct,
    pinned_context_window,
)
from backend.copilot.sdk.route_credentials import (
    CLAUDE_OAUTH_BLANK,
    LOOPBACK_HOSTS,
    attribution_header_env,
    direct_anthropic_env,
    openrouter_user_key_env,
    require_local_oauth_base_url,
    require_loopback_http,
    subscription_env,
)
from backend.copilot.sdk.subscription import validate_subscription

# ChatConfig is stateless (reads env vars) — a separate instance is fine.
# A singleton would require importing service.py which causes the circular dep
# this module was created to avoid.
config = ChatConfig()

# RFC 7230 §3.2.6 — keep only printable ASCII; strip control chars and non-ASCII.
_HEADER_SAFE_RE = re.compile(r"[^\x20-\x7e]")
_MAX_HEADER_VALUE_LEN = 128


def _loopback_no_proxy_value() -> str:
    entries: list[str] = []
    for value in (os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")):
        for entry in value.split(","):
            entry = entry.strip()
            if entry and entry.lower() not in {item.lower() for item in entries}:
                entries.append(entry)
    for host in LOOPBACK_HOSTS:
        if host.lower() not in {item.lower() for item in entries}:
            entries.append(host)
    return ",".join(entries)


def build_user_key_sdk_env(provider: str, api_key: str) -> dict[str, str]:
    """Credentials for a CLI launch on a user's own API key.

    The orchestrator block's launch: ``open_router`` goes through the
    deployment's OpenRouter base URL (per-credential base URLs are not
    supported), any other provider is a direct Anthropic key.  No Claude
    OAuth variable or parent base URL survives.
    """
    if provider != "open_router":
        return direct_anthropic_env(api_key)
    base = (config.base_url or "https://openrouter.ai/api").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return {
        **openrouter_user_key_env(base, api_key),
        **attribution_header_env(config.claude_agent_omit_attribution_header),
    }


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
       Self-hosted only; its base URL may only be empty or loopback.
    3. **Direct Anthropic** — ``direct_anthropic_api_key`` passed
       explicitly, against api.anthropic.com.
    4. **OpenRouter** (default) — overrides base URL and auth token to
       route through the proxy, with Langfuse trace headers.

    Every mode but subscription blanks the Claude OAuth variables and sets
    its own base URL and a non-empty key (``sdk/route_credentials.py``).

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
        require_loopback_http(codex_gateway_url, "Codex gateway")

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
            **CLAUDE_OAUTH_BLANK,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
            **attribution_header_env(config.claude_agent_omit_attribution_header),
        }

    # Transports that don't run the SDK at all (currently: ``local`` —
    # Ollama et al. don't implement Anthropic's wire protocol) must not
    # reach this builder: the engine choice (``engine.resolve_use_sdk``)
    # sends their turns to the baseline engine, so an entry here indicates
    # a bug upstream.  Fail loudly rather than constructing a doomed env.
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
        env = subscription_env()

    # --- Mode 2: Direct Anthropic (no proxy hop) ---
    elif not config.openrouter_active:
        env = direct_anthropic_env(config.direct_anthropic_api_key)

    # --- Mode 3: OpenRouter proxy ---
    else:
        base = (config.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        env = {
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_AUTH_TOKEN": config.api_key or "",
            "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
            **CLAUDE_OAUTH_BLANK,
            **attribution_header_env(config.claude_agent_omit_attribution_header),
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

        # The main conversation's cache TTL; sub-agents and other requests
        # outside it keep the CLI's 5 minutes. See
        # ``claude_agent_prompt_cache_ttl``.
        env["CLAUDE_CODE_PROMPT_CACHE_TTL"] = config.claude_agent_prompt_cache_ttl

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
    # Keep TodoWrite, the checklist the UI renders; the CLI otherwise swaps in
    # its TaskCreate/TaskUpdate tools.
    env["CLAUDE_CODE_ENABLE_TASKS"] = "0"
    # Run sub-agents in the foreground.  Since Claude Code 2.1.198 the
    # Agent/Task tool launches them in the background by default and returns
    # ``async_launched`` at once; their report only arrives as a new CLI turn
    # after the ``ResultMessage``, by which point the turn has ended and the
    # CLI has been closed, killing the sub-agent and losing its work.  This
    # also stops the CLI from moving long tool calls to the background.
    env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
    # Strip Anthropic-specific beta headers that OpenRouter rejects.
    # NOTE: this disables ALL experimental betas, on every mode.  Against
    # Anthropic itself (direct and subscription) it also drops the
    # thinking-token-count, context-management-2025-06-27 and
    # prompt-caching-scope beta headers and the ``context_management`` body
    # field the CLI would otherwise send there.
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

    if config.use_claude_code_subscription:
        require_local_oauth_base_url(env)

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
