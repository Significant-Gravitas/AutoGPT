"""The chat settings an operator sets, checked before any service starts.

AutoPilot's chat is configured by a handful of explicit names: the platform
route (``CHAT_PLATFORM_ROUTE``), that route's key and URL, the two model tiers
and, for a local server, its adapter and real context window. Everything else
about a route is code (``route_table.py``).

Settings parsing ignores names it does not know, so a deployment still setting
a name this format replaced would start normally and quietly lose that setting.
:func:`check_chat_environment` reads the process environment and the dotenv
files the backend loads, before any settings object is built, and fails with one
list naming every replaced setting and what replaces it, plus anything the
selected route still needs. It reports names, never values.

Import-light on purpose: the single-container image runs it through
``runtime_config.py check-chat-env`` before any service starts.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from dotenv import dotenv_values

PLATFORM_ROUTES = (
    "openrouter",
    "direct_anthropic",
    "local",
    "subscription",
    "disabled",
)
LOCAL_ADAPTERS = ("openai_chat", "anthropic_passthrough")
# AutoPilot's prompt and tools beside a working conversation need this much.
MIN_LOCAL_CONTEXT_TOKENS = 65_536

# Every chat setting the old format accepted that this one does not, and what
# replaced it. A deployment that sets any of them, even to an empty value, does
# not start.
RETIRED_CHAT_SETTINGS: dict[str, str] = {
    "CHAT_FAST_STANDARD_MODEL": "replaced by CHAT_STANDARD_MODEL",
    "CHAT_FAST_MODEL": "replaced by CHAT_STANDARD_MODEL",
    "CHAT_FAST_ADVANCED_MODEL": "replaced by CHAT_ADVANCED_MODEL",
    "CHAT_THINKING_STANDARD_MODEL": "replaced by CHAT_STANDARD_MODEL",
    "CHAT_MODEL": "replaced by CHAT_STANDARD_MODEL",
    "CHAT_THINKING_ADVANCED_MODEL": "replaced by CHAT_ADVANCED_MODEL",
    "CHAT_CLAUDE_AGENT_MODEL": (
        "replaced by CHAT_STANDARD_MODEL and CHAT_ADVANCED_MODEL; it set the "
        "model of both tiers, so set each tier you meant"
    ),
    "CHAT_TITLE_MODEL": (
        "removed: the platform route picks the title model (the standard tier "
        "on local)"
    ),
    "CHAT_SIMULATION_MODEL": "removed: the platform route picks the dry-run model",
    "CHAT_GATE_MODEL": "removed: the action check runs on a model the platform vetted",
    "CHAT_GATE_CONTENT_MODEL": (
        "removed: the content check runs on a model the platform vetted"
    ),
    "CHAT_API_KEY": (
        "replaced by the selected route's key: CHAT_OPENROUTER_API_KEY or "
        "CHAT_LOCAL_API_KEY"
    ),
    "CHAT_BASE_URL": (
        "replaced by the selected route's URL: CHAT_OPENROUTER_BASE_URL "
        "(https://openrouter.ai/api, without /v1), CHAT_ANTHROPIC_BASE_URL or "
        "CHAT_LOCAL_BASE_URL"
    ),
    "CHAT_AUX_API_KEY": "removed: titles and other side calls use the platform route",
    "CHAT_AUX_BASE_URL": "removed: titles and other side calls use the platform route",
    "CHAT_DIRECT_ANTHROPIC_API_KEY": "replaced by CHAT_ANTHROPIC_API_KEY",
    "CHAT_USE_CLAUDE_AGENT_SDK": "removed: the platform route decides the engine",
    "CHAT_CLAUDE_AGENT_USE_RESUME": "removed: chats always resume their CLI session",
    "CHAT_CLAUDE_AGENT_FALLBACK_MODEL": "removed: there is no fallback model",
    "CHAT_CLAUDE_AGENT_MAX_TURNS": "replaced by CHAT_AGENT_MAX_TURNS",
    "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW": (
        "removed: each route sets its own context window; a local server's is "
        "CHAT_LOCAL_CONTEXT_TOKENS"
    ),
    "CHAT_CLAUDE_AGENT_AUTOCOMPACT_PCT_OVERRIDE": (
        "removed: each route sets its own compaction trigger"
    ),
    "CHAT_CLAUDE_AGENT_MAX_THINKING_TOKENS": (
        "removed: each route sets its own thinking budget"
    ),
    "CHAT_CLAUDE_AGENT_THINKING_EFFORT": (
        "removed: each route sets its own thinking effort"
    ),
    "CHAT_CLAUDE_AGENT_CROSS_USER_PROMPT_CACHE": (
        "removed: AutoPilot's prompt always replaces the CLI's"
    ),
    "CHAT_BASELINE_PROMPT_CACHE_TTL": (
        "removed: the platform sets the prompt-cache lifetime"
    ),
    "CHAT_SDK_RECONCILE_OPENROUTER_COST": (
        "removed: an OpenRouter turn is always charged OpenRouter's reported cost"
    ),
    "CHAT_USE_OPENROUTER": (
        "replaced by CHAT_PLATFORM_ROUTE=openrouter or "
        "CHAT_PLATFORM_ROUTE=direct_anthropic"
    ),
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION": "replaced by CHAT_PLATFORM_ROUTE=subscription",
    "CHAT_CLAUDE_AGENT_OMIT_ATTRIBUTION_HEADER": (
        "removed: each route decides whether the CLI sends its attribution header"
    ),
    "CHAT_LOCAL_REQUEST_TIMEOUT_S": "removed: a local request may take 30 minutes",
    "CHAT_USE_LOCAL": (
        "replaced by CHAT_PLATFORM_ROUTE=local, with CHAT_LOCAL_BASE_URL, "
        "CHAT_LOCAL_ADAPTER, CHAT_LOCAL_CONTEXT_TOKENS, CHAT_STANDARD_MODEL and "
        "CHAT_ADVANCED_MODEL"
    ),
    "CHAT_PROMPT_PATHS": "removed: nothing reads it",
}

# Every ``CHAT_*`` setting ``ChatConfig`` reads (``chat_env_test.py`` holds the
# two lists together), and the ones read elsewhere: ``CHAT_OPENAI_API_KEY`` is
# the memory layer's embedding key.
ACCEPTED_CHAT_SETTINGS: frozenset[str] = frozenset(
    {
        "CHAT_PLATFORM_ROUTE",
        "CHAT_OPENROUTER_API_KEY",
        "CHAT_OPENROUTER_BASE_URL",
        "CHAT_ANTHROPIC_API_KEY",
        "CHAT_ANTHROPIC_BASE_URL",
        "CHAT_LOCAL_BASE_URL",
        "CHAT_LOCAL_API_KEY",
        "CHAT_LOCAL_ADAPTER",
        "CHAT_LOCAL_CONTEXT_TOKENS",
        "CHAT_STANDARD_MODEL",
        "CHAT_ADVANCED_MODEL",
        "CHAT_LEARNING_REVIEWER_MODEL",
        "CHAT_GATE_TIMEOUT_S",
        "CHAT_GATE_FIRST_STAGE",
        "CHAT_GATE_JEV_MODEL",
        "CHAT_GATE_JEV_ASK_THRESHOLD",
        "CHAT_GATE_JEV_TIMEOUT_S",
        "CHAT_CONTENT_JUDGE_TIMEOUT_S",
        "CHAT_SESSION_TTL",
        "CHAT_MAX_AGENT_RUNS",
        "CHAT_MAX_AGENT_SCHEDULES",
        "CHAT_STREAM_TTL",
        "CHAT_STREAM_LOCK_TTL",
        "CHAT_SESSION_META_PREFIX",
        "CHAT_TURN_STREAM_PREFIX",
        "CHAT_LANGFUSE_PROMPT_NAME",
        "CHAT_LANGFUSE_PROMPT_CACHE_TTL",
        "CHAT_DAILY_COST_LIMIT_MICRODOLLARS",
        "CHAT_WEEKLY_COST_LIMIT_MICRODOLLARS",
        "CHAT_RATE_LIMIT_RESET_COST",
        "CHAT_MAX_DAILY_RESETS",
        "CHAT_CLAUDE_AGENT_MAX_BUFFER_SIZE",
        "CHAT_CLAUDE_AGENT_MAX_SUBTASKS",
        "CHAT_CLAUDE_AGENT_EXTRA_BUILTIN_TOOLS",
        "CHAT_AGENT_MAX_TURNS",
        "CHAT_SDK_STEP_CHECKIN",
        "CHAT_CLAUDE_AGENT_MAX_BUDGET_USD",
        "CHAT_TREE_CEILING_FRACTION_OF_DAILY",
        "CHAT_TREE_CEILING_FLOOR_MICRODOLLARS",
        "CHAT_TREE_CEILING_MICRODOLLARS",
        "CHAT_SPEND_CEILING_RESET",
        "CHAT_TREE_MAX_NODES",
        "CHAT_TREE_BUDGET_SIGNAL_ENABLED",
        "CHAT_TREE_WRAPUP_THRESHOLD",
        "CHAT_STREAM_REPLAY_COUNT",
        "CHAT_CLAUDE_AGENT_MAX_TRANSIENT_RETRIES",
        "CHAT_CLAUDE_AGENT_LIVE_SYSTEM_PROMPT",
        "CHAT_CLAUDE_AGENT_STRIP_PROMPT_SNAPSHOTS",
        "CHAT_SDK_INCLUDE_PARTIAL_MESSAGES",
        "CHAT_CLAUDE_AGENT_CLI_PATH",
        "CHAT_TEST_MODE",
        "CHAT_USE_E2B_SANDBOX",
        "CHAT_E2B_API_KEY",
        "CHAT_E2B_SANDBOX_TEMPLATE",
        "CHAT_E2B_SANDBOX_TIMEOUT",
        "CHAT_E2B_SANDBOX_ON_TIMEOUT",
        "CHAT_VOICE_TTS_MODEL",
        "CHAT_VOICE_TTS_VOICE",
        "CHAT_VOICE_TTS_INSTRUCTIONS",
        "CHAT_VOICE_TTS_SPEED",
        "CHAT_VOICE_TTS_USD_PER_1K_CHARS",
    }
)
CHAT_SETTINGS_READ_ELSEWHERE: frozenset[str] = frozenset({"CHAT_OPENAI_API_KEY"})

# The settings each route reads, so one set for a route nobody selected is named.
# CHAT_ANTHROPIC_API_KEY is left out: it is also the Batch API's key for the
# platform's background jobs on any route, so it is never stray.
_ROUTE_SETTINGS: dict[str, tuple[str, ...]] = {
    "openrouter": ("CHAT_OPENROUTER_API_KEY", "CHAT_OPENROUTER_BASE_URL"),
    "direct_anthropic": ("CHAT_ANTHROPIC_BASE_URL",),
    "local": (
        "CHAT_LOCAL_BASE_URL",
        "CHAT_LOCAL_API_KEY",
        "CHAT_LOCAL_ADAPTER",
        "CHAT_LOCAL_CONTEXT_TOKENS",
    ),
}
# Non-chat names the check reads: generic endpoint overrides that used to move
# the chat, the integration keys the chat used to fall back to, and the
# deployment mode.
_INTEGRATION_KEYS = ("OPEN_ROUTER_API_KEY", "ANTHROPIC_API_KEY")
_OTHER_SETTINGS = (
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "BEHAVE_AS",
    *_INTEGRATION_KEYS,
)
_TRUE = ("1", "true", "yes", "on")


class ChatEnvironmentError(ValueError):
    """The chat settings cannot start a service; ``problems`` names each one."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            "AutoPilot's chat settings need attention before the backend can start "
            "(see docs/platform/copilot-local-llm.md and "
            "docs/platform/single-container.md):\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def check_chat_environment(
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Iterable[Path] | None = None,
) -> list[str]:
    """Raise :class:`ChatEnvironmentError` listing every problem; return the
    warnings, for the caller to log."""
    problems, warnings = chat_environment_problems(
        os.environ if environ is None else environ,
        backend_dotenv_paths() if dotenv_paths is None else dotenv_paths,
    )
    if problems:
        raise ChatEnvironmentError(problems)
    return warnings


def parse_platform_route(value: object) -> str | None:
    """The platform route *value* names, ignoring surrounding whitespace,
    else None. The startup check and ``ChatConfig`` both read the selector
    through this, so the route the check accepts is the route that runs."""
    return _choice(value, PLATFORM_ROUTES)


def parse_local_adapter(value: object) -> str | None:
    """The local adapter *value* names, read like the route."""
    return _choice(value, LOCAL_ADAPTERS)


def _choice(value: object, choices: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text in choices else None


def chat_environment_problems(
    environ: Mapping[str, str],
    dotenv_paths: Iterable[Path] = (),
) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` for these settings. Names only, never values."""
    settings = read_chat_settings(environ, list(dotenv_paths))
    problems = [
        f"{name} is no longer read: {RETIRED_CHAT_SETTINGS[name]}"
        for name in sorted(settings)
        if name in RETIRED_CHAT_SETTINGS
    ]
    warnings = [
        f"{name} is not a setting anything reads"
        for name in sorted(settings)
        if name.startswith("CHAT_")
        and name not in RETIRED_CHAT_SETTINGS
        and name not in ACCEPTED_CHAT_SETTINGS
        and name not in CHAT_SETTINGS_READ_ELSEWHERE
    ]

    selector = settings.get("CHAT_PLATFORM_ROUTE", "")
    if not selector.strip():
        problems.append(
            "CHAT_PLATFORM_ROUTE is not set: choose "
            + ", ".join(PLATFORM_ROUTES)
            + " (disabled runs without the platform's own chat model)"
        )
        return problems, warnings
    route = parse_platform_route(selector)
    if route is None:
        problems.append(
            "CHAT_PLATFORM_ROUTE is not a route: choose " + ", ".join(PLATFORM_ROUTES)
        )
        return problems, warnings

    warnings.extend(_unused_route_settings(route, settings))
    if route == "disabled":
        keys = [name for name in _INTEGRATION_KEYS if settings.get(name, "").strip()]
        if keys:
            warnings.append(
                f"{' and '.join(keys)} {'is' if len(keys) == 1 else 'are'} set, but "
                "CHAT_PLATFORM_ROUTE is disabled: the platform runs no chat model of "
                "its own, and integration keys never serve the chat (choose a route "
                "and give it its own key, e.g. CHAT_PLATFORM_ROUTE=openrouter with "
                "CHAT_OPENROUTER_API_KEY)"
            )
    if settings.get("CHAT_TEST_MODE", "").strip().lower() in _TRUE:
        # The dummy engine answers every turn; no route credential is needed.
        return problems, warnings
    problems.extend(_route_problems(route, settings))
    return problems, warnings


def _route_problems(route: str, settings: Mapping[str, str]) -> list[str]:
    """What the selected route still needs."""

    def needs(*names: str) -> list[str]:
        return [
            f"CHAT_PLATFORM_ROUTE={route} needs {name}"
            for name in names
            if not settings.get(name, "").strip()
        ]

    if route == "openrouter":
        problems = needs("CHAT_OPENROUTER_API_KEY")
        moved = [
            n for n in ("OPENROUTER_BASE_URL", "OPENAI_BASE_URL") if settings.get(n)
        ]
        if moved and not settings.get("CHAT_OPENROUTER_BASE_URL", "").strip():
            problems.append(
                f"{' and '.join(moved)} no longer point the chat anywhere: set "
                "CHAT_OPENROUTER_BASE_URL to the URL the chat should use "
                "(https://openrouter.ai/api for OpenRouter itself)"
            )
        return problems
    if route == "direct_anthropic":
        return needs("CHAT_ANTHROPIC_API_KEY") + _anthropic_tiers(route, settings)
    if route == "subscription":
        problems = _anthropic_tiers(route, settings)
        if settings.get("BEHAVE_AS", "").strip().lower() == "cloud":
            problems.append(
                "CHAT_PLATFORM_ROUTE=subscription is for a self-hosted install; it "
                "cannot run with BEHAVE_AS=cloud"
            )
        return problems
    if route == "local":
        return needs(
            "CHAT_LOCAL_BASE_URL", "CHAT_STANDARD_MODEL", "CHAT_ADVANCED_MODEL"
        ) + _local_shape(settings)
    return []


def _local_shape(settings: Mapping[str, str]) -> list[str]:
    problems: list[str] = []
    adapter = settings.get("CHAT_LOCAL_ADAPTER", "")
    if not adapter.strip():
        problems.append("CHAT_PLATFORM_ROUTE=local needs CHAT_LOCAL_ADAPTER")
    elif parse_local_adapter(adapter) is None:
        problems.append(
            "CHAT_LOCAL_ADAPTER is not an adapter: choose "
            + " or ".join(LOCAL_ADAPTERS)
        )
    window = settings.get("CHAT_LOCAL_CONTEXT_TOKENS", "").strip()
    if not window:
        problems.append(
            "CHAT_PLATFORM_ROUTE=local needs CHAT_LOCAL_CONTEXT_TOKENS, the model's "
            "context window as the server has it loaded"
        )
    elif not window.isdigit():
        problems.append(
            "CHAT_LOCAL_CONTEXT_TOKENS must be the model's context window in "
            "tokens, a whole number"
        )
    elif int(window) < MIN_LOCAL_CONTEXT_TOKENS:
        problems.append(
            f"CHAT_LOCAL_CONTEXT_TOKENS is below {MIN_LOCAL_CONTEXT_TOKENS}: "
            "AutoPilot's prompt and tools leave no room to work; load the model "
            "with a larger context window"
        )
    return problems


def _anthropic_tiers(route: str, settings: Mapping[str, str]) -> list[str]:
    """The Anthropic routes serve Anthropic models only."""
    return [
        f"{name} names a model the {route} route cannot serve: use an "
        "anthropic/* or claude-* model"
        for name in ("CHAT_STANDARD_MODEL", "CHAT_ADVANCED_MODEL")
        if (value := settings.get(name, "").strip())
        and not (value.startswith("anthropic/") or value.startswith("claude-"))
    ]


def _unused_route_settings(route: str, settings: Mapping[str, str]) -> list[str]:
    warnings: list[str] = []
    for other, names in _ROUTE_SETTINGS.items():
        stray = [
            name for name in names if other != route and settings.get(name, "").strip()
        ]
        if stray:
            warnings.append(
                f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} set, but "
                f"CHAT_PLATFORM_ROUTE is {route}: the {other} route is not used"
            )
    return warnings


def read_chat_settings(
    environ: Mapping[str, str], dotenv_paths: Iterable[Path] = ()
) -> dict[str, str]:
    """Every ``CHAT_*`` setting the backend would see, plus the few other names
    the check reads: dotenv files first and the process environment over them,
    as settings parsing layers them. An empty assignment is still a setting."""
    merged: dict[str, str] = {}
    for path in dotenv_paths:
        for name, value in dotenv_values(path).items():
            if name.startswith("CHAT_") or name in _OTHER_SETTINGS:
                merged[name] = value or ""
    for name, value in environ.items():
        if name.startswith("CHAT_") or name in _OTHER_SETTINGS:
            merged[name] = value
    return merged


def backend_dotenv_paths(cwd: Path | None = None) -> list[Path]:
    """The dotenv files the backend reads: ``.env`` in the working directory
    (settings parsing) and the backend's own ``.env`` (``backend/app.py``)."""
    found: list[Path] = []
    for path in (
        (cwd or Path.cwd()) / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ):
        if path.is_file() and all(path.resolve() != seen.resolve() for seen in found):
            found.append(path)
    return found
