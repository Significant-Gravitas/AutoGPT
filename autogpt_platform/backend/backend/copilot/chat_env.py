"""Migrate dev chat settings before service startup or standalone CLI use.

Environment overrides dotenv; a new name in either source wins over aliases.
The pure migration supplies new names and names-only deprecation notes.
run_processes exports those names to its children; ChatConfig uses the same
migration as a settings source. Legacy API fallbacks with no usable credential disable platform chat
with an error note. Local configuration gaps inherited from dev only warn. Invalid explicit route settings and cloud subscriptions
still stop startup.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from functools import lru_cache
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

# Legacy names and their migration guidance.
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
        "CHAT_POSTHOG_BUILTIN_TOOL_SPANS",
        "CHAT_READ_SCREEN_SHADOW_SAMPLE_RATE",
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
        "CHAT_LOAD_TEST_EMAIL_DOMAIN",
        "CHAT_LOAD_TEST_MODEL_URL",
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

# Non-chat names the check reads: generic endpoint overrides that used to move
# the chat, the integration keys the chat used to fall back to, and the
# deployment mode.
_INTEGRATION_KEYS = ("OPEN_ROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_OTHER_SETTINGS = (
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
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
    original = read_chat_settings(environ, dotenv_paths)
    supplied, notes = migrate_chat_settings(original)
    settings = {**original, **supplied}
    warnings = list(notes)
    warnings.extend(
        f"{name} is not a setting anything reads"
        for name in sorted(original)
        if name.startswith("CHAT_")
        and name not in RETIRED_CHAT_SETTINGS
        and name not in ACCEPTED_CHAT_SETTINGS
        and name not in CHAT_SETTINGS_READ_ELSEWHERE
    )
    problems: list[str] = []
    route = parse_platform_route(settings.get("CHAT_PLATFORM_ROUTE"))
    if route is None:
        problems.append("CHAT_PLATFORM_ROUTE is not a route")
    if (
        "CHAT_LOCAL_ADAPTER" in original
        and parse_local_adapter(original["CHAT_LOCAL_ADAPTER"]) is None
    ):
        problems.append("CHAT_LOCAL_ADAPTER is not an adapter")
    if "CHAT_LOCAL_CONTEXT_TOKENS" in original:
        problems.extend(_local_window(original["CHAT_LOCAL_CONTEXT_TOKENS"]))
    if (
        route == "subscription"
        and settings.get("BEHAVE_AS", "").strip().lower() == "cloud"
    ):
        problems.append(
            "CHAT_PLATFORM_ROUTE=subscription cannot run with BEHAVE_AS=cloud"
        )
    if "CHAT_PLATFORM_ROUTE" in original and route in (
        "openrouter",
        "direct_anthropic",
    ):
        key = (
            "CHAT_OPENROUTER_API_KEY"
            if route == "openrouter"
            else "CHAT_ANTHROPIC_API_KEY"
        )
        if (
            not settings.get(key, "").strip()
            and settings.get("CHAT_TEST_MODE", "").lower() not in _TRUE
        ):
            problems.append(f"CHAT_PLATFORM_ROUTE needs {key}")
    if route == "local":
        for name in ("CHAT_LOCAL_BASE_URL", "CHAT_STANDARD_MODEL"):
            if not settings.get(name, "").strip():
                note = f"CHAT_PLATFORM_ROUTE needs {name}"
                (problems if "CHAT_PLATFORM_ROUTE" in original else warnings).append(
                    note
                )
    if route in ("direct_anthropic", "subscription"):
        for name in ("CHAT_STANDARD_MODEL", "CHAT_ADVANCED_MODEL"):
            model = settings.get(name, "").strip()
            if model and not model.startswith(("anthropic/", "claude-")):
                note = f"{name} must name an Anthropic model"
                (problems if name in original else warnings).append(note)
    return problems, warnings


def migrate_chat_settings(
    settings: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Pure migration: supplied new names and one names-only note per old name."""
    supplied: dict[str, str] = {}
    notes = {
        name: f"{name} deprecated; ignored"
        for name in settings
        if name in RETIRED_CHAT_SETTINGS
    }
    legacy = "CHAT_PLATFORM_ROUTE" not in settings

    def get(name: str) -> str:
        return settings.get(name, supplied.get(name, ""))

    def copy(target: str, *sources: str, strip_v1: bool = False) -> None:
        chosen = next(
            (name for name in sources if settings.get(name, "").strip()), None
        )
        for name in sources:
            if name in settings and name in RETIRED_CHAT_SETTINGS:
                notes[name] = f"{name} deprecated; ignored in favour of {target}"
        if chosen is None:
            return
        value = settings[chosen]
        if strip_v1:
            value = value.rstrip("/")
            if value.endswith("/v1"):
                value = value[:-3]
        if target in settings:
            if settings[target] != value and chosen in RETIRED_CHAT_SETTINGS:
                notes[chosen] = (
                    f"{chosen} deprecated; conflicts with {target}; {target} wins"
                )
        else:
            supplied[target] = value
            notes[chosen] = (
                f"{chosen} deprecated; use {target}"
                if chosen in RETIRED_CHAT_SETTINGS
                else f"{target} is not set; chat uses {chosen} as dev did; set {target}"
            )

    def first(target: str, *aliases: str, default: str = "") -> str:
        if target in settings:
            return settings[target]
        return next((settings[name] for name in aliases if settings.get(name)), default)

    router_key = first(
        "CHAT_OPENROUTER_API_KEY",
        "CHAT_API_KEY",
        "OPEN_ROUTER_API_KEY",
        "OPENAI_API_KEY",
    )
    router_url = first(
        "CHAT_OPENROUTER_BASE_URL",
        "CHAT_BASE_URL",
        "OPENROUTER_BASE_URL",
        "OPENAI_BASE_URL",
        default="https://openrouter.ai/api",
    )
    if get("CHAT_USE_LOCAL").strip().lower() in _TRUE:
        old_route = "local"
    elif get("CHAT_USE_CLAUDE_CODE_SUBSCRIPTION").strip().lower() in _TRUE:
        old_route = "subscription"
    elif (
        get("CHAT_USE_OPENROUTER").strip().lower() not in ("false", "0", "no", "off")
        and router_key
        and router_url.startswith(("http://", "https://"))
    ):
        old_route = "openrouter"
    else:
        old_route = "direct_anthropic"
    route = old_route if legacy else get("CHAT_PLATFORM_ROUTE").strip()
    if legacy:
        supplied["CHAT_PLATFORM_ROUTE"] = route
    for name in (
        "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
        "CHAT_USE_LOCAL",
        "CHAT_USE_OPENROUTER",
    ):
        if name in settings:
            notes[name] = (
                f"{name} deprecated; conflicts with CHAT_PLATFORM_ROUTE; CHAT_PLATFORM_ROUTE wins"
                if not legacy and route != old_route
                else f"{name} deprecated; use CHAT_PLATFORM_ROUTE"
            )
    if route == "openrouter":
        copy(
            "CHAT_OPENROUTER_API_KEY",
            "CHAT_API_KEY",
            *(("OPEN_ROUTER_API_KEY", "OPENAI_API_KEY") if legacy else ()),
        )
        copy(
            "CHAT_OPENROUTER_BASE_URL",
            "CHAT_BASE_URL",
            *(("OPENROUTER_BASE_URL", "OPENAI_BASE_URL") if legacy else ()),
            strip_v1=True,
        )
    elif route == "local":
        copy("CHAT_LOCAL_API_KEY", "CHAT_API_KEY")
        copy("CHAT_LOCAL_BASE_URL", "CHAT_BASE_URL")
    copy(
        "CHAT_ANTHROPIC_API_KEY",
        "CHAT_DIRECT_ANTHROPIC_API_KEY",
        *(("ANTHROPIC_API_KEY",) if legacy else ()),
    )
    if route == "direct_anthropic" and legacy:
        copy("CHAT_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL")
    local = route == "local"
    copy(
        "CHAT_STANDARD_MODEL",
        *(
            ("CHAT_FAST_STANDARD_MODEL", "CHAT_FAST_MODEL")
            if local
            else ("CHAT_THINKING_STANDARD_MODEL", "CHAT_MODEL")
        ),
        "CHAT_CLAUDE_AGENT_MODEL",
    )
    copy(
        "CHAT_ADVANCED_MODEL",
        "CHAT_FAST_ADVANCED_MODEL" if local else "CHAT_THINKING_ADVANCED_MODEL",
        "CHAT_CLAUDE_AGENT_MODEL",
    )
    copy("CHAT_AGENT_MAX_TURNS", "CHAT_CLAUDE_AGENT_MAX_TURNS")
    if "CHAT_AGENT_MAX_TURNS" in supplied:
        try:
            int(supplied["CHAT_AGENT_MAX_TURNS"])
        except ValueError:
            del supplied["CHAT_AGENT_MAX_TURNS"]
            notes["CHAT_CLAUDE_AGENT_MAX_TURNS"] = (
                "CHAT_CLAUDE_AGENT_MAX_TURNS deprecated; ignored; set CHAT_AGENT_MAX_TURNS"
            )
    if local and (
        legacy
        or any(
            name in settings
            for name in (
                "CHAT_FAST_STANDARD_MODEL",
                "CHAT_FAST_MODEL",
                "CHAT_FAST_ADVANCED_MODEL",
                "CHAT_CLAUDE_AGENT_MODEL",
                "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW",
            )
        )
    ):
        supplied.setdefault(
            "CHAT_STANDARD_MODEL",
            get("CHAT_STANDARD_MODEL") or "anthropic/claude-sonnet-5-5",
        )
        if "CHAT_ADVANCED_MODEL" not in settings:
            advanced = get("CHAT_ADVANCED_MODEL")
            if not advanced or advanced.startswith(
                (
                    "openai/",
                    "anthropic/",
                    "google/",
                    "meta-llama/",
                    "mistralai/",
                    "deepseek/",
                    "qwen/",
                    "x-ai/",
                    "cohere/",
                )
            ):
                supplied["CHAT_ADVANCED_MODEL"] = get("CHAT_STANDARD_MODEL")
    if local:
        if "CHAT_LOCAL_ADAPTER" not in settings:
            supplied["CHAT_LOCAL_ADAPTER"] = "openai_chat"
        if "CHAT_LOCAL_CONTEXT_TOKENS" not in settings:
            window = settings.get("CHAT_CLAUDE_AGENT_CONTEXT_WINDOW", "")
            supplied["CHAT_LOCAL_CONTEXT_TOKENS"] = (
                window if not _local_window(window) else str(MIN_LOCAL_CONTEXT_TOKENS)
            )
            notes[
                (
                    "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW"
                    if "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW" in settings
                    else "CHAT_LOCAL_CONTEXT_TOKENS"
                )
            ] = (
                "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW deprecated; set CHAT_LOCAL_CONTEXT_TOKENS to the server window"
                if "CHAT_CLAUDE_AGENT_CONTEXT_WINDOW" in settings
                else "CHAT_LOCAL_CONTEXT_TOKENS defaulted; set the server window"
            )
    if (
        legacy
        and route == "direct_anthropic"
        and not get("CHAT_ANTHROPIC_API_KEY").strip()
    ):
        supplied["CHAT_PLATFORM_ROUTE"] = "disabled"
        notes["CHAT_PLATFORM_ROUTE"] = (
            "CHAT_PLATFORM_ROUTE disabled; set CHAT_PLATFORM_ROUTE and "
            "CHAT_ANTHROPIC_API_KEY or CHAT_OPENROUTER_API_KEY"
        )
    return supplied, [notes[name] for name in sorted(notes)]


def _local_window(window: str) -> list[str]:
    if not window.strip().isdecimal():
        return ["CHAT_LOCAL_CONTEXT_TOKENS must be a whole number"]
    try:
        size = int(window)
    except ValueError:
        return ["CHAT_LOCAL_CONTEXT_TOKENS must be a whole number"]
    if size < MIN_LOCAL_CONTEXT_TOKENS:
        return ["CHAT_LOCAL_CONTEXT_TOKENS is below MIN_LOCAL_CONTEXT_TOKENS"]
    return []


_logged_names: set[tuple[int, str]] = set()


def log_chat_notes(notes: Iterable[str]) -> None:
    """Log each named setting once per process, never its value."""
    logger = logging.getLogger(__name__)
    for note in notes:
        name = note.split()[0]
        key = (os.getpid(), name)
        if key not in _logged_names:
            _logged_names.add(key)
            logger.log(
                logging.ERROR if name == "CHAT_PLATFORM_ROUTE" else logging.WARNING,
                note,
            )


_MIGRATION_NAMES = (
    frozenset(RETIRED_CHAT_SETTINGS)
    | frozenset(_OTHER_SETTINGS)
    | frozenset(
        {
            "CHAT_PLATFORM_ROUTE",
            "CHAT_OPENROUTER_API_KEY",
            "CHAT_OPENROUTER_BASE_URL",
            "CHAT_ANTHROPIC_API_KEY",
            "CHAT_ANTHROPIC_BASE_URL",
            "CHAT_LOCAL_API_KEY",
            "CHAT_LOCAL_BASE_URL",
            "CHAT_LOCAL_ADAPTER",
            "CHAT_LOCAL_CONTEXT_TOKENS",
            "CHAT_STANDARD_MODEL",
            "CHAT_ADVANCED_MODEL",
            "CHAT_AGENT_MAX_TURNS",
        }
    )
)


def migration_input(*sources: Mapping[str, object]) -> dict[str, str]:
    """Use already parsed settings sources; do not read dotenv again."""
    merged: dict[str, str] = {}
    for source in sources:
        for name in _MIGRATION_NAMES:
            value = source.get(name, source.get(name.lower()))
            if value is not None:
                merged[name] = str(value)
    return merged


@lru_cache(maxsize=32)
def _cached_chat_migration(
    pid: int, snapshot: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    supplied, notes = migrate_chat_settings(dict(snapshot))
    log_chat_notes(notes)
    return tuple(supplied.items())


def cached_chat_migration(settings: Mapping[str, str]) -> dict[str, str]:
    """Memoise migration per process and relevant settings, logging on misses."""
    return dict(_cached_chat_migration(os.getpid(), tuple(sorted(settings.items()))))


def apply_chat_environment() -> None:
    """Migrate before services start so child processes inherit supplied names."""
    supplied, _ = migrate_chat_settings(
        read_chat_settings(os.environ, backend_dotenv_paths())
    )
    notes = check_chat_environment()
    log_chat_notes(notes)
    os.environ.update(supplied)


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
