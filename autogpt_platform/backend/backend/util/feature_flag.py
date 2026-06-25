import contextlib
import logging
import os
import uuid
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import ldclient
from autogpt_libs.auth.dependencies import get_optional_user_id
from fastapi import HTTPException, Security
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import ParamSpec

from backend.util.cache import cached
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# Load settings at module level
settings = Settings()

P = ParamSpec("P")
T = TypeVar("T")

_is_initialized = False


class Flag(str, Enum):
    """
    Centralized enum for all LaunchDarkly feature flags.

    Add new flags here to ensure consistency across the codebase.
    """

    AUTOMOD = "AutoMod"
    AI_ACTIVITY_STATUS = "ai-agent-execution-summary"
    BETA_BLOCKS = "beta-blocks"
    AGENT_ACTIVITY = "agent-activity"
    ENABLE_PLATFORM_PAYMENT = "enable-platform-payment"
    CHAT = "chat"
    CHAT_MODE_OPTION = "chat-mode-option"
    # Gates the "share chat results" feature end-to-end.  Backend create
    # routes refuse when off so a stale frontend cannot enable shares;
    # frontend share button hides when off so the UI doesn't tease a
    # feature that won't take.  Existing public viewer routes stay on
    # regardless so previously-shared URLs remain valid mid-flight.
    CHAT_SHARING = "chat-sharing"
    COPILOT_SDK = "copilot-sdk"
    COPILOT_COST_LIMITS = "copilot-cost-limits"
    # Self-distilled skills registry (store_skill / read_skill /
    # delete_skill / list_skills + the per-turn <available_skills>
    # context block).  Default-on — flip off in LaunchDarkly to disable
    # the feature without a redeploy.
    COPILOT_SKILLS = "copilot-skills"
    # Scheduled copilot turn followups (schedule_followup MCP tool +
    # the pending_followups awareness inside <session_context>).  The
    # current_session_id line stays regardless — only the followup
    # surface is gated.  Default-on.
    COPILOT_SCHEDULED_FOLLOWUPS = "copilot-scheduled-followups"
    COPILOT_TIER_MULTIPLIERS = "copilot-tier-multipliers"
    COPILOT_TIER_WORKSPACE_STORAGE_LIMITS = "copilot-tier-workspace-storage-limits"
    COPILOT_TIER_STRIPE_PRICES = "copilot-tier-stripe-prices"
    GRAPHITI_MEMORY = "graphiti-memory"

    # Gates the per-user weekly community rebuild registered by
    # ``add_community_rebuild_schedule``. Off by default; opt-in canary
    # so the Leiden + LLM-summarization cost doesn't ramp before
    # retrieval-relevance benefit is measured.
    GRAPHITI_COMMUNITIES_ENABLED = "graphiti-communities-enabled"

    # --- Dream-system gates (P0) ---
    #
    # No "enabled-users list" flag — LD's per-flag targeting natively
    # cohorts the canary (internal team → 5 → 50 → 500 → 5k), and
    # ``is_feature_enabled(..., user_id, ...)`` evaluates each user.
    # The four flags below are the master gate + three per-feature
    # gates. Helper functions live next to the code that consumes them
    # (added when each feature lands); these enum entries are
    # scaffolding so the LD keys can be configured ahead of code.

    # Master gate for the dream pass. When off, no dream-related code
    # paths fire for the user — no schedule registration, no batch
    # consumer pickup, no ratification loop. Defaults False; opt-in
    # only.
    DREAM_PASS_ENABLED = "dream-pass-enabled"

    # Per-feature gate for the web-fact-check tool (P0.5). The tool
    # can only DEMOTE memories on contradiction; new web-derived
    # facts ride the ratification loop as tentative. Off on the
    # local-LLM transport by default (most local installs lack a
    # search-API key); cloud opt-in. Independent of
    # ``DREAM_PASS_ENABLED`` so the dream pass can run without
    # external network calls when this flag is off.
    DREAM_PASS_WEB_FACT_CHECK = "dream-pass-web-fact-check"

    # Orchestrator-level kill switch for the web-fact-check hook
    # introduced alongside the P0.5 scaffolding. Distinct from
    # ``DREAM_PASS_WEB_FACT_CHECK`` so the hook can be wired into the
    # orchestrator without auto-running on every dream pass before a
    # search backend is bound — flip this on per-user once a backend
    # is configured.
    DREAM_WEB_FACT_CHECK_ENABLED = "dream-web-fact-check-enabled"

    # Per-feature gate for the cascading-expiry helper
    # ``invalidate_entity_direct_neighbors`` (P0.3b). When on, the
    # dream pass may demote every ``:RELATES_TO`` edge directly
    # attached to a flagged-as-dead entity (e.g. "this client is
    # gone — expire their associated facts"). Single-hop only —
    # tangentially-related edges are NOT touched. Off by default for
    # the first two weeks after launch because this is the highest
    # blast-radius dream action; a buggy phase-3 sanitizer could
    # demote too many edges before we catch it. Ratification (P0.4)
    # is how good edges caught in the cascade get re-promoted.
    DREAM_PASS_INVALIDATE_ENTITY = "dream-pass-invalidate-entity"

    # Rollout gate for the async Anthropic batch path (P0.1). When on AND a
    # direct Anthropic API key is configured, the dream pass routes through
    # Anthropic's Batch API (~50% cheaper, async, up to 24h) and runs its
    # memory writeback in the batch callback; when off, dreams run on the
    # synchronous baseline regardless of key presence. A direct key is a hard
    # requirement — the native Batch API can't be reached via
    # OpenRouter/subscription — so this flag gates rollout on top of
    # key-presence, not instead of it. Defaults False so the batch path ships
    # dark and is enabled per-cohort.
    DREAM_PASS_BATCH_ENABLED = "dream-pass-batch-enabled"

    # Note: there is intentionally no DREAM_PASS_LOCAL_TRANSPORT
    # flag. Whether to run on local-LLM transport is a CODE decision
    # — ``resolve_dream_execution_path()`` in
    # ``copilot/dream/routing.py`` inspects ``config.transport`` and
    # picks sync-baseline + phase-collapse + extended lock TTL for
    # local users. If a user has ``DREAM_PASS_ENABLED`` true, they
    # get dreams, period — degraded on local, full on cloud. See
    # ``dream/p0-spec.md`` §13.

    GENERIC_TRIGGER_AGENTS = "generic-trigger-agents"
    # Stripe Product ID for top-up Checkout sessions. When unset (default),
    # top_up_intent uses inline product_data (creates ephemeral Stripe products
    # per Checkout). When set to a real Stripe Product ID, line items reference
    # that Product so dashboard reporting groups all top-ups under one entity;
    # the per-Checkout amount stays dynamic via price_data.unit_amount.
    STRIPE_PRODUCT_ID_TOPUP = "stripe-product-id-topup"

    # Copilot model routing — JSON-valued, returns the per-(mode, tier)
    # model identifier (e.g. ``"anthropic/claude-sonnet-4-6"`` or
    # ``"moonshotai/kimi-k2.6"``).  Shape:
    # ``{"fast": {"standard": "...", "advanced": "..."},
    #   "thinking": {"standard": "...", "advanced": "..."}}``.
    # Missing mode, missing tier-within-mode, non-string value, non-dict
    # payload, or LD failure all fall back to the corresponding
    # ``ChatConfig`` default.  Evaluated per user_id so cohorts can be
    # targeted.
    COPILOT_MODEL_ROUTING = "copilot-model-routing"


def is_configured() -> bool:
    """Check if LaunchDarkly is configured with an SDK key."""
    return bool(settings.secrets.launch_darkly_sdk_key)


def get_client() -> LDClient:
    """Get the LaunchDarkly client singleton."""
    if not _is_initialized:
        initialize_launchdarkly()
    return ldclient.get()


def initialize_launchdarkly() -> None:
    sdk_key = settings.secrets.launch_darkly_sdk_key
    logger.debug(
        f"Initializing LaunchDarkly with SDK key: {'present' if sdk_key else 'missing'}"
    )

    if not sdk_key:
        logger.warning("LaunchDarkly SDK key not configured")
        return

    config = Config(sdk_key)
    ldclient.set_config(config)

    global _is_initialized
    _is_initialized = True
    if ldclient.get().is_initialized():
        logger.info("LaunchDarkly client initialized successfully")
    else:
        logger.error("LaunchDarkly client failed to initialize")


def shutdown_launchdarkly() -> None:
    """Shutdown the LaunchDarkly client."""
    if ldclient.get().is_initialized():
        ldclient.get().close()
        logger.info("LaunchDarkly client closed successfully")


async def _fetch_user_context_data(user_id: str) -> Context:
    """
    Fetch user context for LaunchDarkly from Supabase.

    Successful lookups are cached for 24h (see
    ``_fetch_supabase_user_context``).  Failed lookups are NOT cached: the
    degraded anonymous fallback is built outside the cache so the next
    evaluation retries Supabase instead of pinning this process to an
    email-less context for a full TTL — which would make its
    email/role-targeted flag evaluations silently diverge from peer
    processes.  The degraded path costs one failed Supabase call per
    evaluation; bounded, and acceptable versus a 24h-poisoned cache.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        LaunchDarkly Context object
    """
    try:
        uuid.UUID(user_id)
    except ValueError:
        # Non-UUID key (e.g. "system") — skip Supabase lookup, return anonymous context.
        return _anonymous_context(user_id)

    try:
        return await _fetch_supabase_user_context(user_id)
    except Exception as e:
        logger.warning(
            f"Failed to fetch user context for {user_id}: {e} — "
            "falling back to an uncached anonymous context; flag "
            "evaluations for this user may be degraded until the lookup "
            "succeeds"
        )
        return _anonymous_context(user_id)


def _anonymous_context(user_id: str) -> Context:
    """Build a minimal anonymous LD context carrying only the user key."""
    return Context.builder(user_id).kind("user").anonymous(True).build()


@cached(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_supabase_user_context(user_id: str) -> Context:
    """
    Build the full LaunchDarkly context for ``user_id`` from Supabase.

    Raises on Supabase lookup failure: ``@cached`` never stores results
    of calls that raise, so a degraded context can't be cached here —
    the caller handles the fallback outside the cache.
    """
    from backend.util.clients import get_supabase

    builder = Context.builder(user_id).kind("user").anonymous(True)

    # If we have user data, update context
    response = get_supabase().auth.admin.get_user_by_id(user_id)
    if response and response.user:
        user = response.user
        builder.anonymous(False)
        if user.role:
            builder.set("role", user.role)
            # It's weird, I know, but it is what it is.
            builder.set("custom", {"role": user.role})
        if user.email:
            builder.set("email", user.email)
            builder.set("email_domain", user.email.split("@")[-1])
        if user.created_at:
            # ISO-8601 string — LD supports RFC3339 date targeting on
            # this attribute (e.g. cohort users by signup window).
            builder.set("created_at", user.created_at.isoformat())

    return builder.build()


async def get_feature_flag_value(
    flag_key: str,
    user_id: str,
    default: Any = None,
) -> Any:
    """
    Get the raw value of a feature flag for a user.

    This is the generic function that returns the actual flag value,
    which could be a boolean, string, number, or JSON object.

    Args:
        flag_key: The LaunchDarkly feature flag key
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        The flag value from LaunchDarkly
    """
    try:
        client = get_client()

        # Check if client is initialized
        if not client.is_initialized():
            logger.debug(
                f"LaunchDarkly not initialized, using default={default} for {flag_key}"
            )
            return default

        # Get user context from Supabase
        context = await _fetch_user_context_data(user_id)

        # Evaluate flag
        result = client.variation(flag_key, context, default)

        logger.debug(
            f"Feature flag {flag_key} for user {user_id}: {result} (type: {type(result).__name__})"
        )
        return result

    except Exception as e:
        logger.warning(
            f"LaunchDarkly flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default


def _env_flag_override(flag_key: Flag) -> bool | None:
    """Return a local override for ``flag_key`` from the environment.

    Set ``FORCE_FLAG_<NAME>=true|false`` (``NAME`` = flag value with
    ``-`` → ``_``, upper-cased) to bypass LaunchDarkly for a single
    flag in local dev or tests.  Returns ``None`` when no override
    is configured so the caller falls through to LaunchDarkly.

    The ``NEXT_PUBLIC_FORCE_FLAG_<NAME>`` prefix is also accepted so a
    single shared env var can toggle a flag across backend and
    frontend (the frontend requires the ``NEXT_PUBLIC_`` prefix to
    expose the value to the browser bundle).

    Example: ``FORCE_FLAG_CHAT_MODE_OPTION=true`` forces
    ``Flag.CHAT_MODE_OPTION`` on regardless of LaunchDarkly.
    """
    suffix = flag_key.value.upper().replace("-", "_")
    for prefix in ("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_"):
        raw = os.environ.get(prefix + suffix)
        if raw is not None:
            return raw.strip().lower() in ("1", "true", "yes", "on")
    return None


async def is_feature_enabled(
    flag_key: Flag,
    user_id: str,
    default: bool = False,
) -> bool:
    """
    Check if a feature flag is enabled for a user.

    Args:
        flag_key: The Flag enum value
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        True if feature is enabled, False otherwise
    """
    override = _env_flag_override(flag_key)
    if override is not None:
        logger.debug(f"Feature flag {flag_key} overridden by env: {override}")
        return override

    result = await get_feature_flag_value(flag_key.value, user_id, default)

    # If the result is already a boolean, return it
    if isinstance(result, bool):
        return result

    # Log a warning if the flag is not returning a boolean
    logger.warning(
        f"Feature flag {flag_key} returned non-boolean value: {result} (type: {type(result).__name__}). "
        f"This flag should be configured as a boolean in LaunchDarkly. Using default={default}"
    )

    # Return the default if we get a non-boolean value
    # This prevents objects from being incorrectly treated as True
    return default


def feature_flag(
    flag_key: str,
    default: bool = False,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async feature flag protected endpoints.

    Args:
        flag_key: The LaunchDarkly feature flag key
        default: Default value if flag evaluation fails

    Returns:
        Decorator that only works with async functions
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                user_id = kwargs.get("user_id")
                if not user_id:
                    raise ValueError("user_id is required")

                if not get_client().is_initialized():
                    logger.warning(
                        "LaunchDarkly not initialized, "
                        f"using default {flag_key}={repr(default)}"
                    )
                    is_enabled = default
                else:
                    # Use the internal function directly since we have a raw string flag_key
                    flag_value = await get_feature_flag_value(
                        flag_key, str(user_id), default
                    )
                    # Ensure we treat flag value as boolean
                    if isinstance(flag_value, bool):
                        is_enabled = flag_value
                    else:
                        # Log warning and use default for non-boolean values
                        logger.warning(
                            f"Feature flag {flag_key} returned non-boolean value: "
                            f"{repr(flag_value)} (type: {type(flag_value).__name__}). "
                            f"Using default value {repr(default)}"
                        )
                        is_enabled = default

                if not is_enabled:
                    raise HTTPException(status_code=404, detail="Feature not available")

                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error evaluating feature flag {flag_key}: {e}")
                raise

        return async_wrapper

    return decorator


def create_feature_flag_dependency(
    flag_key: Flag,
    default: bool = False,
) -> Callable[[str | None], Awaitable[None]]:
    """
    Create a FastAPI dependency that checks a feature flag.

    This dependency automatically extracts the user_id from the JWT token
    (if present) for proper LaunchDarkly user targeting, while still
    supporting anonymous access.

    Args:
        flag_key: The Flag enum value to check
        default: Default value if flag evaluation fails

    Returns:
        An async dependency function that raises HTTPException if flag is disabled

    Example:
        router = APIRouter(
            dependencies=[Depends(create_feature_flag_dependency(Flag.CHAT))]
        )
    """

    async def check_feature_flag(
        user_id: str | None = Security(get_optional_user_id),
    ) -> None:
        """Check if feature flag is enabled for the user.

        The user_id is automatically injected from JWT authentication if present,
        or None for anonymous access.
        """
        # For routes that don't require authentication, use anonymous context
        check_user_id = user_id or "anonymous"

        if not is_configured():
            logger.debug(
                f"LaunchDarkly not configured, using default {flag_key.value}={default}"
            )
            if not default:
                raise HTTPException(status_code=404, detail="Feature not available")
            return

        try:
            client = get_client()
            if not client.is_initialized():
                logger.debug(
                    f"LaunchDarkly not initialized, using default {flag_key.value}={default}"
                )
                if not default:
                    raise HTTPException(status_code=404, detail="Feature not available")
                return

            is_enabled = await is_feature_enabled(flag_key, check_user_id, default)

            if not is_enabled:
                raise HTTPException(status_code=404, detail="Feature not available")
        except Exception as e:
            logger.warning(
                f"LaunchDarkly error for flag {flag_key.value}: {e}, using default={default}"
            )
            raise HTTPException(status_code=500, detail="Failed to check feature flag")

    return check_feature_flag


@contextlib.contextmanager
def mock_flag_variation(flag_key: str, return_value: Any):
    """Context manager for testing feature flags."""
    original_variation = get_client().variation
    get_client().variation = lambda key, context, default: (
        return_value if key == flag_key else original_variation(key, context, default)
    )
    try:
        yield
    finally:
        get_client().variation = original_variation
