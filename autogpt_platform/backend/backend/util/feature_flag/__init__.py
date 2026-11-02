import logging
import os
import uuid
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import sentry_sdk.feature_flags
from autogpt_libs.auth.dependencies import get_optional_user_id
from fastapi import HTTPException, Security
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import ParamSpec

from backend.util.cache import cached
from backend.util.settings import AppEnvironment, Settings

from . import posthog

logger = logging.getLogger(__name__)

# Load settings at module level
settings = Settings()

P = ParamSpec("P")
T = TypeVar("T")

# The one key every signed-out caller without a consented analytics id is read as.
ANONYMOUS_USER = "anonymous"


# Flags that run a PostHog experiment read only here. Only a read of one of these
# sends the billed `$feature_flag_called` event, which is how an experiment counts
# exposure. A server-assigned arm (`data/experiments.py`) is not listed: the
# browser reports its exposure, consent-gated.
EXPERIMENT_FLAGS: frozenset[str] = frozenset()


class Flag(str, Enum):
    """
    Centralized enum for all feature flags.

    Add new flags here to ensure consistency across the codebase.
    """

    AUTOMOD = "AutoMod"
    ENABLE_PLATFORM_PAYMENT = "enable-platform-payment"
    CHAT_MODE_OPTION = "chat-mode-option"
    COPILOT_SDK = "copilot-sdk"
    COPILOT_COST_LIMITS = "copilot-cost-limits"
    # AutoPilot auto mode: the permission gate in ``copilot/gate``.
    # A ROLLOUT control, not a safety control — off means today's
    # ungated behaviour, which is why it defaults False rather than
    # fail-closed. Failures *inside* an enabled gate fail to ASK.
    COPILOT_AUTO_MODE = "copilot-auto-mode"
    # The prompt-injection check on outside reads (``copilot/gate/reads.py``):
    # true or "enforce" holds a flagged read for the user; "shadow" judges every
    # read and records the verdict, holding nothing. Off when missing.
    COPILOT_READ_SCREEN = "copilot-read-screen"
    # The gate answers the review pauses of workflows AutoPilot started, and
    # its yes satisfies a direct block call's own review. A rollout control
    # that defaults ON by decision; off restores the two approval layers.
    COPILOT_UNIFIED_APPROVALS = "copilot-unified-approvals"
    # Self-distilled skills registry (store_skill / read_skill /
    # delete_skill / list_skills + the per-turn <available_skills>
    # context block).  Default-on — flip off in PostHog to disable
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
    CARD_REQUIRED_TRIAL_OFFER = "card-required-trial-offer"

    # Gates Otto voice mode end-to-end. The speech endpoint 404s when
    # off so a stale frontend cannot spend TTS budget. Fail-closed.
    COPILOT_VOICE_MODE = "copilot-voice-mode"

    # --- Dream-system gates (P0) ---
    #
    # No "enabled-users list" flag — per-flag targeting natively
    # cohorts the canary (internal team → 5 → 50 → 500 → 5k), and
    # ``is_feature_enabled(..., user_id, ...)`` evaluates each user.
    # The flags below are the master gate + its per-feature
    # gates. Helper functions live next to the code that consumes them
    # (added when each feature lands); these enum entries are
    # scaffolding so the flag keys can be configured ahead of code.

    # Master gate for the dream pass. When off, no dream-related code
    # paths fire for the user — no schedule registration, no batch
    # consumer pickup, no ratification loop. Defaults False; opt-in
    # only.
    DREAM_PASS_ENABLED = "dream-pass-enabled"

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

    # Nightly skill learning stage: turns eligible ordinary conversations
    # into versioned, evidence-linked skill updates for the owning scope.
    # Independent of ``DREAM_PASS_ENABLED`` — it registers its own cron
    # row and enabling it does not switch on any other experimental
    # memory stage. Defaults False; opt-in per cohort.
    DREAM_SKILL_LEARNING_ENABLED = "dream-skill-learning-enabled"

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
    # payload, or a failed read all fall back to the corresponding
    # ``ChatConfig`` default.  Evaluated per user_id so cohorts can be
    # targeted.
    COPILOT_MODEL_ROUTING = "copilot-model-routing"

    # Shows a connection the user's plan does not include as a locked entry
    # in the connection list, rather than omitting it.  Merchandising, not
    # access: the entitlement still decides what may actually run, and a
    # locked offer is never routable.  Off by default so the upsell reaches
    # a cohort before it reaches everyone.
    CHAT_CONNECTION_UPSELL = "chat-connection-upsell"


def initialize_feature_flags() -> None:
    """Start the PostHog flag client and load its definitions."""
    if posthog.is_configured() and not settings.secrets.posthog_secret_key:
        logger.warning(
            "Feature flags without POSTHOG_SECRET_KEY: every flag read becomes a "
            "remote /flags call instead of an in-process evaluation"
        )
    posthog.initialize_posthog_flags()


def shutdown_feature_flags() -> None:
    """Reverse of :func:`initialize_feature_flags`."""
    posthog.shutdown_posthog_flags()


class _UserContext(BaseModel):
    """Who a flag read is for, as the targeting rules see them.

    ``attributes`` holds the targeted facts (role, email, signup date and any
    per-request ones such as the country); an anonymous context carries none.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    anonymous: bool = True
    attributes: dict[str, Any] = Field(default_factory=dict)


async def _fetch_user_context_status(user_id: str) -> tuple[_UserContext, bool]:
    """``(context, resolved)`` for the user a flag is read for.

    Successful lookups are cached for 24h (see ``_fetch_user_context``).
    Failed lookups are NOT cached: the degraded anonymous fallback is built
    outside the cache so the next evaluation retries the lookup instead of
    pinning this process to an email-less context for a full TTL — which
    would make its email/role-targeted flag evaluations silently diverge
    from peer processes.  The degraded path costs one failed lookup per
    evaluation; bounded, and acceptable versus a 24h-poisoned cache.

    ``resolved`` is False only when the lookup FAILED and the anonymous
    context is standing in for real user data. A non-UUID key such as
    ``"system"`` is anonymous by design and counts as resolved. The
    distinction matters because an evaluation against a degraded context
    still succeeds — it just answers for the wrong user — so callers acting
    irreversibly on a ``False`` must not trust one.
    """
    try:
        uuid.UUID(user_id)
    except ValueError:
        # Non-UUID key (e.g. "system") — skip user lookup, return anonymous context.
        return _anonymous_context(user_id), True

    try:
        return await _fetch_user_context(user_id), True
    except Exception as e:
        logger.warning(
            f"Failed to fetch user context for {user_id}: {e} — "
            "falling back to an uncached anonymous context; flag "
            "evaluations for this user may be degraded until the lookup "
            "succeeds"
        )
        return _anonymous_context(user_id), False


def _with_request_attributes(
    context: _UserContext, attributes: dict[str, str]
) -> _UserContext:
    """*context* plus facts known only for this request, such as the country.

    The user context is cached for a day, so anything that can change between
    requests has to be layered on per evaluation rather than baked into it.
    A copy is returned; the cached context is never modified. An anonymous
    context still sends none of them (see ``_context_properties``).
    """
    return context.model_copy(
        update={"attributes": {**context.attributes, **attributes}}
    )


def _anonymous_context(user_id: str) -> _UserContext:
    """A minimal anonymous context carrying only the user key."""
    return _UserContext(key=user_id)


@cached(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_user_context(user_id: str) -> _UserContext:
    """
    Build the full flag context for ``user_id`` from the auth user table.

    Raises on lookup failure: ``@cached`` never stores results of calls
    that raise, so a degraded context can't be cached here — the caller
    handles the fallback outside the cache.
    """
    # Local import to avoid a util <-> data import cycle.
    from backend.data.db_accessors import user_db

    # user_db() falls back to the DatabaseManager RPC client in processes
    # without a locally-connected Prisma client (scheduler, executors, ...).
    fields = await user_db().get_auth_user_flag_fields(user_id)

    if fields is None:
        # Raise instead of returning an anonymous context: @cached would pin
        # the anonymous result for 24h even after the user's row appears
        # (possible during the auth-migration copy window). The caller falls
        # back to an uncached anonymous context.
        raise LookupError(f"No auth user row for {user_id}")

    # Keep the same role values previously issued in JWTs so existing
    # targeting rules keep matching.
    attributes: dict[str, Any] = {
        "role": "admin" if fields.role == "admin" else "authenticated"
    }
    if fields.email:
        attributes["email"] = fields.email
        attributes["email_domain"] = fields.email.split("@")[-1]
    if fields.created_at:
        # ISO-8601 string, so rules can cohort users by signup window.
        attributes["created_at"] = fields.created_at.isoformat()

    return _UserContext(key=user_id, anonymous=False, attributes=attributes)


async def get_feature_flag_value(
    flag_key: str,
    user_id: str,
    default: Any = None,
    *,
    attributes: dict[str, str] | None = None,
) -> Any:
    """
    Get the raw value of a feature flag for a user.

    This is the generic function that returns the actual flag value,
    which could be a boolean, string, number, or JSON object.

    Args:
        flag_key: The feature flag key
        user_id: The user ID to evaluate the flag for
        default: Default value if PostHog is unavailable or evaluation fails
        attributes: Facts known only for this request, such as ``country``,
            sent with the user's person properties for this read only

    Returns:
        The flag value from PostHog
    """
    value, _ = await _evaluate_flag_value(
        flag_key, user_id, default, attributes=attributes
    )
    return value


async def evaluate_experiment_variant(flag_key: str, user_id: str) -> str | None:
    """The arm PostHog buckets *user_id* into for an experiment, or None.

    None too when the user lookup failed: the first arm is stored for good, and
    one read without the user's properties can be the wrong one.
    """
    user_context, resolved = await _fetch_user_context_status(user_id)
    if not resolved:
        return None
    return await posthog.evaluate_variant(
        flag_key, user_id, _person_properties(user_context)
    )


def evaluate_flags_in_memory(flag_keys: list[str], key: str) -> dict[str, Any]:
    """Each of *flag_keys* PostHog answers in memory for the anonymous *key*.

    No network, no analytics event and no executor, so a caller may pick *key*:
    a new one costs only the hashing. A flag that needs more is left out.
    """
    values = posthog.evaluate_flags_in_memory(
        flag_keys, key, _person_properties(_anonymous_context(key))
    )
    return {
        flag_key: values[flag_key]
        for flag_key in flag_keys
        if values.get(flag_key) is not None
    }


def defined_flag_keys() -> frozenset[str] | None:
    """Every flag PostHog has a definition of, or None if it cannot tell yet.

    A key outside the set will never be answered, so a reader can stop waiting on it.
    """
    return posthog.defined_flag_keys()


async def _evaluate_flag_value(
    flag_key: str,
    user_id: str,
    default: Any = None,
    *,
    attributes: dict[str, str] | None = None,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` for one raw flag read.

    ``evaluated`` is False whenever *default* is standing in for an answer
    PostHog could not give. A degraded context lookup makes the answer
    unauthoritative too: the flag evaluates fine, it just answers for a user
    without the targeted attributes. Only a read of an experiment flag sends
    PostHog its exposure event.
    """
    if posthog.get_flag_client() is None:
        # No client means the default, without a user lookup.
        result = (default, False)
    else:
        context, resolved = await _fetch_user_context_status(user_id)
        if attributes:
            context = _with_request_attributes(context, attributes)
        value, evaluated = await posthog.evaluate_flag(
            flag_key,
            user_id,
            _person_properties(context),
            default,
            exposure=flag_key in EXPERIMENT_FLAGS,
        )
        result = (value, evaluated and resolved)
    _record_flag_for_sentry(flag_key, *result)
    return result


def _record_flag_for_sentry(flag_key: str, value: Any, evaluated: bool) -> None:
    """Put a served flag on Sentry's scope so errors show which flags were on,
    and which of them were a stand-in rather than PostHog's answer."""
    try:
        # Sentry's flag context holds booleans only; JSON and string flags are skipped.
        if not isinstance(value, bool):
            return
        sentry_sdk.feature_flags.add_feature_flag(flag_key, value)
        # A pseudo-flag on the scope's buffer only: a span keeps 10 flags and then
        # ignores every write, so there it would take a value's slot and never clear.
        flags = sentry_sdk.get_isolation_scope().flags
        marker = f"{flag_key}.fallback"
        if not evaluated:
            flags.set(marker, True)
        elif any(f["flag"] == marker for f in flags.get()):
            flags.set(marker, False)
    except Exception:
        logger.debug(f"Could not record flag {flag_key} for Sentry", exc_info=True)


# The person properties a read may send; anything else a caller layers on is dropped.
_PERSON_PROPERTY_KEYS = ("role", "email", "email_domain", "created_at", "country")


def _person_properties(context: _UserContext) -> dict[str, Any]:
    """PostHog person properties for one read: the user's, plus ``environment``.

    ``environment`` rides on every read, anonymous ones included, so a release
    condition can tell deployments apart. It is the app env the analytics events
    also carry.
    """
    return {
        **_context_properties(context),
        "environment": settings.config.app_env.value,
    }


def _context_properties(context: _UserContext) -> dict[str, Any]:
    """PostHog person properties from the user context.

    ``email`` is sent because the email rules and the ``ld:employee`` cohort
    match on it, and a rule on a property the read omits cannot be evaluated
    in-process. These ride on the read only and are never stored on the
    person profile. ``is_anonymous`` is true only for the shared
    ``ANONYMOUS_USER`` key, so a rollout can leave out the visitors nobody
    can measure.
    """
    if context.anonymous:
        return {"is_anonymous": context.key == ANONYMOUS_USER}
    return {
        attribute: context.attributes[attribute]
        for attribute in _PERSON_PROPERTY_KEYS
        if context.attributes.get(attribute) is not None
    } | {"is_anonymous": False}


_TRUTHY = ("1", "true", "yes", "on")

# Flags whose callers read a string / JSON value through
# ``get_feature_flag_value`` rather than a bool. The master switch below skips
# them so it never hands a bare ``True`` to a caller expecting a payload
# (mirrors the frontend's ``ARRAY_TYPED_FLAGS``). ``get_feature_flag_value``
# itself never consults the env override; this set only matters on the boolean
# paths (``evaluate_feature_flag``, ``feature_flag``,
# ``create_feature_flag_dependency``).
_NON_BOOLEAN_FLAG_VALUES: frozenset[str] = frozenset(
    {
        Flag.STRIPE_PRODUCT_ID_TOPUP.value,
        Flag.COPILOT_MODEL_ROUTING.value,
        Flag.COPILOT_TIER_MULTIPLIERS.value,
        Flag.COPILOT_COST_LIMITS.value,
        Flag.COPILOT_TIER_WORKSPACE_STORAGE_LIMITS.value,
        Flag.COPILOT_TIER_STRIPE_PRICES.value,
        Flag.CARD_REQUIRED_TRIAL_OFFER.value,
    }
)

# Log the master switch's state once per process, not once per evaluation.
_force_all_logged = False


def _force_all_flags_enabled() -> bool:
    """Master local-dev switch to turn every boolean flag on at once.

    Set ``FORCE_ALL_FLAGS=true`` (or the ``NEXT_PUBLIC_FORCE_ALL_FLAGS`` the
    frontend reads, so one shared var flips both sides) to force every boolean
    flag on without listing them. A per-flag ``FORCE_FLAG_<NAME>`` still wins,
    so a single flag can be excluded with ``=false`` while the rest stay on.
    Defaults off. Intended for local dev, where PostHog is unconfigured
    and every flag is otherwise off.

    Ignored (with an error log) unless ``app_env`` is local: one env var must
    not open every fail-closed gate for every user at once, and ``dev`` is a
    real, publicly reachable deployment rather than a developer's machine.
    That also rules out the single-container image, whose entrypoint exports
    ``APP_ENV=dev``; per-flag ``FORCE_FLAG_<NAME>`` remains the escape hatch
    there, since those overrides are unaffected by this guard.
    """
    global _force_all_logged
    switched_on = False
    for name in ("FORCE_ALL_FLAGS", "NEXT_PUBLIC_FORCE_ALL_FLAGS"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip().lower() in _TRUTHY:
            switched_on = True
            break
    if not switched_on:
        return False
    if settings.config.app_env != AppEnvironment.LOCAL:
        if not _force_all_logged:
            logger.error(
                "FORCE_ALL_FLAGS is set but app_env is "
                f"{settings.config.app_env.value}, not local; ignoring it. "
                "The master switch is for local dev only."
            )
            _force_all_logged = True
        return False
    if not _force_all_logged:
        logger.warning(
            "FORCE_ALL_FLAGS is on: every boolean feature flag is forced on "
            "(per-flag FORCE_FLAG_<NAME>=false still wins)."
        )
        _force_all_logged = True
    return True


def _env_flag_override(flag_key: Flag | str) -> bool | None:
    """Return a local override for ``flag_key`` from the environment.

    Set ``FORCE_FLAG_<NAME>=true|false`` (``NAME`` = flag value with
    ``-`` → ``_``, upper-cased) to bypass PostHog for a single
    flag in local dev or tests.  Returns ``None`` when no override is
    configured so the caller falls through to PostHog.

    The ``NEXT_PUBLIC_FORCE_FLAG_<NAME>`` prefix is also accepted so a
    single shared env var can toggle a flag across backend and
    frontend (the frontend requires the ``NEXT_PUBLIC_`` prefix to
    expose the value to the browser bundle).

    When no per-flag override is set, the ``FORCE_ALL_FLAGS`` master switch
    (see :func:`_force_all_flags_enabled`) forces every boolean flag on;
    non-boolean flags are left to PostHog.

    Example: ``FORCE_FLAG_CHAT_MODE_OPTION=true`` forces
    ``Flag.CHAT_MODE_OPTION`` on regardless of PostHog's answer.

    Accepts a raw flag key string as well as a :class:`Flag`, so the
    ``feature_flag`` decorator (which holds a raw key) shares this path.
    """
    key_value = flag_key.value if isinstance(flag_key, Flag) else flag_key
    suffix = key_value.upper().replace("-", "_")
    for prefix in ("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_"):
        raw = os.environ.get(prefix + suffix)
        if raw is not None:
            return raw.strip().lower() in _TRUTHY
    if _force_all_flags_enabled() and key_value not in _NON_BOOLEAN_FLAG_VALUES:
        return True
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
        default: Default value if PostHog is unavailable or evaluation fails

    Returns:
        True if feature is enabled, False otherwise
    """
    enabled, _ = await evaluate_feature_flag(flag_key, user_id, default)
    return enabled


async def evaluate_feature_flag(
    flag_key: Flag,
    user_id: str,
    default: bool = False,
) -> tuple[bool, bool]:
    """``(enabled, authoritative)`` for one flag read.

    ``authoritative`` is False when *enabled* is only the default, because the
    flag could not be evaluated or came back as a non-boolean. Use this rather
    than :func:`is_feature_enabled` wherever "off" triggers something
    irreversible — a failed read is indistinguishable from a real "off" on the
    value alone.
    """
    return await _evaluate_boolean_flag(flag_key.value, user_id, default)


async def _evaluate_boolean_flag(
    flag_key: str,
    user_id: str,
    default: bool = False,
) -> tuple[bool, bool]:
    override = _env_flag_override(flag_key)
    if override is not None:
        logger.debug(f"Feature flag {flag_key} overridden by env: {override}")
        return override, True

    result, evaluated = await _evaluate_flag_value(flag_key, user_id, default)

    # If the result is already a boolean, return it
    if isinstance(result, bool):
        return result, evaluated

    # Log a warning if the flag is not returning a boolean
    logger.warning(
        f"Feature flag {flag_key} returned non-boolean value: {result} (type: {type(result).__name__}). "
        f"This flag should be configured as a boolean in PostHog. Using default={default}"
    )

    # A misconfigured flag is not an answer either: fall back to the default,
    # but never let a caller take an irreversible action on it.
    _record_flag_for_sentry(flag_key, default, False)
    return default, False


async def get_flag_variant(flag_key: Flag, user_id: str, default: Any = False) -> Any:
    """For a flag served as true/false or as a variant key: its env override when
    one is set, as ``is_feature_enabled`` honours it, else PostHog's answer."""
    override = _env_flag_override(flag_key)
    if override is not None:
        return override
    value, _ = await _evaluate_flag_value(flag_key.value, user_id, default)
    return value


def feature_flag(
    flag_key: str,
    default: bool = False,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async feature flag protected endpoints.

    Args:
        flag_key: The feature flag key
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

                # A local env override (per-flag FORCE_FLAG_*, or the
                # FORCE_ALL_FLAGS master switch) wins over PostHog, and applies
                # even when PostHog cannot answer — the normal local-dev
                # state, where this would otherwise 404.
                override = _env_flag_override(flag_key)
                if override is not None:
                    logger.debug(
                        f"Feature flag {flag_key} overridden by env: {override}"
                    )
                    is_enabled = override
                elif not _flag_backend_initialized():
                    logger.warning(
                        "Feature flag backend not initialized, "
                        f"using default {flag_key}={repr(default)}"
                    )
                    is_enabled = default
                    _record_flag_for_sentry(flag_key, default, False)
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
                        _record_flag_for_sentry(flag_key, default, False)

                if not is_enabled:
                    raise HTTPException(status_code=404, detail="Feature not available")

                return await func(*args, **kwargs)
            except HTTPException:
                # A disabled flag is an expected outcome, not an evaluation
                # error: logging it here would file an ERROR for every request
                # to a gated-off route. The status is already correct.
                raise
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
    (if present) for proper user targeting, while still supporting
    anonymous access.

    Args:
        flag_key: The Flag enum value to check
        default: Default value if flag evaluation fails

    Returns:
        An async dependency function that raises HTTPException if flag is disabled

    Example:
        router = APIRouter(
            dependencies=[Depends(create_feature_flag_dependency(Flag.COPILOT_VOICE_MODE))]
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
        check_user_id = user_id or ANONYMOUS_USER

        # A local env override (per-flag FORCE_FLAG_*, or the FORCE_ALL_FLAGS
        # master switch) wins over PostHog, and applies even when PostHog is
        # not configured — the normal local-dev state, where this dependency
        # would otherwise 404.
        override = _env_flag_override(flag_key)
        if override is not None:
            logger.debug(f"Feature flag {flag_key.value} overridden by env: {override}")
            if not override:
                raise HTTPException(status_code=404, detail="Feature not available")
            return

        if not _flag_backend_configured():
            logger.debug(
                "Feature flag backend not configured, using default "
                f"{flag_key.value}={default}"
            )
            _record_flag_for_sentry(flag_key.value, default, False)
            if not default:
                raise HTTPException(status_code=404, detail="Feature not available")
            return

        try:
            if not _flag_backend_initialized():
                logger.debug(
                    "Feature flag backend not initialized, using default "
                    f"{flag_key.value}={default}"
                )
                _record_flag_for_sentry(flag_key.value, default, False)
                if not default:
                    raise HTTPException(status_code=404, detail="Feature not available")
                return

            is_enabled = await is_feature_enabled(flag_key, check_user_id, default)

            if not is_enabled:
                raise HTTPException(status_code=404, detail="Feature not available")
        except HTTPException:
            # A disabled flag is an answer, not a failure: the 404s raised
            # above must not be rewritten as a 500 by the handler below.
            raise
        except Exception as e:
            logger.warning(
                f"Feature flag error for {flag_key.value}: {e}, using default={default}"
            )
            raise HTTPException(status_code=500, detail="Failed to check feature flag")

    return check_feature_flag


def _flag_backend_configured() -> bool:
    """Whether PostHog has the credentials a read needs."""
    return posthog.is_configured()


def _flag_backend_gave_up() -> bool:
    """Whether the flag client has stopped trying to start, as on a rejected key.

    PostHog keeps retrying, so it never gives up for good.
    """
    return False


def _flag_backend_initialized() -> bool:
    """Whether PostHog can answer right now."""
    return posthog.get_flag_client() is not None
