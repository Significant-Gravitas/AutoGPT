import asyncio
import contextlib
import hashlib
import json
import logging
import os
import uuid
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import ldclient
import sentry_sdk.feature_flags
from autogpt_libs.auth.dependencies import get_optional_user_id
from fastapi import HTTPException, Security
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import ParamSpec

from backend.util.cache import cached
from backend.util.settings import AppEnvironment, FeatureFlagBackend, Settings

from . import posthog

logger = logging.getLogger(__name__)
# Own logger so a dual-run diff week can be queried without trawling the rest.
mismatch_logger = logging.getLogger(f"{__name__}.mismatch")

# Load settings at module level
settings = Settings()

P = ParamSpec("P")
T = TypeVar("T")

_is_initialized = False
_init_attempted = False

# Strong references to in-flight shadow evaluations; without them asyncio only
# holds a weak one and the task can be collected mid-flight.
_shadow_evaluations: set[asyncio.Task] = set()
_shadow_evaluations_stopped = False
MAX_CONCURRENT_SHADOW_EVALUATIONS = 100
# Flags PostHog could not answer for, already reported once. Before phase 2
# creates the flags that is every read; one record per flag keeps the signal.
_unanswered_flags_reported: set[str] = set()


class Flag(str, Enum):
    """
    Centralized enum for all feature flags.

    Add new flags here to ensure consistency across the codebase.
    """

    AUTOMOD = "AutoMod"
    AI_ACTIVITY_STATUS = "ai-agent-execution-summary"
    ENABLE_PLATFORM_PAYMENT = "enable-platform-payment"
    CHAT_MODE_OPTION = "chat-mode-option"
    COPILOT_SDK = "copilot-sdk"
    COPILOT_COST_LIMITS = "copilot-cost-limits"
    # AutoPilot auto mode: the permission gate in ``copilot/gate``.
    # A ROLLOUT control, not a safety control — off means today's
    # ungated behaviour, which is why it defaults False rather than
    # fail-closed. Failures *inside* an enabled gate fail to ASK.
    COPILOT_AUTO_MODE = "copilot-auto-mode"
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
    CARD_REQUIRED_TRIAL_OFFER = "card-required-trial-offer"
    GRAPHITI_MEMORY = "graphiti-memory"

    # Gates Otto voice mode end-to-end. The speech endpoint 404s when
    # off so a stale frontend cannot spend TTS budget. Fail-closed.
    COPILOT_VOICE_MODE = "copilot-voice-mode"

    # Gates the onboarding voice "brain dump" end-to-end.  The upload /
    # finalize / status / download endpoints 404 when off so a stale
    # frontend can't start writing recordings, and the wizard keeps
    # rendering the pillbox step.  Fail-closed (default False).
    ONBOARDING_BRAIN_DUMP = "onboarding-brain-dump"

    # Gates the per-user weekly community rebuild registered by
    # ``add_community_rebuild_schedule``. Off by default; opt-in canary
    # so the Leiden + LLM-summarization cost doesn't ramp before
    # retrieval-relevance benefit is measured.
    GRAPHITI_COMMUNITIES_ENABLED = "graphiti-communities-enabled"

    # Parks expert work for the user's approval once her credit spend in the
    # window reaches the approval threshold (SECRT-2599). Off by default.
    EXPERT_SPEND_APPROVAL = "expert-spend-approval"

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

    # Mirror of the frontend `hire-experts` flag (same LaunchDarkly key,
    # so both sides evaluate one cohort switch). Gates the experts
    # surface, including the scheduled morning-briefing job end-to-end:
    # schedule registration, generation, and delivery. Deliberately no
    # independent briefing kill switch — briefings ship to exactly the
    # experts cohort.
    HIRE_EXPERTS = "hire-experts"

    # Child of ``HIRE_EXPERTS``: onboarding ends with a team. The brain
    # dump also produces expert recommendations, the copilot greeting
    # grows a team section, and Otto's empty-roster context tells
    # it that it is the Head of AI. Effective only when both are on;
    # fail-closed (default False).
    ONBOARDING_EXPERT_TEAM = "onboarding-expert-team"

    # Child of ``HIRE_EXPERTS``: the role-first harness. Otto is prompted
    # as the head of the user's team and an expert session as one hired
    # employee, rather than both getting one undifferentiated prompt.
    # Effective only when both are on; fail-closed (default False). With it
    # off every session is prompted as Otto is, so the system prompt is the
    # one that shipped before the split — the flag leaves nothing behind.
    EXPERT_TASK_MANAGEMENT = "expert-task-management"

    # Mirror of the frontend `skills-hub` flag. Gates marketplace skill
    # browse and install end-to-end: the routes 404 when off, so the dark
    # launch is not reachable by URL with the shelf hidden. Fail-closed.
    SKILLS_HUB = "skills-hub"

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

    # Shows a connection the user's plan does not include as a locked entry
    # in the connection list, rather than omitting it.  Merchandising, not
    # access: the entitlement still decides what may actually run, and a
    # locked offer is never routable.  Off by default so the upsell reaches
    # a cohort before it reaches everyone.
    CHAT_CONNECTION_UPSELL = "chat-connection-upsell"

    # Shrinks what Otto reads: strips builder-UI annotations from the
    # block schemas, and digests oversized tool results to the workspace.


# LaunchDarkly keys whose targeting is segment membership or an individual-user
# target — neither of which any PostHog cohort reproduces until phase 2 creates
# one. From ld-targeting-summary.md; attribute rules (email domain, role,
# signup date) are not listed because `_person_properties` already carries them.
LD_UNPORTED_TARGETING = frozenset(
    {
        "ai-agent-execution-summary",
        "artifacts",
        "artifacts-page",
        "autogpt-new-layout",
        "AutoMod",
        "beta-blocks",
        "chat",
        "chat-mode-option",
        "chat-search",
        "chat-sharing",
        "copilot-bot-platforms",
        "dream-pass-enabled",
        "enable-platform-payment",
        "generic-trigger-agents",
        "graphiti-memory",
        "new-tool-ui",
        "nightly-copilot",
        "SHOW_ORG_SETTINGS",
        "task-progress-bar",
    }
)


def initialize_feature_flags() -> None:
    """Start whichever vendor(s) the configured backend reads from."""
    global _shadow_evaluations_stopped
    _shadow_evaluations_stopped = False
    _unanswered_flags_reported.clear()
    backend = settings.config.feature_flag_backend
    if backend is not FeatureFlagBackend.POSTHOG:
        initialize_launchdarkly()
    if backend is not FeatureFlagBackend.LAUNCHDARKLY:
        if not settings.secrets.posthog_personal_api_key:
            logger.warning(
                f"Feature flag backend is {backend.value} without "
                "POSTHOG_PERSONAL_API_KEY: every flag read becomes a remote "
                "/flags call instead of an in-process evaluation"
            )
        posthog.initialize_posthog_flags()


def shutdown_feature_flags() -> None:
    """Reverse of :func:`initialize_feature_flags`."""
    _stop_shadow_evaluations()
    backend = settings.config.feature_flag_backend
    try:
        if backend is not FeatureFlagBackend.POSTHOG:
            shutdown_launchdarkly()
    finally:
        # `ldclient.close()` can raise; skipping PostHog teardown would leak its
        # poller and latch `_init_attempted` against an in-process restart.
        if backend is not FeatureFlagBackend.LAUNCHDARKLY:
            posthog.shutdown_posthog_flags()


def _stop_shadow_evaluations() -> None:
    """Cancel in-flight shadow reads and refuse new ones until the next init.

    A shadow read that resumed after teardown would call ``get_flag_client()``
    again and, since shutdown clears the "did we try" gate so an in-process
    restart works, build a fresh PostHog client with a poller thread nothing
    closes. This is the same hazard ``shutdown_launchdarkly`` documents.
    """
    global _shadow_evaluations_stopped
    _shadow_evaluations_stopped = True
    for task in list(_shadow_evaluations):
        task.cancel()
    _shadow_evaluations.clear()


def serves_launchdarkly() -> bool:
    """Whether LaunchDarkly answers flag reads under the configured backend."""
    return settings.config.feature_flag_backend is not FeatureFlagBackend.POSTHOG


def is_configured() -> bool:
    """Check if LaunchDarkly is configured with an SDK key."""
    return bool(settings.secrets.launch_darkly_sdk_key)


def get_client() -> LDClient:
    """Get the LaunchDarkly client singleton."""
    # Gate on "did we try" rather than "did it work". Without a key
    # `_is_initialized` never becomes True, so gating on it re-entered
    # `initialize_launchdarkly` on every flag evaluation -- a warning and a
    # raise per call on the unconfigured deployments this appliance ships as.
    if not _init_attempted:
        initialize_launchdarkly()
    return ldclient.get()


def initialize_launchdarkly() -> None:
    global _init_attempted
    _init_attempted = True

    sdk_key = settings.secrets.launch_darkly_sdk_key
    logger.debug(
        f"Initializing LaunchDarkly with SDK key: {'present' if sdk_key else 'missing'}"
    )

    if not sdk_key:
        logger.warning("LaunchDarkly SDK key not configured")
        return

    config = Config(sdk_key)
    ldclient.set_config(config)

    # Read the client before recording that one exists: if constructing it
    # raised, `shutdown_launchdarkly` would otherwise build a fresh one purely
    # to close it. Being unreachable is not a construction failure -- that
    # returns an uninitialized client rather than raising -- so this only
    # covers the abnormal case.
    global _is_initialized
    connected = ldclient.get().is_initialized()
    _is_initialized = True
    if connected:
        logger.info("LaunchDarkly client initialized successfully")
    else:
        logger.error("LaunchDarkly client failed to initialize")


def shutdown_launchdarkly() -> None:
    """Shutdown the LaunchDarkly client."""
    if not _is_initialized:
        # `initialize_launchdarkly` returns early when no SDK key is configured,
        # so `ldclient.set_config` was never called and `ldclient.get()` would
        # raise "set_config was not called". Callers pair init/shutdown on
        # app_env alone (see `rest_api.feature_flag_context` and
        # `scheduler._shutdown_feature_flags_for_scheduler`), so an unconfigured
        # non-LOCAL deployment would otherwise raise out of service teardown and
        # leave the process alive instead of exiting.
        return

    # Close whenever a client was constructed, not just when it connected, so
    # buffered events are flushed and sockets are torn down in order. This is
    # not about keeping the process alive: every SDK thread is a daemon, so
    # they never blocked interpreter exit -- it was the escaping exception that
    # did. Note `close()` flushes synchronously, so an unreachable
    # LaunchDarkly can make it wait on the SDK's HTTP timeouts.
    #
    # `_is_initialized` is deliberately left set: `get_client` reads a false
    # value as "never started", and clearing it here would let a flag
    # evaluation arriving during shutdown -- an in-flight request served through
    # FastAPI's lifespan teardown -- rebuild the client we just closed.
    ldclient.get().close()
    logger.info("LaunchDarkly client closed successfully")


async def _fetch_user_context_data(user_id: str) -> Context:
    """
    Fetch user context for LaunchDarkly from the auth user table.

    Successful lookups are cached for 24h (see ``_fetch_user_context``).
    Failed lookups are NOT cached: the degraded anonymous fallback is built
    outside the cache so the next evaluation retries the lookup instead of
    pinning this process to an email-less context for a full TTL — which
    would make its email/role-targeted flag evaluations silently diverge
    from peer processes.  The degraded path costs one failed lookup per
    evaluation; bounded, and acceptable versus a 24h-poisoned cache.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        LaunchDarkly Context object
    """
    context, _ = await _fetch_user_context_status(user_id)
    return context


async def _fetch_user_context_status(user_id: str) -> tuple[Context, bool]:
    """``(context, resolved)`` — see :func:`_fetch_user_context_data`.

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


def _with_request_attributes(context: Context, attributes: dict[str, str]) -> Context:
    """*context* plus facts known only for this request, such as the country.

    The user context is cached for a day, so anything that can change between
    requests has to be layered on per evaluation rather than baked into it.
    A copy is returned; the cached context is never modified. ``key`` and
    ``kind`` are identity and cannot be overridden this way.
    """
    merged = {**context.to_dict(), **attributes}
    merged.update(key=context.key, kind=context.kind)
    return Context.from_dict(merged)


def _anonymous_context(user_id: str) -> Context:
    """Build a minimal anonymous LD context carrying only the user key."""
    return Context.builder(user_id).kind("user").anonymous(True).build()


@cached(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_user_context(user_id: str) -> Context:
    """
    Build the full LaunchDarkly context for ``user_id`` from the auth user
    table.

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

    builder = Context.builder(user_id).kind("user").anonymous(False)
    # Keep the same role values previously issued in JWTs so existing
    # LaunchDarkly targeting rules keep matching.
    role = "admin" if fields.role == "admin" else "authenticated"
    builder.set("role", role)
    # It's weird, I know, but it is what it is.
    builder.set("custom", {"role": role})
    if fields.email:
        builder.set("email", fields.email)
        builder.set("email_domain", fields.email.split("@")[-1])
    if fields.created_at:
        # ISO-8601 string — LD supports RFC3339 date targeting on
        # this attribute (e.g. cohort users by signup window).
        builder.set("created_at", fields.created_at.isoformat())

    return builder.build()


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
        default: Default value if the vendor is unavailable or evaluation fails

    Returns:
        The flag value from the configured backend
    """
    value, _ = await _evaluate_flag_value(
        flag_key, user_id, default, attributes=attributes
    )
    return value


async def _evaluate_flag_value(
    flag_key: str,
    user_id: str,
    default: Any = None,
    *,
    attributes: dict[str, str] | None = None,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` for one raw flag read, from the configured vendor.

    ``evaluated`` is False whenever *default* is standing in for an answer the
    vendor could not give.
    """
    context = None
    if attributes:
        # Layered once, so every vendor evaluates the same per-request facts.
        user_context, context_resolved = await _fetch_user_context_status(user_id)
        context = (_with_request_attributes(user_context, attributes), context_resolved)
    backend = settings.config.feature_flag_backend
    if backend is FeatureFlagBackend.POSTHOG:
        result = await _evaluate_posthog(flag_key, user_id, default, context)
    elif backend is FeatureFlagBackend.DUAL:
        result = await _evaluate_dual(flag_key, user_id, default, context)
    else:
        result = await _evaluate_launchdarkly(flag_key, user_id, default, context)
    _record_flag_for_sentry(flag_key, *result)
    return result


def _record_flag_for_sentry(flag_key: str, value: Any, evaluated: bool) -> None:
    """Put a served flag on Sentry's scope so errors show which flags were on,
    and which of them were a stand-in rather than the vendor's answer."""
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


async def _evaluate_dual(
    flag_key: str,
    user_id: str,
    default: Any = None,
    context: tuple[Context, bool] | None = None,
) -> tuple[Any, bool]:
    """Evaluate both vendors, serve LaunchDarkly's answer, log disagreements.

    Serving LaunchDarkly is what makes the diff week free of user-visible
    risk: PostHog's answer is only ever observed, never acted on.
    """
    # One lookup for both vendors: the failure path is deliberately uncached,
    # so evaluating them independently would double its database reads.
    context = context or await _fetch_user_context_status(user_id)
    ld_result = await _evaluate_launchdarkly(flag_key, user_id, default, context)
    _probe_posthog(flag_key, user_id, default, context, ld_result)
    return ld_result


def _probe_posthog(
    flag_key: str,
    user_id: str,
    default: Any,
    context: tuple[Context, bool],
    ld_result: tuple[Any, bool],
) -> None:
    """Compare PostHog's answer to LaunchDarkly's without making a caller wait.

    The shadow answer is never served, so awaiting it would only add PostHog's
    latency to the request path — up to a 3s remote ``/flags`` call wherever
    ``POSTHOG_PERSONAL_API_KEY`` is unset and local evaluation is off.
    """
    if _shadow_evaluations_stopped:
        return

    if len(_shadow_evaluations) >= MAX_CONCURRENT_SHADOW_EVALUATIONS:
        # A slow PostHog must cost the diff week its samples, not the process
        # its memory.
        logger.debug(f"Shadow evaluation backlog full, skipping {flag_key}")
        return

    task = asyncio.create_task(
        _record_mismatch(flag_key, user_id, default, context, ld_result)
    )
    _shadow_evaluations.add(task)
    task.add_done_callback(_shadow_evaluations.discard)


async def _record_mismatch(
    flag_key: str,
    user_id: str,
    default: Any,
    context: tuple[Context, bool],
    ld_result: tuple[Any, bool],
) -> None:
    try:
        ph_result = await _evaluate_posthog(flag_key, user_id, default, context)
    except Exception as e:
        logger.warning(f"PostHog shadow evaluation raised for {flag_key}: {e}")
        return

    if ld_result == ph_result:
        return

    ld_value, ld_evaluated = ld_result
    ph_value, ph_evaluated = ph_result
    if not ph_evaluated:
        if flag_key in _unanswered_flags_reported:
            return
        _unanswered_flags_reported.add(flag_key)
    mismatch_logger.warning(
        "feature-flag mismatch: "
        + json.dumps(
            {
                "flag": flag_key,
                "user": _user_digest(user_id),
                "launchdarkly": {"value": ld_value, "evaluated": ld_evaluated},
                "posthog": {"value": ph_value, "evaluated": ph_evaluated},
                # Segment and individual-user targeting has no PostHog cohort
                # until phase 2 builds one, so these flags are expected to
                # disagree — the diff-week report has to separate them from
                # real divergence rather than drown in them.
                "expected_until_cohorts_exist": flag_key in LD_UNPORTED_TARGETING,
            },
            default=repr,
            sort_keys=True,
        )
    )


async def _evaluate_posthog(
    flag_key: str,
    user_id: str,
    default: Any = None,
    context: tuple[Context, bool] | None = None,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` from PostHog for one raw flag read.

    A degraded context lookup makes the answer unauthoritative for the same
    reason it does on LaunchDarkly: the flag evaluates fine, it just answers
    for a user without the targeted attributes.
    """
    user_context, context_resolved = context or await _fetch_user_context_status(
        user_id
    )
    value, evaluated = await posthog.evaluate_flag(
        flag_key,
        user_id,
        _person_properties(user_context),
        default,
    )
    return value, evaluated and context_resolved


async def _evaluate_launchdarkly(
    flag_key: str,
    user_id: str,
    default: Any = None,
    context: tuple[Context, bool] | None = None,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` from LaunchDarkly for one raw flag read.

    ``evaluated`` is False whenever *default* is standing in for an answer
    LaunchDarkly could not give — no client, an uninitialised one, a failed
    user-context lookup, or an evaluation that raised. An initialised client
    is not on its own enough: the context lookup is a database read, so a
    live client can still fail to produce a value.
    """
    try:
        client = get_client()

        # Check if client is initialized
        if not client.is_initialized():
            logger.debug(
                f"LaunchDarkly not initialized, using default={default} for {flag_key}"
            )
            return default, False

        # Get user context (role/email) from the Better Auth user table
        user_context, context_resolved = context or await _fetch_user_context_status(
            user_id
        )

        # Evaluate flag
        result = client.variation(flag_key, user_context, default)

        logger.debug(
            f"Feature flag {flag_key} for user {user_id}: {result} (type: {type(result).__name__})"
        )
        # A degraded context evaluates fine, it just answers for an anonymous
        # user rather than this one — so the value is a guess, not an answer.
        return result, context_resolved

    except Exception as e:
        logger.warning(
            f"LaunchDarkly flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default, False


def _person_properties(context: Context) -> dict[str, Any]:
    """PostHog person properties from the shared user context.

    Reads the LaunchDarkly context rather than the auth row so both vendors
    see one cached lookup and cannot drift; ``custom.role`` is dropped
    because PostHog targets flat properties, and ``email`` because no ported
    rule reads it — individual targets key on the ``distinct_id``.
    """
    if context.anonymous:
        return {}
    return {
        attribute: context.get(attribute)
        for attribute in ("role", "email_domain", "created_at", "country")
        if context.get(attribute) is not None
    }


def _user_digest(user_id: str) -> str:
    """Short stable hash — a mismatch record must be joinable, not identifying."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]


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
    Defaults off. Intended for local dev, where LaunchDarkly is unconfigured
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
    ``-`` → ``_``, upper-cased) to bypass the flag vendor for a single
    flag in local dev or tests.  Returns ``None`` when no override is
    configured so the caller falls through to the configured backend.

    The ``NEXT_PUBLIC_FORCE_FLAG_<NAME>`` prefix is also accepted so a
    single shared env var can toggle a flag across backend and
    frontend (the frontend requires the ``NEXT_PUBLIC_`` prefix to
    expose the value to the browser bundle).

    When no per-flag override is set, the ``FORCE_ALL_FLAGS`` master switch
    (see :func:`_force_all_flags_enabled`) forces every boolean flag on;
    non-boolean flags are left to the configured backend.

    Example: ``FORCE_FLAG_CHAT_MODE_OPTION=true`` forces
    ``Flag.CHAT_MODE_OPTION`` on regardless of the vendor's answer.

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
        default: Default value if the vendor is unavailable or evaluation fails

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
    override = _env_flag_override(flag_key)
    if override is not None:
        logger.debug(f"Feature flag {flag_key} overridden by env: {override}")
        return override, True

    result, evaluated = await _evaluate_flag_value(flag_key.value, user_id, default)

    # If the result is already a boolean, return it
    if isinstance(result, bool):
        return result, evaluated

    # Log a warning if the flag is not returning a boolean
    logger.warning(
        f"Feature flag {flag_key} returned non-boolean value: {result} (type: {type(result).__name__}). "
        f"This flag should be configured as a boolean in LaunchDarkly. Using default={default}"
    )

    # A misconfigured flag is not an answer either: fall back to the default,
    # but never let a caller take an irreversible action on it.
    _record_flag_for_sentry(flag_key.value, default, False)
    return default, False


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
                # FORCE_ALL_FLAGS master switch) wins over whichever vendor is
                # configured, and applies even when it cannot answer — the
                # normal local-dev state, where this would otherwise 404.
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
            dependencies=[Depends(create_feature_flag_dependency(Flag.SKILLS_HUB))]
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

        # A local env override (per-flag FORCE_FLAG_*, or the FORCE_ALL_FLAGS
        # master switch) wins over whichever vendor is configured, and applies
        # even when none is — the normal local-dev state, where this dependency
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
    """Whether the configured backend has the credentials a read needs."""
    # Dual serves LaunchDarkly's answer, so LaunchDarkly is what gates a route.
    if serves_launchdarkly():
        return is_configured()
    return posthog.is_configured()


def _flag_backend_initialized() -> bool:
    """Whether the configured backend can answer right now."""
    if serves_launchdarkly():
        return get_client().is_initialized()
    return posthog.get_flag_client() is not None


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
