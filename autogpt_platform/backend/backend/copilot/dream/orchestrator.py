"""Three-phase dream-pass orchestrator (sync_baseline path).

Walks a user's recent memory window through the consolidate → recombine
→ sanitize pipeline, then applies the sanitizer's ``DreamOperations``
to Graphiti + Postgres.

Slice 1 of P-0 deliberately bypasses Anthropic batch — every phase
calls the OpenRouter-fronted OpenAI-compat client. The batch path
slots in below this layer (`routing.py` returns ``"batch"``) in a
future PR.

The orchestrator never raises out — every failure becomes a
``DreamPassResult`` with ``error`` set, so the admin trigger always
gets a structured response back.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid as uuidlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TypeVar

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, is_feature_enabled

from .apply import apply_operations, drain_status_from_stats
from .billing import check_dream_budget, record_phase_cost
from .fetch import (
    DreamInput,
    EpisodeRow,
    gather_dream_input,
    is_dream_authored_episode,
    parse_episode_timestamp,
)
from .llm import (
    CompletionUsage,
    DreamLLMError,
    StructuredCompletion,
    structured_completion,
)
from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    LOCAL_LOCK_TTL_SECONDS,
    DreamLockHandle,
    DreamLockHeld,
    dream_lock,
)
from .model_pricing import compute_cost_usd, execution_path_discount
from .prompts import (
    MAX_DEMOTIONS_PER_PASS,
    MAX_PROPOSALS_PER_PASS,
    MAX_WRITES_PER_PASS,
    build_consolidate_prompt,
    build_recombine_prompt,
    build_sanitize_prompt,
)
from .routing import ExecutionPath, resolve_dream_execution_path
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassResult,
    DreamPassUsage,
    PhaseUsage,
    ProposedFinding,
    RecombinationOutput,
)

logger = logging.getLogger(__name__)


# Per-step temperature + max output token budgets. Spec ref:
# ``dream/p0-spec.md`` §2. Named by what the step does rather than its
# position; the consolidation→recombination→sanitization order is
# enforced by the call graph below.
CONSOLIDATE_TEMP = 0.2
RECOMBINE_TEMP = 0.9
SANITIZE_TEMP = 0.0
CONSOLIDATE_MAX_TOKENS = 4096
# Recombine + sanitize emit list-heavy JSON (up to 30 writes + 20
# proposals + 10 demotions, each with uuid arrays). At 8192 the
# response truncates mid-array and the JSON-balanced-brace extractor
# returns None, failing the whole phase. 16384 covers worst-case +
# headroom; cost impact is bounded by the per-phase caps anyway.
RECOMBINE_MAX_TOKENS = 16384
SANITIZE_MAX_TOKENS = 16384

# Per-phase LLM wall-clock ceilings, passed through structured_completion
# → call_provider. The shared DEFAULT_REQUEST_TIMEOUT_SECONDS (120s) is
# sized for short chat completions; recombine/sanitize carry 16384-token
# output budgets precisely because real responses exceed 8192 tokens, and
# at real decode speeds (Opus-class ~40-60 tok/s via OpenRouter) an 8-16K
# response takes 150-400s — the 120s wall killed exactly the responses
# the token-cap raise above was meant to save.
#
# A conservative ~20 tok/s decode floor would ask for max_output_tokens/20
# ≈ 205s / 819s / 819s per phase, but that sums to 1843s of LLM budget
# alone — past both SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS (1800s,
# scheduler.py) and DEFAULT_LOCK_TTL_SECONDS (1800s, locks.py) before
# fetch/apply even run. So the phases get values that fit the envelope
# instead. Budget math (three explicit line items, with real slack):
#
#   consolidate 240s + recombine 600s + sanitize 480s   = 1320s LLM ceiling
#   + apply.INGESTION_DRAIN_TIMEOUT_SECONDS 300s          drain cap
#   + DREAM_NON_LLM_HEADROOM_SECONDS 120s (budget check + fetch + enqueue
#     + demotions + summary + cost logging, EXCLUDING the drain)
#   = 1740s <= 1800s SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS  (60s slack)
#
# The drain no longer counts against the lock TTL: apply renews the dream
# lock to a fresh budget right before the drain (see
# apply.LOCK_DRAIN_RENEWAL_SECONDS), so the lock only has to cover
# LLM + non-drain headroom = 1440s <= 0.9 * DEFAULT_LOCK_TTL_SECONDS (1620s)
# with genuine margin, and the drain itself runs under the renewed lock.
# Both invariants are pinned by
# test_phase_timeouts_plus_headroom_fit_scheduler_and_lock_envelope —
# bumping any value here fails that test until the budget is re-balanced.
#
# Recombine (fast_advanced_model, Opus-class — the slowest decoder) gets
# the largest share: 600s covers 16384 tokens at ~27 tok/s. Sanitize runs
# on the faster fast_standard_model: 480s covers 16384 at ~34 tok/s.
# Consolidate's 4096-token budget fits 240s at ~17 tok/s.
CONSOLIDATE_TIMEOUT_SECONDS = 240
RECOMBINE_TIMEOUT_SECONDS = 600
SANITIZE_TIMEOUT_SECONDS = 480
# Reserved for the non-LLM segments of the pass OTHER than the ingestion
# drain (budget check, fetch, enqueue, demotions, summary write, cost
# logging). The drain is budgeted separately as
# apply.INGESTION_DRAIN_TIMEOUT_SECONDS and covered by a lock renewal.
DREAM_NON_LLM_HEADROOM_SECONDS = 120

# Entity invalidation is the most destructive op the sanitizer can emit:
# each one single-hop demotes EVERY :RELATES_TO edge on the entity, so a
# hub entity multiplies the blast radius far past the demotion caps. We
# cap the *count* of invalidations per pass here; the per-entity edge
# blast radius is bounded by the ``DREAM_PASS_INVALIDATE_ENTITY`` LD flag
# (staged rollout, off by default) plus the single-hop-only guarantee in
# ``invalidate_entity_direct_neighbors`` (no multi-hop propagation).
# Deliberately NOT degree-aware — fetching entity degrees is async graph
# work that doesn't belong in this sync clamp.
MAX_ENTITY_INVALIDATIONS_PER_PASS = 2

# Per-user marker stamped after a successful (non-skipped) sync apply so
# the next nightly pass can skip all three LLM phases when no new episode
# landed since. Single key — prod Redis runs in cluster mode (see
# locks.py), so no multi-key primitives. The 35-day TTL comfortably
# outlives the 14-day episode window; an expired or missing marker just
# means one extra full pass (fail-open). The batch path does NOT stamp
# this marker yet — batch users simply never benefit from the skip.
LAST_COMPLETED_KEY_PREFIX = "dream:last_completed:"
LAST_COMPLETED_TTL_SECONDS = 35 * 24 * 60 * 60


def _resolve_lock_ttl(transport_is_local: bool) -> int:
    return LOCAL_LOCK_TTL_SECONDS if transport_is_local else DEFAULT_LOCK_TTL_SECONDS


# High-precision filter for "transient intent" facts — content that
# records what the user is ASKING/wants to KNOW rather than a durable
# fact about them. The sanitize prompt is told to drop these, but
# prompt-only sanitization leaks them (#13388: "User is asking how
# Kubernetes works", "User is interested in knowing which PRs are open"),
# so this is a deterministic belt-and-suspenders gate.
#
# Deliberately NARROW to knowledge-seeking intent. We do NOT match goals
# like "user wants to create/build X" — those are legitimate durable
# memories. Generic world-knowledge pollution ("Kubernetes uses pods…")
# is left to the sanitize prompt: it needs LLM judgment (is the subject
# the user?) that a regex can't do without false-positives.
#
# Interrogative complements are the sharp edge — several verbs are durable
# aspirations on their own but transient questions once they take a
# question word:
#   * ``asking`` — "asking FOR weekly reports" / "asking the agent TO
#     monitor PRs" are durable requests (semantically like "wants X"), so
#     only interrogative ``asking HOW/WHAT/…`` counts as transient.
#   * ``learn``/``understand``/``find out`` — "wants to learn Spanish",
#     "wants to understand distributed systems", "wants to find out about
#     new markets" are durable skill/aspiration GOALS; they only read as
#     transient with an interrogative ("wants to learn HOW X works").
#   * ``curious`` — "curious ABOUT X" is transient, but "curious by nature"
#     is a durable personality trait, so a ``curious about`` complement is
#     required (mirroring ``confused about``/``unsure about``).
# ``know`` stays complement-free: "wants to know X" is transient curiosity
# regardless of phrasing (there is no durable "wants to know" aspiration —
# that role is served by ``learn``).
# Known limitation (nice-to-have, low frequency): a standing notification
# preference phrased "wants to know when X happens" is dropped; separating
# it from a one-off "wants to know when the deploy is" isn't reliably
# regex-able, so it's left to the sanitize prompt + human review.
#
# Subject scope: the gate deliberately anchors on the generic ``user``
# subject only. Name-phrased transient intent ("Nick is asking how the
# auth flow works") is intentionally NOT matched here — broadening the
# subject to arbitrary proper nouns would risk false-positives on
# non-user entities ("Kubernetes is asking for more nodes"), so
# name-first phrasing is left to the sanitize prompt's LLM judgment. The
# leading auxiliary allows perfect-progressive forms ("has been asking").
_TRANSIENT_INTENT_RE = re.compile(
    r"^(the\s+)?user\s+(?:(?:is|has|was)\s+(?:been\s+)?)?"
    r"(asking\s+(how|what|why|whether|if|when|where|who|which|about)\b"
    r"|wondering\b"
    r"|curious\s+about\b"
    r"|confused\s+about\b"
    r"|unsure\s+about\b"
    r"|trying\s+to\s+understand\b"
    r"|interested\s+in\s+(knowing|understanding)\b"
    r"|interested\s+in\s+learning\s+(how|what|why|whether|if|when|where)\b"
    r"|wants?\s+to\s+know\b"
    r"|wants?\s+to\s+(understand|find\s+out)\s+(how|what|why|whether|if|when|where)\b"
    r"|wants?\s+to\s+learn\s+(how|what|why|whether|if|when|where)\b"
    r"|asked\s+(how|what|why|whether|if|when|where|about)\b)",
    re.IGNORECASE,
)


def _is_transient_intent(content: str) -> bool:
    """True when ``content`` reads as a question/knowledge-seeking intent
    rather than a durable fact about the user."""
    return bool(_TRANSIENT_INTENT_RE.match(content.strip()))


_ContentItem = TypeVar("_ContentItem", ConsolidatedFact, ProposedFinding)


def _drop_transient_intent(
    items: Sequence[_ContentItem],
) -> tuple[list[_ContentItem], int]:
    """Filter ConsolidatedFact / ProposedFinding items whose ``.content``
    is a transient intent. Returns (kept, dropped_count)."""
    kept = [it for it in items if not _is_transient_intent(it.content)]
    return kept, len(items) - len(kept)


# Intra-pass near-duplicate write rejection (#13387). The consolidate
# prompt asks the model to merge near-duplicates, but a single pass still
# emits the same fact phrased 2-3 ways ("Nick uses Terminus on iPhone for
# CLI work" / "Nick wants Terminus on iPhone to show more ASCII"). This is
# a CONSERVATIVE, deterministic backstop: it only collapses writes that are
# near-identical by content-word overlap, keeping the longest (most
# specific) of each cluster. It deliberately does NOT do semantic
# (embedding) merging or compare against the user's EXISTING active facts —
# both need real-data threshold tuning + edge-merge design and are tracked
# as the follow-up P2 dedup pass. High threshold so distinct facts about
# the same entity ("prefers Python" vs "prefers Rust") are never merged.
_DEDUP_JACCARD_THRESHOLD = 0.7
# A short fact whose content words are (nearly) all contained in a longer
# fact is a duplicate the longer one subsumes — even when the longer fact's
# extra detail drags Jaccard below the threshold ("Nick uses Terminus on
# iPhone for CLI" ⊂ "…for CLI and wants more ASCII"). Set high (0.9): for
# facts under ~10 words it effectively requires the shorter to be a strict
# SUBSET of the longer, so a single distinguishing word keeps them apart
# ("deployed the AUTH service" vs "deployed the BILLING service" → 4/5=0.8,
# NOT merged). Only applied when the smaller fact has enough words to be
# specific, so 1-2 word fragments don't spuriously match.
_DEDUP_CONTAINMENT_THRESHOLD = 0.9
_DEDUP_CONTAINMENT_MIN_TOKENS = 3
# Facts with IDENTICAL content-word sets can still describe different
# events when word order differs ("Alice introduced Bob to Carol" vs
# "Alice introduced Carol to Bob"). Equal token sets therefore also
# require agreement on adjacent-word bigrams before merging.
_DEDUP_BIGRAM_THRESHOLD = 0.6
# A negation marker present in one fact but not the other flips the
# meaning ("uses vim" vs "never uses vim") — never merge across it.
# The tokenizer splits contractions ("doesn't" → "doesn", "t"), so the
# bare stems cover the apostrophe forms; a false veto only ever KEEPS
# both facts, which is the conservative direction.
_DEDUP_NEGATION_MARKERS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nor",
        "cannot",
        "nt",
        "dont",
        "don",
        "doesnt",
        "doesn",
        "didnt",
        "didn",
        "isnt",
        "isn",
        "wasnt",
        "wasn",
        "werent",
        "weren",
        "wont",
        "cant",
        "hasnt",
        "hasn",
        "havent",
        "haven",
        "hadnt",
        "hadn",
        "wouldnt",
        "wouldn",
        "shouldnt",
        "shouldn",
        "couldnt",
        "couldn",
        "arent",
        "aren",
    }
)
_DEDUP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "for",
        "on",
        "in",
        "at",
        "by",
        "with",
        "that",
        "this",
        "it",
        "their",
        "they",
        "them",
        "his",
        "her",
        "its",
        "as",
        "be",
        "has",
        "have",
        "had",
        "s",
        "so",
        "more",
        "user",
        "users",
    }
)


def _content_tokens(content: str) -> frozenset[str]:
    """Lowercased alphanumeric content words, minus a small stopword set.
    ``user`` is a stopword because nearly every fact begins with it."""
    return frozenset(
        t
        for t in re.findall(r"[a-z0-9]+", content.lower())
        if t not in _DEDUP_STOPWORDS
    )


def _word_bigrams(content: str) -> frozenset[tuple[str, str]]:
    """Adjacent word pairs over the full (unfiltered) word sequence — the
    order-sensitive signal used when two facts' token sets are equal."""
    words = re.findall(r"[a-z0-9]+", content.lower())
    return frozenset(zip(words, words[1:]))


def _near_duplicate(
    a: frozenset[str],
    b: frozenset[str],
    a_bigrams: frozenset[tuple[str, str]],
    b_bigrams: frozenset[tuple[str, str]],
) -> bool:
    """Two content-word sets are near-duplicate when their Jaccard overlap
    is high, OR the smaller (sufficiently specific) set is nearly contained
    in the larger one. A negation marker on only one side always vetoes;
    identical sets additionally need word-order (bigram) agreement."""
    if not a or not b:
        return False
    if _DEDUP_NEGATION_MARKERS & (a ^ b):
        return False
    if a == b:
        bigram_union = len(a_bigrams | b_bigrams)
        if not bigram_union:
            return True
        return len(a_bigrams & b_bigrams) / bigram_union >= _DEDUP_BIGRAM_THRESHOLD
    inter = len(a & b)
    union = len(a | b)
    if union and inter / union >= _DEDUP_JACCARD_THRESHOLD:
        return True
    smaller = min(len(a), len(b))
    if smaller < _DEDUP_CONTAINMENT_MIN_TOKENS:
        return False
    return inter / smaller >= _DEDUP_CONTAINMENT_THRESHOLD


def _dedupe_near_duplicate_writes(
    writes: list[ConsolidatedFact],
) -> tuple[list[ConsolidatedFact], int]:
    """Collapse near-duplicate writes, keeping the longest of each cluster.

    Greedy by content length (desc) so the most specific phrasing survives;
    output preserves the writes' original order. Writes are only compared
    within the same ``scope`` (facts must never merge across scopes), and a
    dropped write's ``source_episode_uuids`` are unioned into its cluster's
    survivor so provenance is never lost. Returns (kept, dropped)."""
    by_len_desc = sorted(
        range(len(writes)), key=lambda i: len(writes[i].content), reverse=True
    )
    tokens = [_content_tokens(w.content) for w in writes]
    bigrams = [_word_bigrams(w.content) for w in writes]
    kept_idx: list[int] = []
    absorbed: dict[int, list[str]] = {}
    for i in by_len_desc:
        survivor = next(
            (
                k
                for k in kept_idx
                if writes[k].scope == writes[i].scope
                and _near_duplicate(tokens[i], tokens[k], bigrams[i], bigrams[k])
            ),
            None,
        )
        if survivor is None:
            kept_idx.append(i)
            absorbed[i] = []
        else:
            absorbed[survivor].extend(writes[i].source_episode_uuids)
    kept = [_absorb_provenance(writes[i], absorbed[i]) for i in sorted(kept_idx)]
    return kept, len(writes) - len(kept)


def _absorb_provenance(
    write: ConsolidatedFact, absorbed_uuids: list[str]
) -> ConsolidatedFact:
    """Union dropped near-duplicates' episode uuids into the survivor."""
    extra = [
        u for u in dict.fromkeys(absorbed_uuids) if u not in write.source_episode_uuids
    ]
    if not extra:
        return write
    return write.model_copy(
        update={"source_episode_uuids": [*write.source_episode_uuids, *extra]}
    )


def _clamp_operations(
    ops: DreamOperations,
    active_fact_count: int,
    known_fact_uuids: set[str] | None = None,
) -> DreamOperations:
    """Hard-trim oversized phase 3 outputs.

    Phase 3's prompt asks for these caps but the model can still
    over-emit. The orchestrator enforces them in code so apply.py
    never writes more than the policy allows.

    Demotions carry a second ceiling — 5% of the active fact set — so a
    single pass can never wipe a meaningful fraction of a user's memory
    even if the absolute ``MAX_DEMOTIONS_PER_PASS`` cap would allow it.
    The 5% ceiling floors at 1 when there is at least one active fact:
    small graphs (< 20 facts) would otherwise round to a cap of 0 and
    never get even a single contradicted fact demoted.
    ``active_fact_count < 0`` means the count is unknown (the batch path
    lost its persisted input bundle); fall back to the absolute cap only
    rather than silently dropping every demotion.

    When ``known_fact_uuids`` is provided, demotions targeting uuids
    outside it are dropped BEFORE the cap slice — otherwise a
    hallucinated uuid at the head of the model's list consumes a cap
    slot (the entire budget on a floor-of-1 small graph) and displaces
    a valid demotion that apply.py would have accepted. ``None`` skips
    the pre-filter; apply.py's idempotent known-uuid filter remains the
    security chokepoint either way.

    Entity invalidations are count-capped at
    ``MAX_ENTITY_INVALIDATIONS_PER_PASS``; see the constant's comment
    for why the per-entity edge blast radius is bounded elsewhere (LD
    flag + single-hop guarantee), not here.
    """
    demotions = ops.demotions
    if known_fact_uuids is not None:
        demotions = [d for d in demotions if d.edge_uuid in known_fact_uuids]
        dropped = len(ops.demotions) - len(demotions)
        if dropped:
            logger.warning(
                "Dream clamp: dropped %d demotion(s) targeting edge uuids "
                "outside known_fact_uuids before applying the demotion cap",
                dropped,
            )
    demotion_cap = MAX_DEMOTIONS_PER_PASS
    if active_fact_count == 0:
        demotion_cap = 0
    elif active_fact_count > 0:
        demotion_cap = min(MAX_DEMOTIONS_PER_PASS, max(1, active_fact_count * 5 // 100))

    # Drop transient-intent pollution ("user is asking…") before the cap
    # slice so a leaked question never displaces a real fact (#13388).
    writes, w_intent_dropped = _drop_transient_intent(ops.writes)
    proposals, p_intent_dropped = _drop_transient_intent(ops.proposals)
    if w_intent_dropped or p_intent_dropped:
        logger.info(
            "Dream clamp: dropped %d transient-intent write(s) and %d "
            "proposal(s) (questions captured as facts)",
            w_intent_dropped,
            p_intent_dropped,
        )
    # Collapse intra-pass near-duplicate writes before the cap slice so the
    # cap counts distinct facts, not paraphrases of one (#13387).
    writes, w_dup_dropped = _dedupe_near_duplicate_writes(writes)
    if w_dup_dropped:
        logger.info(
            "Dream clamp: collapsed %d near-duplicate write(s) into their "
            "canonical (longest) phrasing",
            w_dup_dropped,
        )
    return DreamOperations(
        writes=writes[:MAX_WRITES_PER_PASS],
        proposals=proposals[:MAX_PROPOSALS_PER_PASS],
        demotions=demotions[:demotion_cap],
        entity_invalidations=ops.entity_invalidations[
            :MAX_ENTITY_INVALIDATIONS_PER_PASS
        ],
        summary_for_user=ops.summary_for_user,
    )


async def _run_consolidate(
    config: ChatConfig, input_bundle: DreamInput
) -> StructuredCompletion[ConsolidationOutput]:
    """First step: merge near-duplicate recent facts into canonical statements."""
    messages = build_consolidate_prompt(input_bundle)
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=ConsolidationOutput,
        temperature=CONSOLIDATE_TEMP,
        max_output_tokens=CONSOLIDATE_MAX_TOKENS,
        timeout_seconds=CONSOLIDATE_TIMEOUT_SECONDS,
    )


async def _run_recombine(
    config: ChatConfig,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
) -> StructuredCompletion[RecombinationOutput]:
    """Second step: propose novel connections + weak-link findings."""
    messages = build_recombine_prompt(input_bundle, consolidated.model_dump_json())
    return await structured_completion(
        model=config.fast_advanced_model,
        messages=messages,
        response_model=RecombinationOutput,
        temperature=RECOMBINE_TEMP,
        max_output_tokens=RECOMBINE_MAX_TOKENS,
        timeout_seconds=RECOMBINE_TIMEOUT_SECONDS,
    )


async def _run_sanitize(
    config: ChatConfig,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
    recombined: RecombinationOutput,
) -> StructuredCompletion[DreamOperations]:
    """Third step: gate writes/proposals/demotions before apply.py runs."""
    messages = build_sanitize_prompt(
        input_bundle,
        consolidated.model_dump_json(),
        recombined.model_dump_json(),
    )
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=DreamOperations,
        temperature=SANITIZE_TEMP,
        max_output_tokens=SANITIZE_MAX_TOKENS,
        timeout_seconds=SANITIZE_TIMEOUT_SECONDS,
    )


def _phase_usage_from_completion(
    phase: str, completion_usage: CompletionUsage, execution_path: ExecutionPath
) -> PhaseUsage:
    """Build a ``PhaseUsage`` from a raw ``CompletionUsage``, computing
    cost from the rate card when the provider didn't supply one.

    Provider-supplied cost (OpenRouter ``usage.cost``) wins when
    present; otherwise we fall back to ``model_pricing.compute_cost_usd``.
    Either way the execution-path discount has been applied: OpenRouter
    spot prices are already post-discount (they ARE the rate we paid),
    and the rate-card fallback explicitly multiplies the discount in.
    """
    cost = completion_usage.cost_usd
    if cost is None:
        cost = compute_cost_usd(
            model=completion_usage.model,
            input_tokens=completion_usage.input_tokens,
            output_tokens=completion_usage.output_tokens,
            cache_read_tokens=completion_usage.cache_read_tokens,
            cache_creation_tokens=completion_usage.cache_creation_tokens,
            execution_path=execution_path,
        )
    return PhaseUsage(
        phase=phase,  # type: ignore[arg-type]
        model=completion_usage.model,
        input_tokens=completion_usage.input_tokens,
        output_tokens=completion_usage.output_tokens,
        cache_read_tokens=completion_usage.cache_read_tokens,
        cache_creation_tokens=completion_usage.cache_creation_tokens,
        cost_usd=cost,
    )


def _aggregate_usage(
    phases: list[PhaseUsage], execution_path: ExecutionPath
) -> DreamPassUsage:
    """Roll up per-phase usage into a ``DreamPassUsage``.

    ``total_cost_usd`` is None when any single phase had unknown cost
    so we never silently bill at a partial figure.
    """
    total_cost: float | None = 0.0
    for p in phases:
        if p.cost_usd is None:
            total_cost = None
            break
        total_cost += p.cost_usd
    return DreamPassUsage(
        phases=phases,
        total_input_tokens=sum(p.input_tokens for p in phases),
        total_output_tokens=sum(p.output_tokens for p in phases),
        total_cache_read_tokens=sum(p.cache_read_tokens for p in phases),
        total_cache_creation_tokens=sum(p.cache_creation_tokens for p in phases),
        total_cost_usd=total_cost,
        discount_applied=execution_path_discount(execution_path),
    )


def _last_completed_key(user_id: str) -> str:
    return f"{LAST_COMPLETED_KEY_PREFIX}{user_id}"


async def _read_last_completed_marker(user_id: str) -> datetime | None:
    """When the user's last dream pass completed, or ``None``.

    Best-effort: a Redis error or an unparseable value fails open
    (``None`` ⇒ the pass runs) — the marker only exists to save LLM
    spend, so it must never block a dream.
    """
    # Lazy import matching locks.py — keeps the module cheap to import
    # in tests that mock redis.
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        raw = await redis.get(_last_completed_key(user_id))
    except Exception:
        logger.warning(
            "Failed to read dream last-completed marker for user %s — "
            "running the pass",
            user_id[:12],
            exc_info=True,
        )
        return None
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        marker = datetime.fromisoformat(str(raw))
    except (UnicodeDecodeError, ValueError):
        logger.warning(
            "Unparseable dream last-completed marker for user %s — running the pass",
            user_id[:12],
        )
        return None
    return marker if marker.tzinfo else marker.replace(tzinfo=timezone.utc)


async def _stamp_last_completed_marker(user_id: str, as_of: datetime) -> None:
    """Record the upper bound of the episode window the pass consolidated.

    ``as_of`` must be the gather snapshot time (``DreamInput.window_end``),
    NOT apply-completion time: the three LLM phases + apply take minutes,
    and an episode enqueued in that window was absent from the bundle —
    stamping "now" would mark it as already consolidated and skip it until
    the next genuinely-new episode (or the marker TTL).

    Best-effort: a failed stamp only costs one extra full pass on the
    next nightly tick.
    """
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        await redis.set(
            _last_completed_key(user_id),
            as_of.isoformat(),
            ex=LAST_COMPLETED_TTL_SECONDS,
        )
    except Exception:
        logger.warning(
            "Failed to stamp dream last-completed marker for user %s",
            user_id[:12],
            exc_info=True,
        )


def _has_new_episodes_since(episodes: list[EpisodeRow], marker: datetime) -> bool:
    """True when any *user* episode is newer than the last-completed marker.

    Dream-authored episodes are excluded: a productive pass enqueues its
    own writes with ``valid_at = now()`` (after the stamped ``window_end``),
    so counting them would make every subsequent nightly run a paid no-op
    that only re-reads its own output.

    An episode with no parseable timestamp counts as new — we can't
    prove it's old, and skipping a pass we owed the user is worse than
    running one we didn't.
    """
    for episode in episodes:
        if is_dream_authored_episode(episode):
            continue
        episode_at = parse_episode_timestamp(episode)
        if episode_at is None or episode_at > marker:
            return True
    return False


async def _execute_dream_pass_async(
    user_id: str,
    *,
    transport_is_local: bool = False,
    config: ChatConfig | None = None,
    status_id: str | None = None,
) -> DreamPassResult:
    config = config or ChatConfig()
    pass_id = str(uuidlib.uuid4())
    started_at = datetime.now(timezone.utc)
    monotonic_start = asyncio.get_event_loop().time()

    has_anthropic_key = bool(config.direct_anthropic_api_key)
    # Step 5: the async Anthropic batch path is gated by the
    # ``DREAM_PASS_BATCH_ENABLED`` LD flag AND a direct Anthropic key (the
    # native Batch API can't be reached via OpenRouter/subscription, so the
    # key is a hard requirement). When the flag is off, dreams run on the
    # synchronous baseline regardless of key presence — the flag lets the
    # batch path ship dark and roll out per-cohort. When on, phase 1 submits
    # via call_provider(execution_mode="batch"); the BatchExecutor polls and
    # dream's batch_callbacks chain phases 2 → 3 + apply when results land
    # (~50% off the rate card, see model_pricing.execution_path_discount).
    #
    # ``transport_name`` short-circuits to sync_baseline for transports that
    # can't honour a batch path (local backends have no batch API;
    # subscription mode shouldn't dual-bill the user's Anthropic key when the
    # chat layer is on Claude Code OAuth).
    batch_enabled = await is_feature_enabled(Flag.DREAM_PASS_BATCH_ENABLED, user_id)
    execution_path: ExecutionPath = resolve_dream_execution_path(
        has_anthropic_key=has_anthropic_key,
        batch_processing_enabled=batch_enabled,
        transport_name=config.transport.name,
    )

    ttl = _resolve_lock_ttl(transport_is_local)
    try:
        async with dream_lock(user_id, ttl_seconds=ttl) as dream_lock_handle:
            # Pre-flight billing check. Runs inside the lock so a
            # paywalled user doesn't burn the slot for an eligible
            # concurrent pass on a shared FalkorDB.
            budget_ok, budget_skip = await check_dream_budget(user_id, config=config)
            if not budget_ok:
                if budget_skip == "rate_limit_unavailable":
                    return _failure_result(
                        user_id,
                        pass_id,
                        started_at,
                        monotonic_start,
                        execution_path,
                        f"billing: {budget_skip}",
                    )
                return DreamPassResult(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                    execution_path=execution_path,
                    skipped=True,
                    skip_reason=budget_skip or "insufficient_credits",
                )

            input_bundle = await gather_dream_input(user_id)

            if not input_bundle.episodes and not input_bundle.facts:
                # Nothing to consolidate — early-return as skipped so the
                # admin UI can render "nothing to dream about yet".
                return DreamPassResult(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                    execution_path=execution_path,
                    skipped=True,
                    skip_reason="no_input",
                )

            # No NEW activity since the last completed pass — every episode
            # in the bundle predates the marker, so re-running all three
            # LLM phases would only re-chew already-consolidated material
            # (and, before the empty-pass guard in apply.py, manufacture an
            # empty dream chat). Marker read is best-effort: missing,
            # unparseable, or Redis-down all mean "run the pass".
            #
            # Manual admin triggers (the only callers that set status_id)
            # bypass the marker: "dream now" is the memory-debugging tool
            # for re-running a pass after prompt/flag/model changes, and a
            # silent no_new_activity skip would neuter it for up to the
            # marker's 35-day TTL.
            last_completed = (
                await _read_last_completed_marker(user_id)
                if status_id is None
                else None
            )
            if last_completed is not None and not _has_new_episodes_since(
                input_bundle.episodes, last_completed
            ):
                logger.info(
                    "Dream pass %s skipped for user %s — no episodes newer "
                    "than last completed pass at %s",
                    pass_id,
                    user_id[:12],
                    last_completed.isoformat(),
                )
                return DreamPassResult(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                    execution_path=execution_path,
                    skipped=True,
                    skip_reason="no_new_activity",
                )

            # ---- Anthropic batch path -----------------------------------
            #
            # When routing picks ``anthropic_batch`` we persist the
            # input bundle, submit phase 1 to Anthropic's Messages
            # Batches API with forced tool_use structured output, and
            # return ``status="submitted"`` immediately. The BatchExecutor
            # polls; dream's batch_callbacks chain phases 2 → 3 + apply
            # when each phase result lands. Total latency is provider-
            # driven (typically <30min, hard cap 24h via
            # BatchExecutor.MAX_BATCH_LIFETIME_SECONDS); the user sees
            # JobStatus.current_phase advance through consolidate →
            # recombine → sanitize → complete.
            if execution_path == "anthropic_batch":
                return await _submit_dream_pass_batch(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    monotonic_start=monotonic_start,
                    execution_path=execution_path,
                    config=config,
                    input_bundle=input_bundle,
                    status_id=status_id,
                    dream_lock_handle=dream_lock_handle,
                )

            step_usages: list[PhaseUsage] = []

            try:
                consolidate_completion = await _run_consolidate(config, input_bundle)
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"consolidate: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "consolidate", consolidate_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            consolidated = consolidate_completion.value

            try:
                recombine_completion = await _run_recombine(
                    config, input_bundle, consolidated
                )
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"recombine: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "recombine", recombine_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            recombined = recombine_completion.value

            try:
                sanitize_completion = await _run_sanitize(
                    config, input_bundle, consolidated, recombined
                )
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"sanitize: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "sanitize", sanitize_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            sanitized = sanitize_completion.value

            ops = _clamp_operations(
                sanitized,
                len(input_bundle.facts),
                known_fact_uuids=input_bundle.known_fact_uuids,
            )
            apply_stats = await apply_operations(
                user_id,
                pass_id,
                ops,
                known_fact_uuids=input_bundle.known_fact_uuids,
                lock_handle=dream_lock_handle,
            )
            # Apply succeeded (even as a no-op) — stamp the marker so the
            # next nightly pass can skip when nothing new has landed.
            # Stamped with the gather-window end so episodes that arrived
            # mid-pass still count as new next time. Sync path only: batch
            # apply runs hours later in batch_callbacks, which doesn't
            # stamp yet.
            await _stamp_last_completed_marker(user_id, input_bundle.window_end)

            completed_at = datetime.now(timezone.utc)
            snapshot = apply_stats.get("snapshot")
            raw_session_id = apply_stats.get("session_id")

            def _as_int(key: str) -> int:
                v = apply_stats.get(key, 0)
                return int(v) if isinstance(v, (int, str)) and v else 0

            # Fail-closed: a missing/malformed drain flag reads as
            # ``timed_out`` (writes at risk), never a confirmed drain.
            ingestion_drain_status = drain_status_from_stats(apply_stats)

            return DreamPassResult(
                user_id=user_id,
                pass_id=pass_id,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                execution_path=execution_path,
                consolidated_count=_as_int("consolidated_count"),
                proposal_count=_as_int("proposal_count"),
                demotion_count=_as_int("demotion_count"),
                entity_invalidation_count=_as_int("entity_invalidation_count"),
                summary_for_user=ops.summary_for_user,
                ingestion_drain_status=ingestion_drain_status,
                # ``None`` (key absent) on an empty pass — apply skipped the
                # dream session entirely, so there is no id to surface.
                dream_session_id=(
                    raw_session_id if isinstance(raw_session_id, str) else None
                ),
                operations=(
                    snapshot if isinstance(snapshot, DreamOperationsSnapshot) else None
                ),
                usage=_aggregate_usage(step_usages, execution_path),
            )

    except DreamLockHeld:
        return DreamPassResult(
            user_id=user_id,
            pass_id=pass_id,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
            execution_path=execution_path,
            skipped=True,
            skip_reason="lock_held",
        )
    except Exception as exc:  # pragma: no cover — last-resort guard
        logger.exception("Dream pass crashed for user %s: %s", user_id[:12], exc)
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            str(exc),
        )


def _failure_result(
    user_id: str,
    pass_id: str,
    started_at: datetime,
    monotonic_start: float,
    execution_path: ExecutionPath,
    error: str,
    usage: DreamPassUsage | None = None,
) -> DreamPassResult:
    """Build a failure ``DreamPassResult`` that still carries usage for
    the phases that completed before the error — billing has to charge
    for tokens we already paid for, even on partial failure."""
    return DreamPassResult(
        user_id=user_id,
        pass_id=pass_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        elapsed_seconds=asyncio.get_event_loop().time() - monotonic_start,
        execution_path=execution_path,
        error=error,
        usage=usage,
    )


async def execute_dream_pass(
    user_id: str, *, status_id: str | None = None
) -> DreamPassResult:
    """Public async entry point used by the scheduler + admin trigger.

    ``status_id`` is the optional JobStatus row id when the caller is
    the polling admin trigger (Step 2). The orchestrator threads it
    into the batch path so the BatchExecutor callbacks can update the
    user-visible status row as each phase lands. AgentProbe + other
    sync callers pass ``None`` and never hit the batch path.
    """
    return await _execute_dream_pass_async(user_id, status_id=status_id)


async def _submit_dream_pass_batch(
    *,
    user_id: str,
    pass_id: str,
    started_at: datetime,
    monotonic_start: float,
    execution_path: ExecutionPath,
    config: ChatConfig,
    input_bundle: DreamInput,
    status_id: str | None,
    dream_lock_handle: DreamLockHandle,
) -> DreamPassResult:
    """Submit phase 1 of the dream pass via Anthropic batch + return.

    The orchestrator hands control to the BatchExecutor here: phase 1
    (consolidate) is enqueued; phases 2 (recombine) and 3 (sanitize)
    fire from dream's batch_callbacks as each prior phase's result
    lands. Apply + cost log + JobStatus complete run when sanitize
    lands. This function only does the kickoff.

    The job_id link to JobStatus comes from the scheduler's
    ``execute_dream_pass_with_status`` wrapper (Step 2). When the
    orchestrator is invoked outside that path (e.g. AgentProbe eval),
    ``job_id`` is empty and the callback skips the JobStatus updates —
    apply still runs, cost still logs.
    """
    from .batch_submit import (
        persist_input_bundle,
        phase_models_for_config,
        submit_phase,
    )

    api_key = config.direct_anthropic_api_key
    if not api_key:
        # Shouldn't happen — routing.py only picks anthropic_batch when
        # the key is present. Guard for type-narrowing + future safety.
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            "anthropic_batch: no Anthropic API key (routing bug)",
        )

    # Persist DreamInput so the per-phase callbacks can rebuild the
    # next phase's prompt without re-fetching from Postgres + FalkorDB.
    try:
        await persist_input_bundle(
            pass_id, input_bundle, lock_token=dream_lock_handle.token
        )
    except Exception as exc:
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            f"anthropic_batch: input persist failed: {exc}",
        )

    # ``status_id`` ties the per-phase callback updates back to the
    # JobStatus row the admin trigger created. Empty string when no
    # caller wired it (AgentProbe eval, ad-hoc invocation) — the
    # callback skips status writes in that case but apply still runs.
    try:
        submission = await submit_phase(
            user_id=user_id,
            pass_id=pass_id,
            job_id=status_id or "",
            phase="consolidate",
            phase_models=phase_models_for_config(config),
            api_key=api_key,
            input_bundle=input_bundle,
        )
    except Exception as exc:
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            f"anthropic_batch: phase 1 submit failed: {exc}",
        )

    completed_at = datetime.now(timezone.utc)
    logger.info(
        "Dream pass %s submitted via Anthropic batch=%s (phase=consolidate)",
        pass_id,
        submission.provider_batch_id,
    )
    # Phase 1 is enqueued — hand the dream lock to the batch callback so it
    # spans the full async lifetime (apply runs hours later). Extend the TTL
    # to the batch window first; the callback releases it on terminal/failure.
    # A failed extend means the lock expired before the handoff — a newer
    # pass may already own the graph, so the just-submitted batch must never
    # be applied: revoke the pending entry (the poller then never dispatches
    # the callback chain) and drop the input bundle. The provider batch is
    # orphaned; its results are discarded. The lock is NOT disowned, so the
    # context manager's compare-and-delete release stays a safe no-op.
    if not await dream_lock_handle.extend(BATCH_LOCK_TTL_SECONDS):
        from backend.executor.batch_executor import remove_pending

        from .batch_submit import delete_input_bundle

        await remove_pending(submission.provider_batch_id)
        await delete_input_bundle(pass_id)
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            "anthropic_batch: dream lock lost before handoff — batch revoked",
        )
    dream_lock_handle.disown()
    return DreamPassResult(
        user_id=user_id,
        pass_id=pass_id,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=asyncio.get_event_loop().time() - monotonic_start,
        execution_path=execution_path,
        skipped=False,
        # The pass is async-pending — no usage / ops / session_id yet.
        # The BatchExecutor + dream callbacks deliver those when phase
        # 3 lands and apply runs.
    )
