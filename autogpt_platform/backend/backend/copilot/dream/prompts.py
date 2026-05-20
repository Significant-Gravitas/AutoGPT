"""Prompt builders for the three-phase dream pipeline.

Slice-1 ships hardcoded prompts. Once Langfuse seeds land, swap the
body of each ``build_*`` for a Langfuse fetch with these strings as
the local fallback.

Phase 1 — Consolidation (Sonnet, temp 0.2, structured JSON output)
Phase 2 — Recombination (Opus, temp 0.9, structured JSON output)
Phase 3 — Sanitize + Gate (Sonnet, temp 0.0, structured JSON output)

Each prompt asks the model to emit JSON matching the corresponding
Pydantic schema in ``schemas.py``. We rely on ``response_format=
{"type": "json_object"}`` plus post-validation, which works against
both OpenAI and OpenRouter without provider-specific shenanigans.
"""

from __future__ import annotations

import json

from .fetch import DreamInput

# Hard cap shared across phases — phase 3 must reject demotion lists
# longer than this regardless of what phase 1/2 produced. Mirrors the
# guardrail in ``dream/p0-spec.md`` §2.
MAX_DEMOTIONS_PER_PASS = 10
MAX_PROPOSALS_PER_PASS = 20
MAX_WRITES_PER_PASS = 30


def _format_episodes(input_bundle: DreamInput, max_chars_per_episode: int = 500) -> str:
    if not input_bundle.episodes:
        return "(no recent episodes in window)"
    lines: list[str] = []
    for e in input_bundle.episodes:
        body = (e.content or "")[:max_chars_per_episode]
        lines.append(f"- uuid={e.uuid} valid_at={e.valid_at}\n  {body}".rstrip())
    return "\n".join(lines)


def _format_facts(input_bundle: DreamInput) -> str:
    if not input_bundle.facts:
        return "(no active facts)"
    by_scope: dict[str, list[str]] = {}
    for f in input_bundle.facts:
        scope = f.scope or "real:global"
        bucket = by_scope.setdefault(scope, [])
        bucket.append(
            f"  - uuid={f.uuid} confidence={f.confidence} "
            f"{(f.source or '?')} —[{f.name or '?'}]→ {(f.target or '?')}: "
            f"{(f.fact or '').strip()}"
        )
    parts: list[str] = []
    for scope, bucket in sorted(by_scope.items()):
        parts.append(f"[scope={scope}]")
        parts.extend(bucket)
    return "\n".join(parts)


def _format_sessions(input_bundle: DreamInput) -> str:
    if not input_bundle.recent_sessions:
        return "(no recent sessions)"
    parts: list[str] = []
    for s in input_bundle.recent_sessions:
        title = s.title or "(untitled)"
        parts.append(f"--- session {s.session_id} — {title} ---\n{s.body}".rstrip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1 — Consolidation
# ---------------------------------------------------------------------------


PHASE_1_SYSTEM = (
    "You are the consolidation step of a sleep-time memory pass for an "
    "AI assistant. Your job is to look at the user's recent episodes "
    "and active facts and emit a small set of *consolidated* statements "
    "that merge near-duplicates into a canonical form.\n\n"
    "Rules:\n"
    " * Do not invent new information that isn't present in the inputs.\n"
    " * Prefer the longer, more specific phrasing when merging duplicates.\n"
    " * Group by scope — never merge a fact in 'project:foo' with one in "
    "'real:global'.\n"
    " * Carry forward the source episode uuids exactly. Never make uuids up.\n"
    " * Confidence is your own — 0.0 means 'I am unsure', 1.0 means "
    "'this is restated verbatim across multiple sources'.\n"
    f" * Output AT MOST {MAX_WRITES_PER_PASS} consolidated facts. Better "
    "to consolidate hard than to flood phase 2.\n\n"
    "Return a JSON object matching this schema:\n"
    '{ "facts": [ { "content": str, "scope": str, "confidence": float, '
    '"source_episode_uuids": [str, ...] } ] }'
)


def build_phase_1_prompt(input_bundle: DreamInput) -> list[dict[str, str]]:
    user_body = (
        "## Recent episodes (within window)\n"
        + _format_episodes(input_bundle)
        + "\n\n## Active facts\n"
        + _format_facts(input_bundle)
        + "\n\n## Recent sessions (for context only — do not extract facts from these)\n"
        + _format_sessions(input_bundle)
    )
    return [
        {"role": "system", "content": PHASE_1_SYSTEM},
        {"role": "user", "content": user_body},
    ]


# ---------------------------------------------------------------------------
# Phase 2 — Recombination
# ---------------------------------------------------------------------------


PHASE_2_SYSTEM = (
    "You are the recombination step of a sleep-time memory pass. The "
    "previous step consolidated the user's recent facts. Your job is to "
    "look for *novel connections* — weak-link findings, implications "
    "across scopes, useful rules the user has implicitly demonstrated.\n\n"
    "Rules:\n"
    " * Every proposal must cite at least one source_fact_uuid or "
    "source_episode_uuid from phase 1's input. Proposals with no citation "
    "will be rejected by phase 3.\n"
    " * Stay in scope — proposed findings live in the same scope as "
    "their evidence.\n"
    ' * Do not propose first-person-as-user statements ("I think X"). '
    "All proposals must describe the user's world, not the assistant's.\n"
    f" * Output AT MOST {MAX_PROPOSALS_PER_PASS} proposals. Quality > quantity.\n"
    " * Confidence here is your own self-rating of how likely the "
    "proposed finding is to be useful to the user.\n\n"
    "Return a JSON object matching this schema:\n"
    '{ "proposals": [ { "content": str, "scope": str, "memory_kind": str, '
    '"confidence": float, "rationale": str, "source_episode_uuids": [str, ...], '
    '"source_fact_uuids": [str, ...] } ] }'
)


def build_phase_2_prompt(
    input_bundle: DreamInput, phase_1_json: str
) -> list[dict[str, str]]:
    user_body = (
        "## Phase 1 consolidated output\n"
        + phase_1_json
        + "\n\n## Active facts (for reference)\n"
        + _format_facts(input_bundle)
        + "\n\n## Recent episodes (for reference)\n"
        + _format_episodes(input_bundle, max_chars_per_episode=300)
    )
    return [
        {"role": "system", "content": PHASE_2_SYSTEM},
        {"role": "user", "content": user_body},
    ]


# ---------------------------------------------------------------------------
# Phase 3 — Sanitize + Gate
# ---------------------------------------------------------------------------


PHASE_3_SYSTEM = (
    "You are the sanitizer step of a sleep-time memory pass. You receive "
    "phase 1's consolidated facts and phase 2's proposals; you decide "
    "what actually lands in memory and what gets demoted.\n\n"
    "Rules (HARD — outputs that violate these will be dropped):\n"
    f" * AT MOST {MAX_WRITES_PER_PASS} writes.\n"
    f" * AT MOST {MAX_PROPOSALS_PER_PASS} proposals.\n"
    f" * AT MOST {MAX_DEMOTIONS_PER_PASS} demotions per pass. Demotions "
    "are surgical — only demote a fact when phase 1 or phase 2 directly "
    "contradicts it, or when it is obviously stale.\n"
    " * Demotion edge_uuids MUST exist in the provided list of known "
    "fact uuids. Do not invent uuids.\n"
    " * Entity invalidations require an entity_uuid present in the "
    "active-facts source/target set. Single-hop demotion only.\n"
    " * Reject any proposal that has zero source citations.\n"
    " * Reject any proposal whose content starts with first-person "
    '("I think", "I believe", "In my opinion") — those are not user '
    "memories.\n"
    " * Write a short ``summary_for_user`` (1-3 sentences) describing "
    "what the pass found.\n\n"
    "Return a JSON object matching the DreamOperations schema:\n"
    '{ "writes": [ConsolidatedFact...], "proposals": [ProposedFinding...], '
    '"demotions": [{"edge_uuid": str, "reason": str, '
    '"new_status": "superseded"|"contradicted"} ...], '
    '"entity_invalidations": [{"entity_uuid": str, "reason": str} ...], '
    '"summary_for_user": str }'
)


def build_phase_3_prompt(
    input_bundle: DreamInput,
    phase_1_json: str,
    phase_2_json: str,
) -> list[dict[str, str]]:
    known_fact_uuids = sorted(input_bundle.known_fact_uuids)
    user_body = (
        "## Phase 1 output (consolidated facts)\n"
        + phase_1_json
        + "\n\n## Phase 2 output (proposed findings)\n"
        + phase_2_json
        + "\n\n## Known fact uuids (only these are valid demotion targets)\n"
        + json.dumps(known_fact_uuids)
        + "\n\n## Active facts (for context when deciding demotions)\n"
        + _format_facts(input_bundle)
    )
    return [
        {"role": "system", "content": PHASE_3_SYSTEM},
        {"role": "user", "content": user_body},
    ]
