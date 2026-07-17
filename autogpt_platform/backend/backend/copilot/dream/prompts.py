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
from .staleness import identify_stale_candidates

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
# Consolidation step — merge near-duplicate recent facts into canonical form.
# ---------------------------------------------------------------------------


CONSOLIDATE_SYSTEM = (
    "OUTPUT FORMAT — MANDATORY:\n"
    "Your entire response MUST be a single valid JSON object and "
    "NOTHING ELSE. The first character of your response MUST be `{`. "
    "Do NOT preface with prose like 'Looking at the inputs...'. "
    "Do NOT wrap the JSON in markdown code fences. Do NOT use "
    "markdown formatting anywhere (no **bold**, no numbered lists, "
    "no headings). Do NOT explain your reasoning before or after "
    "the JSON. If you need to think, do it silently — only the JSON "
    "is returned.\n\n"
    "ROLE:\n"
    "You are the consolidation step of a sleep-time memory pass for an "
    "AI assistant. Your job is to look at the user's recent episodes "
    "and active facts and emit a small set of *consolidated* statements "
    "that merge near-duplicates into a canonical form.\n\n"
    "RULES:\n"
    " * Do not invent new information that isn't present in the inputs.\n"
    " * Prefer the longer, more specific phrasing when merging duplicates.\n"
    " * Group by scope — never merge a fact in 'project:foo' with one in "
    "'real:global'.\n"
    " * Carry forward the source episode uuids exactly. Never make uuids up.\n"
    " * Confidence is your own — 0.0 means 'I am unsure', 1.0 means "
    "'this is restated verbatim across multiple sources'.\n"
    f" * Output AT MOST {MAX_WRITES_PER_PASS} consolidated facts. Better "
    "to consolidate hard than to flood the recombination step.\n\n"
    "JSON SCHEMA (your response MUST match this shape):\n"
    '{ "facts": [ { "content": str, "scope": str, "confidence": float, '
    '"source_episode_uuids": [str, ...] } ] }\n\n'
    'If you have nothing to emit, return `{"facts": []}` — still JSON, '
    "still starting with `{`."
)


def build_consolidate_prompt(input_bundle: DreamInput) -> list[dict[str, str]]:
    user_body = (
        "## Recent episodes (within window)\n"
        + _format_episodes(input_bundle)
        + "\n\n## Active facts\n"
        + _format_facts(input_bundle)
        + "\n\n## Recent sessions (for context only — do not extract facts from these)\n"
        + _format_sessions(input_bundle)
        + "\n\nReturn ONLY the JSON object. Begin your response with `{`."
    )
    return [
        {"role": "system", "content": CONSOLIDATE_SYSTEM},
        {"role": "user", "content": user_body},
    ]


# ---------------------------------------------------------------------------
# Recombination step — propose novel connections + weak-link findings.
# ---------------------------------------------------------------------------


RECOMBINE_SYSTEM = (
    "OUTPUT FORMAT — MANDATORY:\n"
    "Your entire response MUST be a single valid JSON object and "
    "NOTHING ELSE. The first character of your response MUST be `{`. "
    "Do NOT preface with prose. Do NOT wrap the JSON in markdown "
    "code fences. Do NOT use markdown formatting anywhere. Do NOT "
    "explain reasoning before or after the JSON. Empty result is "
    '`{"proposals": []}`.\n\n'
    "ROLE:\n"
    "You are the recombination step of a sleep-time memory pass. The "
    "previous step consolidated the user's recent facts. Your job is to "
    "look for *novel connections* — weak-link findings, implications "
    "across scopes, useful rules the user has implicitly demonstrated.\n\n"
    "RULES:\n"
    " * Every proposal must cite at least one source_fact_uuid or "
    "source_episode_uuid from the consolidation step's input. Proposals "
    "with no citation will be rejected by the sanitizer.\n"
    " * Stay in scope — proposed findings live in the same scope as "
    "their evidence.\n"
    ' * Do not propose first-person-as-user statements ("I think X"). '
    "All proposals must describe the user's world, not the assistant's.\n"
    f" * Output AT MOST {MAX_PROPOSALS_PER_PASS} proposals. Quality > quantity.\n"
    " * Confidence here is your own self-rating of how likely the "
    "proposed finding is to be useful to the user.\n"
    ' * ``memory_kind`` MUST be one of: "finding", "rule", '
    '"preference", "plan". Do NOT invent new values (e.g. '
    '"inferred_fact", "recommendation", "insight") — those '
    "proposals are silently dropped.\n\n"
    "JSON SCHEMA (your response MUST match this shape):\n"
    '{ "proposals": [ { "content": str, "scope": str, "memory_kind": str, '
    '"confidence": float, "rationale": str, "source_episode_uuids": [str, ...], '
    '"source_fact_uuids": [str, ...] } ] }'
)


def build_recombine_prompt(
    input_bundle: DreamInput, consolidated_json: str
) -> list[dict[str, str]]:
    user_body = (
        "## Consolidated facts (from the consolidation step)\n"
        + consolidated_json
        + "\n\n## Active facts (for reference)\n"
        + _format_facts(input_bundle)
        + "\n\n## Recent episodes (for reference)\n"
        + _format_episodes(input_bundle, max_chars_per_episode=300)
        + "\n\nReturn ONLY the JSON object. Begin your response with `{`."
    )
    return [
        {"role": "system", "content": RECOMBINE_SYSTEM},
        {"role": "user", "content": user_body},
    ]


# ---------------------------------------------------------------------------
# Sanitization step — decide what lands as a write and what gets demoted.
# ---------------------------------------------------------------------------


SANITIZE_SYSTEM = (
    "OUTPUT FORMAT — MANDATORY:\n"
    "Your entire response MUST be a single valid JSON object and "
    "NOTHING ELSE. The first character of your response MUST be `{`. "
    "Do NOT preface with prose. Do NOT wrap the JSON in markdown "
    "code fences. Do NOT use markdown formatting anywhere. Do NOT "
    "explain reasoning before or after the JSON. An empty result is "
    'still JSON: `{"writes": [], "proposals": [], "demotions": [], '
    '"entity_invalidations": [], "summary_for_user": ""}`.\n\n'
    "ROLE:\n"
    "You are the sanitizer step of a sleep-time memory pass. You receive "
    "the consolidated facts and the recombination proposals; you decide "
    "what actually lands in memory and what gets demoted.\n\n"
    "RULES (HARD — outputs that violate these will be dropped):\n"
    f" * AT MOST {MAX_WRITES_PER_PASS} writes.\n"
    f" * AT MOST {MAX_PROPOSALS_PER_PASS} proposals.\n"
    f" * AT MOST {MAX_DEMOTIONS_PER_PASS} demotions per pass. Demotions "
    "are surgical — only demote a fact when (a) a consolidated fact or "
    "recombination proposal directly contradicts it, (b) it appears in "
    "the stale-fact candidates list below AND general knowledge tells "
    "you it has gone stale, or (c) the user has clearly retracted it.\n"
    " * STALE-FACT GUARDRAIL: the candidates list flags facts whose "
    "phrasing suggests they go stale fast (pricing, leadership, "
    "model versions, 'best'/'current'/'latest' claims). It is NOT a "
    "demote-on-sight list — stable facts can match the heuristics by "
    "coincidence (e.g. 'user's birthday is March 5' is dated but not "
    "stale). For each candidate, demote only when general knowledge or "
    "a phase-1 consolidated fact contradicts it. When in doubt, "
    "preserve.\n"
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
    "JSON SCHEMA (your response MUST match this DreamOperations shape):\n"
    '{ "writes": [ConsolidatedFact...], "proposals": [ProposedFinding...], '
    '"demotions": [{"edge_uuid": str, "reason": str, '
    '"new_status": "superseded"|"contradicted"} ...], '
    '"entity_invalidations": [{"entity_uuid": str, "reason": str} ...], '
    '"summary_for_user": str }'
)


def _format_stale_candidates(input_bundle: DreamInput) -> str:
    """Render the staleness-heuristic candidate list for the sanitize prompt.

    ``identify_stale_candidates`` returns ``(fact, score)`` ordered by
    descending score so the most-suspect candidates render first when
    the prompt is near its budget. Use the explicit demotion reason
    string ``"stale_fact"`` so apply.py's audit trail matches the
    spec's reason enumeration.
    """
    candidates = identify_stale_candidates(input_bundle.facts)
    if not candidates:
        return "(no stale-fact candidates flagged this pass)"
    lines: list[str] = []
    for fact, score in candidates:
        lines.append(
            f"- uuid={fact.uuid} score={score:.2f} "
            f"created_at={fact.created_at or '?'}: "
            f"{(fact.fact or '').strip()}"
        )
    return "\n".join(lines)


def build_sanitize_prompt(
    input_bundle: DreamInput,
    consolidated_json: str,
    recombined_json: str,
) -> list[dict[str, str]]:
    known_fact_uuids = sorted(input_bundle.known_fact_uuids)
    user_body = (
        "## Consolidated facts (from the consolidation step)\n"
        + consolidated_json
        + "\n\n## Recombination proposals (from the recombination step)\n"
        + recombined_json
        + "\n\n## Known fact uuids (only these are valid demotion targets)\n"
        + json.dumps(known_fact_uuids)
        + "\n\n## Stale-fact candidates (heuristic flags — verify before demoting)\n"
        + _format_stale_candidates(input_bundle)
        + "\n\n## Active facts (for context when deciding demotions)\n"
        + _format_facts(input_bundle)
        + "\n\nReturn ONLY the JSON object. Begin your response with `{`."
    )
    return [
        {"role": "system", "content": SANITIZE_SYSTEM},
        {"role": "user", "content": user_body},
    ]
