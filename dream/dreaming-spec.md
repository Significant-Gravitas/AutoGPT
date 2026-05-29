# Dreaming — Cognitive Spec

Per-user nightly dream pass. Implements the three defensible functions from research
Part II — (1) replay + recombination, (2) affective recalibration, (3) scenario
simulation — on top of existing chat, Graphiti, and APScheduler primitives. Mirrors
the NREM/REM split. All outputs are `MemoryEnvelope` instances; no parallel type
system.

---

## 1. Dream pass anatomy

Three sequential LLM calls per pass.

**Phase 1 — Consolidation (NREM-equivalent)**
- Model: `claude-sonnet-4-6`, standard mode. Temp `0.2`, max output `8000`.
- Structured JSON output: `Phase1Output = {consolidated_facts, demotions, summary}`.
- Prompt input: system prompt (§4), `<sessions>`, `<episodes>`, `<active_facts>`,
  `<dream_context>`, `<scope_index>`.
- Job: extract stably-true facts implicit in transcripts; flag existing memories
  to demote.

**Phase 2 — Recombination (REM-equivalent)**
- Model: `claude-opus-4-7` with extended thinking (`thinking.budget_tokens=4000`).
  Temp `0.9` — hyper-associativity is the function. Max output `6000`.
- Structured JSON: `Phase2Output = {proposed_findings, scenario_seeds}`.
- Input: phase 1's `summary` + `ConsolidatedFact[]` (not raw transcripts),
  `<active_facts>`, `<scope_index>`.
- Job: find weak links across sessions; emit tentative findings/rules/procedures.

**Phase 3 — Sanitize & commit**
- Model: `claude-sonnet-4-6`. Temp `0.0`, max output `2000`. Same schemas, filtered.
- Drops proposals that (a) duplicate existing active facts, (b) reference entities
  absent from inputs, (c) cross scopes (output scope doesn't match evidence scope).
- Survivors written via `enqueue_episode()` with `source_kind=assistant_derived`,
  `provenance="dream:{job_id}:{phase}"`.

Three phases give cost control (one Opus call), failure isolation (phase 3 is a
deterministic gate), and a clean audit trail.

---

## 2. Inputs

| Block | Volume | Reasoning |
|---|---|---|
| Sessions | Last **10**, **8 KB content each** (80 KB cap) | Covers a workweek; 8 KB is roughly the median session today. |
| Episodes | Last **50** Graphiti episodes | Below *Dreams*' 100-session ceiling because we also bring facts. |
| Active facts | All `status=active` touched in last **14 days** + all `status=tentative` | "Touched" = added, edited, retrieved, referenced. Cap at **500**; sample by recency then retrieval-frequency if exceeded. |
| Scope index | Full enumerated list, no content | Cheap. Prevents scope hallucination. |
| Dream context | Last pass timestamp + last 20 ratification outcomes | Lets phase 1 see which prior proposals stuck. |

Total budget ~120 KB. Comfortable inside Sonnet's window.

**Skip the pass entirely** if: zero sessions in 30 days, single-session user with
no prior memory, or `dream_enabled=false` on `ChatSessionMetadata`. Short-circuits
before the first LLM call.

---

## 3. Outputs

All three classes are `MemoryEnvelope` operations.

**Consolidated facts** (phase 1, high confidence, `status=active`):
```jsonc
{ "kind": "consolidated_fact",
  "envelope": { "content": "Sarah is on the billing team.",
                "source_kind": "assistant_derived", "scope": "real:global",
                "memory_kind": "fact", "status": "active", "confidence": 0.85,
                "provenance": "dream:{job_id}:phase1" },
  "evidence": ["session:abc#msg14", "episode:7f3..."] }
```

**Demotions** (operations on existing UUIDs):
```jsonc
{ "kind": "demotion", "target_uuid": "5e1f...",
  "new_status": "superseded",                   // or "contradicted"
  "reason": "Replaced by newer fact in session:abc#msg9.",
  "evidence": ["session:abc#msg9"],
  "provenance": "dream:{job_id}:phase1" }
```

**Proposed findings** (phase 2, lower confidence, always `tentative`):
```jsonc
{ "kind": "proposed_finding",
  "envelope": { "content": "User checks Stripe logs before app logs when debugging payments.",
                "source_kind": "assistant_derived", "scope": "project:billing-app",
                "memory_kind": "finding", "status": "tentative", "confidence": 0.55,
                "provenance": "dream:{job_id}:phase2" },
  "evidence": ["session:abc#msg10", "session:xyz#msg88"],
  "ratification_hint": "Confirm on next Stripe debugging session." }
```

Ratification extends an existing mechanism: warm-context retrieval surfaces
`tentative` facts; if the next user turn references the entity, flip to `active`;
if contradicted, flip to `contradicted`; otherwise 30-day TTL flips to `superseded`
with reason `"unratified"`.

---

## 4. Prompts (draft)

### Phase 1 — Consolidation

```
You are the consolidation pass of a memory system. You do not talk to the user.
You read recent sessions and existing memories, and you emit JSON.

Your job: identify facts that are (a) stably true across recent transcripts,
(b) cleanly extractable into one sentence each, and (c) genuinely new or
genuinely contradicted by existing memory.

Inputs: <sessions>, <episodes>, <active_facts> (each with UUID), <dream_context>,
<scope_index> (legal scope values — pick from this list).

Good output:
- One short content string per fact. No prose or caveats inside content.
- Every fact cites ≥1 evidence reference (session_id#msg_seq or episode UUID).
- Demotions are surgical: name UUID, contradicting evidence, status reason.
- Confidence is honest; if sessions disagree, lower it and consider a demotion.

You must NOT:
- Invent entities not present in inputs.
- Cross scopes: a `project:alpha` fact cites only `project:alpha` or `real:global`.
- Write in the user's voice — third person, system voice.
- Summarize whole sessions as one fact; break down or drop.
- Propose creative findings (next phase's job).

EXAMPLE — sessions say "Sarah moved to billing"; existing fact says "Sarah is on auth (uuid=abc)":
{ "consolidated_facts": [{ "envelope": {"content":"Sarah is on the billing team.",
   "scope":"real:global","memory_kind":"fact","status":"active","confidence":0.9,...},
   "evidence":["session:s1#msg14"] }],
  "demotions": [{"target_uuid":"abc","new_status":"superseded",
   "reason":"Moved to billing per session s1.","evidence":["session:s1#msg14"]}],
  "summary": "Sarah's team assignment changed." }

COUNTER-EXAMPLE — inputs never mention Slack:
  {"content":"User probably uses Slack"}   // hallucinated, do not emit
```

### Phase 2 — Recombination

```
You are the recombination pass. Consolidation already ran; you receive its summary
and structured facts, plus existing memory. Find non-obvious connections, weak
links, and patterns across sessions that no single session would reveal — propose
them as TENTATIVE memories.

Think REM sleep: broader associations, relaxed reality testing, but content is
still grounded in this user's actual concerns. Every proposal points at evidence
already in the inputs.

Good output:
- Findings span ≥2 sessions, or 1 session + 1 prior fact.
- Rules/procedures inferred from repeated behavior, not single instances.
- Confidence 0.4–0.7. >0.7 means you're restating a consolidated fact (drop it);
  <0.4 means you're guessing.
- At most 12 proposals. Phase 3 dedupes; do not pad.

You must NOT:
- Invent people, projects, tools, or events.
- Write first-person ("I should..."). Third person, system voice.
- Cite another proposed finding as evidence — only consolidated facts or
  session/episode refs.
- Tag a scope whose evidence is from a different scope.

EXAMPLE — user debugged three bugs by checking Stripe logs first:
{ "proposed_findings": [{ "envelope": {
    "content":"User tends to check Stripe logs before app logs when debugging payments.",
    "scope":"project:billing-app","memory_kind":"finding","status":"tentative",
    "confidence":0.6,"provenance":"dream:{job_id}:phase2",...},
    "evidence":["session:s1#msg10","session:s2#msg22","session:s4#msg7"],
    "ratification_hint":"Confirm on next Stripe debugging session." }] }

COUNTER-EXAMPLES:
  {"content":"I always check Stripe first.","status":"active"}   // wrong voice, wrong status
  {"content":"User likes coffee."}                                // not in inputs
```

---

## 5. Evaluations

**5a. Demotion precision.** Sample 50 demotions/day. Evaluator (`claude-opus-4-7`,
temp 0) sees same evidence + current fact text; emits
`{agree-superseded, agree-contradicted, disagree, insufficient-evidence}`. Target
**≥85% agree**. Below 70% halts the next pass for investigation.

**5b. Warm-context recall lift.** Hold out 30 representative queries per power user
(sample first-turn waking questions, anonymize via LLM). Score retrieval relevance
T0 (pre-dream) vs T1 (post-dream) on 1–5 Likert via judge LLM. Target **median lift
≥ +0.3 Likert per user-week**.

**5c. Ratification rate.** For each `tentative` written, track within 7 days:
explicit confirm, implicit use (cited and not contradicted), or contradiction.
Target **≥35% ratified, ≤10% contradicted**. If contradicted >15% one week, drop
phase 2 temp by 0.1 the next.

**5d. Cache-hit rate (scenario seeds, v2).** Of next 24h of user turns, fraction
served with a material cache hit. Ship-bar: **≥20% hit-rate with ≥30% latency
reduction on hits**; falls back to off below 10% after two weeks.

**5e. Negative-example battery (CI):**
- Empty user (0 sessions, 0 memories) → short-circuit, zero LLM cost.
- Single-session user → phase 1 runs, phase 2 skipped, no `proposed_finding`.
- Contradictory-facts user (two `active` facts that flatly disagree) → ≥1 demotion
  with `new_status=contradicted` and evidence pointing to the other.

---

## 6. Failure modes & mitigations

| Failure | Mitigation |
|---|---|
| **Drift (dream-of-dreams)** — phase 2 cites phase 2 findings, compounding. | Prompt forbids it; phase 3 drops any proposal whose evidence is entirely `dream:*` provenance. |
| **Hallucinated entities** — facts about people/projects not in inputs. | Phase 3 builds an entity allowlist (cheap NER over transcripts + active facts); drops violators with reason `unknown_entity`. |
| **Scope leakage** — project A facts get tagged project B / global. | Explicit `scope_index` in prompts + deterministic enforcement: output scope must match evidence scope, or be `real:global` only with multi-project evidence. |
| **Persona drift** — dream picks up user voice ("I prefer..."). | Prompts require third-person system voice; phase 3 regex-rejects any envelope content starting `I `, `my `, `me `, `we `. |
| **Runaway demotion** — half the fact base flipped after one weird session. | Hard cap: **≤10 demotions per pass and ≤5% of active fact set**, whichever is smaller. Extras logged as deferred, not silently dropped. |
| **Cost blowup** — large user runs Opus over hundreds of KB nightly. | Hard input budget (§2); Opus only in phase 2; phase 2 sees consolidated phase 1 output, not raw transcripts; scheduler `max_instances=1`. |
| **Stale ratification queue** — `tentative` piles up unreviewed. | 30-day TTL → auto-superseded `unratified`. Surfaced in admin dashboards. |

---

## 7. Open questions for product

1. **Surface dream output to users?** (a) silent / (b) read-only / (c) read + veto.
   **Recommend (c)** — vetoes are training signal and welfare-adjacent without being mystical.
2. **One pass per user, or per (user × scope)?** **Recommend (a) combined** for v1
   — cross-scope recombination is part of the value; *Dreams* uses one store per pass.
   Per-scope is a v2 toggle.
3. **Non-English dreams?** **Recommend (b) detect-and-emit** — system prompt stays
   English, but include `<user_dominant_language>` and instruct content in that language.
4. **Welfare framing in copy?** (a) "memory hygiene" / (b) "your assistant's quiet
   time" / (c) silent. **Recommend (a)** — matches Anthropic's framing, sets up (b) later.
5. **Ratification: automatic, prompted, or both?** **Recommend (a) automatic** from
   waking-chat signals. Don't make users do chores. TTL handles unratified buildup.
6. **Roll out to all users, or beta cohort?** **Recommend (b) LaunchDarkly cohort**
   at 1%, scale on the 85% demotion-precision gate.

---

**v1 scope**: phases 1–3, evals 5a/5b/5c/5e, all §6 mitigations, §7 recommendations.
Defer 5d and scenario-simulation seeding to v2 once warm-context lift is verified.
Minimum loop first; creative reach extends after the floor holds.
