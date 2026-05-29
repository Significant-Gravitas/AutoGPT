# Dreaming P0.6 — AgentProbe survey

Source: `/Users/ntindle/code/agpt/AgentProbe`, fast-forwarded from `origin/main`. Working tree was clean apart from an untracked `.claude/settings.local.json` (not modified). After pull, HEAD is:

```
a61df07 Add human scoring view with rubric correlation (#38)
c6aa022 narrow scenario_runs projection on http getRun paths
00d0a0a trim run-detail payload to fix slow postgres response times
03f5f05 instrument request-scoped perf spans for slow run detail path
0bafd60 fix crash issue
c4d9953 add logging for slow requests
c9c8840 refactor
aad69b8 adding loading skeletons
8a69513 update chat components
3202390 updating to full react app
```

## 1. Repo orientation

AgentProbe is a Bun + TypeScript CLI (and React dashboard) for running repeatable conversational-agent evaluations against HTTP, WebSocket, and local-harness endpoints. The runtime contract is documented in `/Users/ntindle/code/agpt/AgentProbe/AGENTS.md:7-12` and `/Users/ntindle/code/agpt/AgentProbe/docs/ARCHITECTURE.md:10-26`. The architecture follows a layered domain model: `types -> config -> sdk/providers -> repositories -> services -> runtime -> cli`. The CLI entry point is `/Users/ntindle/code/agpt/AgentProbe/src/cli/main.ts` and the run loop lives in `/Users/ntindle/code/agpt/AgentProbe/src/domains/evaluation/run-suite.ts`. Persisted run history is keyed in SQLite for local dev and Postgres for the hosted "server" mode; schema is in `/Users/ntindle/code/agpt/AgentProbe/src/providers/persistence/drizzle/postgres-schema.ts`. There is no LICENSE file at the repo root (`find . -maxdepth 2 -iname "LICENSE*"` returns nothing); the GitHub remote is `Significant-Gravitas/AgentProbe`, same org as AutoGPT.

The mental model is "one harness, many adapters." Scenarios are YAML; personas are YAML; rubrics are YAML; an endpoint adapter wires the SUT (system-under-test) to HTTP/WS/OpenClaw; an LLM persona drives turns; an LLM judge scores transcripts against the rubric. Runs are persisted (turns, target_events, tool_calls, checkpoints, judge_dimension_scores, human_dimension_scores) so the React dashboard can replay them. There is no purpose-built retrieval/recall scorer — memory evals run through the same transcript-judge pipeline as everything else.

## 2. Eval inventory

| Path | What it is | Notes |
| --- | --- | --- |
| `src/domains/evaluation/run-suite.ts` | Suite runner: prepares scenarios, manages sessions, runs persona+adapter, emits transcripts, calls judge | The execution heart. `runScenario` (lines 553-1059) handles sessions, `fresh_agent` resets, checkpoint assertions, max-turn caps, harness-vs-agent failure classification |
| `src/domains/evaluation/judge.ts` | LLM judge over a rubric | Single-model OpenAI Responses call, JSON-schema constrained output, prompt-cache key (lines 200-214), 3-attempt retry. Multi-judge is config-declared but **not implemented**: `judge.ts:316-319` throws if `judge.provider !== 'openai'`, no aggregation code |
| `src/domains/evaluation/simulator.ts` | LLM persona that drafts user turns | Default model `moonshotai/kimi-k2.6`, structured-output schema for `{message}` or `{status, message}` |
| `src/domains/evaluation/ports.ts` | `EndpointAdapter`, `LlmResponsesClient` interfaces | Where SUT integration plugs in |
| `src/domains/validation/load-suite.ts` (1210 lines) | YAML→typed-domain parser for scenarios/personas/rubrics/endpoints | Strict boundary validation. Implements `parseTimeOffset` for session `time_offset` strings |
| `src/domains/reporting/render-report.ts` + `dashboard.ts` | HTML/JSON report and live dashboard server | |
| `src/providers/persistence/drizzle/postgres-schema.ts:115-202` | `turns` and `target_events` carry `latencyMs` and `usageJson` per exchange | This is the cost/latency raw material |
| `src/providers/persistence/drizzle/postgres-schema.ts:203-242` | `judge_dimension_scores` + `human_dimension_scores` | Per-dimension judge output and parallel human grading stream |
| `data/rubric.yaml` (958 lines) | All rubric definitions | Includes `product`, `multi-session-memory`, `memory-temporal`, `memory-abstention`, `memory-crossdomain`, `memory-compositional`, `memory-introspection`, `memory-hygiene` |
| `data/scenarios.yaml` | 5 hand-written reference scenarios (refund, flight rebooking, prompt injection, RAG pricing, escalation) | The "starter" suite |
| `data/baseline-scenarios.yaml` (3294 lines, 100 scenarios) | The "tasks-100" pack — broad founder-ops coverage | Generated, attaches fixtures from `data/fixtures/` |
| `data/adversarial-scenarios.yaml` (318 lines) | Prompt-injection / jailbreak scenarios | Single-turn red-team prompts |
| `data/multi-session-memory.yaml` (1762 lines, ~30 scenarios) | **The memory pack** — 12 categories, multi-session, `fresh_agent` resets, named failure modes per scenario | Closest existing analog to dream P0.6 |
| `data/personas.yaml` | Persona library (`frustrated-customer`, `smb-founder`, `business-traveler`, etc.) | Drives the simulator |
| `data/fixtures/` (237 files) + `data/fixture-manifest.json` | Scenario-attached files (CSVs, JSONs, PDFs, etc.) | LLM-generated by `scripts/generate-fixtures-llm.ts` |
| `scripts/seed-test-scores.ts` | Synthetic score seeding for dashboard testing | |
| `scripts/generate-tasks-100-scenarios.ts` | Generator for the 100-scenario baseline | |
| `tests/unit/judge.test.ts` | Unit tests for the judge | |
| `tests/unit/runner.test.ts` | Unit tests for `runScenario` | |
| `tests/e2e/judge-transcript-order.e2e.test.ts` | Guards judge cache-friendliness | Verifies stable-prefix ordering for prompt-cache hits |

Grep for `eval|benchmark|judge|rubric|golden|score|metric|recall|precision|ndcg`: the meaningful code hits are exactly `domains/evaluation/*`, `data/rubric.yaml`, and the persistence schema's `*_dimension_scores` tables. Grep for `ndcg|mrr|precision_at|recall_at|hit_rate|golden_hits`: zero hits. **There is no quantitative retrieval-ranking scorer anywhere in the codebase.**

## 3. Memory-specific evals (deep dive)

`data/multi-session-memory.yaml` is the only memory-specific artifact, but it is substantial. Top-level structure (lines 1-42): version, suite id, `defaults` (persona `smb-founder`, rubric `multi-session-memory`, `user_name: "Jordan Rivera"`, optional `copilot_mode`). The pack documents its own format extensions: `sessions:` replaces `turns:`, each session has `id`, `time_offset` (logical, not wall-clock), `reset` policy, and optional per-session `max_turns`.

Reset semantics (from comments at `data/multi-session-memory.yaml:24-30` and enforced in `run-suite.ts:49,693-718`):

- `none` — continue same process/context
- `new` — clear context window, same process/identity
- `fresh_agent` — **fully cold boot**: new process, empty context window; the memory backend is the *only* persistent channel. The runner enforces this by closing the adapter and asking the `adapterFactory` for a new one (`run-suite.ts:703-712`). If no factory is provided, it logs a warning that results may be invalid — the spec note in the YAML is explicit that "most scenarios test nothing if `fresh_agent` is faked."

The 12 categories (`data/multi-session-memory.yaml`, search `# ====`):

1. **Retention** — `mem-retention-basic-identity` (line 48), `mem-retention-incidental-facts` (line 90). Floor test: declared facts from S1 surface in S2 without restating.
2. **Distilled procedural knowledge** — `mem-distill-authed-http-image-gen` (157), `mem-distill-onboarding-workflow` (213), `mem-distill-weekly-report-format` (266), `mem-distill-implicit-tool-preferences` (324), `mem-distill-lead-cleaning-procedure` (374). Multi-step procedures survive a cold boot.
3. **Rigidity** — `mem-rigidity-email-tone-override` (428), `mem-rigidity-tool-migration` (485), `mem-rigidity-pricing-update` (538). Updated preferences replace, not average. Uses `memory-temporal` rubric with `auto_fail_conditions: temporal_reasoning < 2` (`rubric.yaml:622-625`).
4. **Abstention** — `mem-abstain-ambiguous-reference` (595), `mem-abstain-no-fabricated-preferences` (638).
5. **Temporal** — `mem-temporal-stale-team-member` (688), `mem-temporal-deprecated-procedure` (741).
6. **Continuation** — `mem-continuation-interrupted-task` (809), `mem-continuation-project-state` (856).
7. **Cross-domain** — five scenarios (913, 961, 1018, 1071, 1125) including a Notion-rate-limit scenario.
8. **Procedural update** — `mem-procupdate-clean-replacement` (1179), `mem-procupdate-additive` (1244).
9. **Compositional** — `mem-compositional-board-prep` (1305). Two facts seeded in separate sessions, derived answer asked in S3. The tester note (`yaml:1356-1360`) explicitly flags this as "the test that distinguishes graph memory from flat stores" — directly relevant to a Graphiti-backed dream system.
10. **Introspection** — `mem-introspection-what-do-you-know` (1366), `mem-introspection-gaps` (1427).
11. **Long-tail** — `mem-longtail-lawyer-recall` (1477).
12. **Hygiene** — `mem-hygiene-bounded-time` (1553), `mem-hygiene-temporary-status` (1610). Time-bounded constraints; `memory-hygiene` rubric auto-fails on `constraint_expiry < 2` (`rubric.yaml:955-958`).
13. **Negative tests** — `mem-negative-one-off-qualifier` (1671), `mem-negative-forget-on-request` (1710). The "forget on request" scenario (line 1710) is *directly* a demotion-correctness probe: seed a budget figure in S1, user says "forget that" in S2, ask in S3 — failure modes include `forget_acknowledged_but_leaked: returns $50K with a note that it was asked to forget (proves forget didn't clear storage — worst failure)`.

Every scenario has a `failure_modes:` list naming distinct ways to fail (e.g., `fabrication`, `partial`, `exposed_seam`, `dropped_empty_company`). The judge JSON schema (`judge.ts:163-173`) carries `failure_mode_detected` as a free-form string output so the rubric meta-prompt can ask the judge to pick a named failure mode from the scenario. This is a nice pattern for diagnostic-level scoring rather than pure pass/fail.

Six memory rubrics live in `data/rubric.yaml:483-958`:

- `multi-session-memory` (line 483): four dimensions weighted 0.35/0.30/0.20/0.15 — `retention_accuracy`, `fabrication_detection`, `context_separation`, `temporal_reasoning`. Pass threshold 0.65. Judge is `anthropic/claude-opus-4.6` via OpenRouter, temperature 0.
- `memory-temporal` (581): reuses dimension anchors via YAML anchors (`*scale_retention_accuracy` etc.) — auto-fails on temporal_reasoning < 2.
- `memory-abstention` (628): adds `clarification_behavior` (weight 0.40).
- `memory-crossdomain` (703): adds `derived_answer_accuracy` (weight 0.25).
- `memory-compositional` (767): `inferential_synthesis` weighted 0.40, auto-fails below 2.
- `memory-introspection` (823): adds `summary_structure` and `gap_accuracy`. Auto-fails on `fabrication_detection < 2`.
- `memory-hygiene` (905): `constraint_expiry` weight 0.35 + auto-fail.

The meta_prompt for each memory rubric (e.g. `rubric.yaml:488-498`) is templated with Nunjucks-style `{{ expectations.expected_behavior }}` and a per-scenario `failure_modes` loop — so the judge is told both what good looks like *and* what each named failure mode looks like.

## 4. Eval runner architecture

Entry point: `src/cli/main.ts:235-303` (`handleRun`). It loads endpoint, scenarios, personas, rubric paths from `--endpoint --scenarios --personas --rubric`, builds an `OpenAiResponsesClient` against OpenRouter, builds a `SqliteRunRecorder`, and calls `runSuite` (`src/domains/evaluation/run-suite.ts:1073-1469`).

Inside `runSuite`:

1. Parse each YAML through `parseEndpointsYaml`/`parseScenariosInput`/`parsePersonaYaml`/`parseRubricsYaml` (`load-suite.ts`). Strict validation: missing persona or rubric references throw `AgentProbeConfigError` (`run-suite.ts:1197-1217`).
2. Filter scenarios by `--scenario` (ID/name) or `--tags`.
3. Build `PreparedRun` objects with a pinned `userId` (random UUID) per scenario iteration — critical so memory backends can isolate test data per scenario. The auth flow forges a JWT for AutoGPT in `providers/sdk/autogpt-auth.ts` so each user is a real persisted user.
4. For each prepared run, instantiate an adapter via `adapterFactory(endpointConfig, {userId, userName, baseUrlOverride, ...})` and call `runScenario`.
5. `runScenario` iterates `sessions` (one or more). On session boundaries with `fresh_agent`, it closes the adapter, calls `adapterFactory()` again, and resets transcript/sessionState (`run-suite.ts:693-718`). It injects a system `--- Session boundary: ... ---` turn so the judge transcript shows the discontinuity (`run-suite.ts:720-735`).
6. For each turn: if `use_exact_message: true`, send the literal string; otherwise let the persona-simulator LLM generate the next user turn given the transcript and persona instructions (`simulator.ts:generatePersonaStep`).
7. Checkpoint turns evaluate `tool_called`/`with_args`/`response_mentions`/`response_contains_any`/`response_must_not_contain` against the most recent assistant reply (`run-suite.ts:280-344`). These are not LLM-judged — they're regex/equality checks.
8. After all sessions, render the rubric templates with the run context (transcript, expectations, scenario, persona) using `renderTemplate` (Nunjucks), format the transcript with tool-call interleaving (`run-suite.ts:360-401`), and call `judgeResponse`.
9. `judgeResponse` (`judge.ts:303-421`) builds a JSON-schema-constrained Responses-API call with a deterministic `promptCacheKey` (`judge.ts:200-214`) for prompt caching. Anthropic models also get `cacheControl: { type: "ephemeral" }`. Three retries with exponential-ish backoff; auth errors fail fast.
10. Per-dimension scores are recorded via `recordJudgeResult` and turn-level latency/usage via `recordAssistantReply` (`run-suite.ts:138-145` + persistence layer).

**Input formats:**

- Endpoint YAML: example at `data/autogpt-endpoint.yaml` — preset name, transport, connection (base_url + timeout), auth type, named endpoints with Nunjucks-templated `body_template`, session lifecycle, response format (`sse|json`), JSONPath for `content_path` and session-id extraction.
- Scenario YAML: `data/scenarios.yaml` for single-session, `data/multi-session-memory.yaml` for multi-session. Top-level `version`, `defaults`, `scenarios: [...]`. Each scenario carries `id`, `name`, `tags`, `persona`, `rubric`, `priority`, `context`, `turns` or `sessions`, `expectations`.
- Persona YAML: `data/personas.yaml` — demographics, personality, behavior, optional `model` override.
- Rubric YAML: `data/rubric.yaml` — `judge` config + `rubrics: [...]` with `dimensions: [{ id, name, weight, scale, judge_prompt }]`, `meta_prompt`, `pass_threshold`, `scoring_overrides.auto_fail_conditions`. Nunjucks templates evaluated at run time.

**Scoring math** (`run-suite.ts:403-432`): per dimension, `normalized = raw / scale.points`. Weighted sum / total weight = `computedScore`. Pass if `computedScore >= rubric.passThreshold` AND no auto-fail conditions trip.

**Persistence** (`postgres-schema.ts`): runs → scenario_runs → turns/target_events/tool_calls/checkpoints/judge_dimension_scores. `turns.latency_ms` and `turns.usage_json` exist but in `runScenario` they're recorded via `recordTurn` only as part of `recordAssistantReply` (`run-suite.ts:672-676`). The `target_events` table is per-exchange and carries `usage_json` (token counts) and `latency_ms` — this is where cost-per-scenario lives. `judge_dimension_scores.reasoning` + `evidence_json` give you a per-dimension audit trail.

**Adding a new eval** is purely a data exercise: drop a scenario YAML in `data/`, drop a rubric YAML, run `bun run agentprobe run --endpoint ... --scenarios path/to.yaml --personas data/personas.yaml --rubric data/rubric.yaml`. There is *no plugin or extension hook* for custom scorers — everything goes through the same LLM-judge pipeline. The only assertion path that bypasses the LLM is checkpoint turns.

## 5. LLM-judge patterns

- **Model**: default judge is OpenRouter-routed `anthropic/claude-opus-4.6`, temperature 0, max_tokens 4096 (`data/rubric.yaml:6-22`). Provider abstraction in `src/providers/sdk/openai-responses.ts` — uses OpenAI's Responses API style.
- **System instructions**: hardcoded constant at `judge.ts:13-14`: `"You are an expert rubric judge. Evaluate only the provided response using the supplied evaluation context."` Per-rubric instructions go into the user-role evaluation context built by `judgeEvaluationContext` (`judge.ts:186-198`).
- **Structured output**: `text.format: { type: "json_schema", strict: true }` with a per-rubric schema built from the dimension list (`judge.ts:115-184`). Each dimension demands `{ reasoning, evidence[], score }`. Top-level adds `overall_notes`, `pass`, `failure_kind ∈ {agent, harness}`, `failure_mode_detected: string|null`.
- **Validation**: `validateRubricScore` (`judge.ts:284-301`) checks the returned dimension IDs are exactly the rubric's dimension IDs — no extras, no missing. Retries on shape mismatch.
- **Cache control**: `promptCacheKey` is a SHA-256 over `rubric.id + evaluationContext + schema`, truncated to 16 hex chars (`judge.ts:200-214`). For Anthropic models it also passes `cacheControl: { type: "ephemeral" }`. Prompt prefixes are kept stable (rubric context first, then per-run transcript) so cache hit rates are high; an e2e test guards this (`tests/e2e/judge-transcript-order.e2e.test.ts`).
- **Bias mitigation**: `data/rubric.yaml:12-18` declares `randomize_order`, `chain_of_thought`, `structured_output`, `multiple_judges: false`, `judge_count: 3`, `aggregation: median`. **None of this is wired through to `judge.ts` except `structured_output`.** Multi-judge aggregation is config-only — implementation gap.
- **Failure-mode tagging**: the `failure_mode_detected` field is populated from the per-scenario `failure_modes:` list, surfaced into the meta_prompt template (`rubric.yaml:495-498`). This gives diagnostic granularity beyond pass/fail without needing code changes.

## 6. Golden datasets

There is no separate "golden dataset" notion. The closest things are:

1. **Scenario `expectations.ground_truth`** — free-form text used inside meta_prompt templates. Examples: `data/scenarios.yaml:167-171` (RAG pricing), `data/multi-session-memory.yaml:1411-1415` (introspection summary), `data/multi-session-memory.yaml:1108` (cross-domain). Not structured, not indexed; it's just a string the judge reads.
2. **`expectations.must_include / must_not_include`** — string-containment lists. Checked... actually nowhere automatically. They're documented in the YAML and the judge sees them via templated prompts, but there is no scorer that asserts `assistantText.includes(must_include[i])` outside of checkpoint turns.
3. **`data/fixtures/`** — 237 files (`ls data/fixtures | wc -l`) attached to scenarios as conversation attachments. Manifest at `data/fixture-manifest.json`. Generated by `scripts/generate-fixtures-llm.ts` using an OpenRouter model. Used as attachments via `currentAdapter.uploadFile(...)` (`run-suite.ts:854-878`). Not "golden" in the eval sense — they're inputs.

Versioning is git + YAML. No semver or dataset-id tagging. There is no concept of "expected memory hits" or "expected retrieval set" in the data model.

## 7. Reusability matrix

| Thing | Verdict | Why |
| --- | --- | --- |
| Multi-session scenario format (`sessions:` + `fresh_agent` reset + `time_offset`) | **Lift** | Already designed for the exact problem space. The `fresh_agent` semantics in particular are the only way to verify memory backend isolation. Reuse the YAML schema. |
| Memory rubrics (`multi-session-memory`, `memory-temporal`, `memory-abstention`, `memory-crossdomain`, `memory-compositional`, `memory-introspection`, `memory-hygiene`) | **Lift** | Cover seven of the failure modes P0.6 cares about. Pass thresholds and weights are reasonable starting points. |
| The ~30 memory scenarios in `data/multi-session-memory.yaml` | **Lift** | This is months of curation work we don't need to redo. In particular: `mem-negative-forget-on-request` is a forget-precision scenario for free; `mem-compositional-board-prep` is the graph-vs-flat differentiator the YAML comment explicitly calls out. |
| Named `failure_modes:` per scenario + `failure_mode_detected` judge field | **Lift** | Cleaner than free-form failure notes. Gives diagnostic-level reporting and stable taxonomies. |
| LLM-judge pattern (JSON-schema strict, prompt-cache key, dimension-list validation, 3-retry) | **Learn-from** | The pattern is solid but it's TS and lives in AgentProbe's runtime. We'll need a Python implementation in `autogpt_platform/backend/copilot/dream/eval/`. Port the rubric structure and the JSON schema shape; don't import the TS code. |
| Persistence schema for judge_dimension_scores + per-turn latency/usage | **Learn-from** | The shape (per-dimension `{ raw_score, normalized_score, reasoning, evidence_json }`) and the parallel `human_dimension_scores` table are good models for how we record P0.6 results in our own Postgres. Don't share the schema; copy the design. |
| Persona simulator (`simulator.ts`) | **Skip** | Memory benchmarks for dream are not conversational — they're "given a warm-context query, did the right memories surface?" and "was this demotion correct?". We don't need a persona LLM to drive turns. |
| Checkpoint assertions (regex/equality on assistant reply) | **Learn-from** | Useful pattern for the warm-context retrieval relevance eval: we want hard assertions on "did memory X appear in the returned set?" not LLM judgments. Adopt the *idea* of inline non-LLM checks, write a Python version. |
| `EndpointAdapter` / `runSuite` orchestration | **Skip** | We're not exposing dream as an HTTP endpoint to be probed; we're evaluating it via direct service calls. The adapter layer would be wasted overhead. |
| AutoGPT JWT auth helper (`providers/sdk/autogpt-auth.ts`) | **Learn-from** | If we ever want AgentProbe to *run against* our backend (e.g. to drive the memory pack scenarios end-to-end), this is the integration. Out of scope for P0.6 in-process eval. |
| Dashboard / report renderer | **Skip** | We have Langfuse + our own backend metrics; another dashboard is overhead. |
| YAML loaders (`load-suite.ts`, 1210 lines) | **Learn-from** | Demonstrates how to do strict YAML→domain boundary validation. We can rewrite in Python with Pydantic for the subset we lift. |
| `bias_mitigation.multiple_judges` config | **Skip** | Not implemented in `judge.ts`. Either build it ourselves or rely on single-judge runs at higher temperature/sample-count. |
| NDCG/MRR/precision@k retrieval scoring | **Build ourselves** | Doesn't exist in AgentProbe at all. |
| Ratification-rate calculation | **Build ourselves** | Doesn't exist. The closest concept is `memory-hygiene.constraint_expiry`, but it scores agent behavior not memory lifecycle. |
| Cost-per-run aggregation | **Lift the schema, build the rollup** | `target_events.usage_json` carries per-call token counts; AgentProbe doesn't aggregate to "$ per scenario." We'd need that for dream pass cost reporting anyway. |
| LLM-generated fixtures (`scripts/generate-fixtures-llm.ts`) | **Skip** | The fixtures are conversation attachments for founder-ops scenarios. We want curated ground-truth web-fact-check examples, which is a different curation problem. |

## 8. Integration cost

**Option A — Vendor the YAML packs and rubrics, reimplement the runner in Python.** Cheapest. Copy `data/multi-session-memory.yaml` and the memory rubrics from `data/rubric.yaml:483-958` into `autogpt_platform/backend/copilot/dream/eval/datasets/`. Write a Pydantic loader (~500 LOC), a judge runner using the existing `LiteLLM`/Anthropic SDK already in the platform, and a results recorder. Boundaries with the YAML format are well-documented (see `data/multi-session-memory.yaml:17-41` comments). License risk: the AgentProbe repo has no LICENSE file, but it's owned by `Significant-Gravitas` which is the same org as AutoGPT — copying internally should be cleared with whoever owns AgentProbe (probably swiftyos).

**Option B — Run AgentProbe as a subprocess from our backend.** Heavier. Adds Bun + TypeScript to the platform's dependency footprint (platform is Python + pnpm; AgentProbe is Bun + TS). Also wires through SQLite, an HTTP server we don't want, the dashboard React app, and Postgres migrations we don't want to colocate. Realistically only worth it if we want the dashboard. The `start-server` mode (`src/cli/main.ts:433`) means we could run AgentProbe as a sidecar service, but that's deploy complexity we don't need for P0.6.

**Option C — Import AgentProbe as a sibling repo, run from CI only.** Reasonable middle ground. CI checks out `Significant-Gravitas/AgentProbe`, runs `bun run agentprobe run --endpoint autogpt-endpoint.yaml --scenarios data/multi-session-memory.yaml ...` against a docker-compose'd autogpt_platform backend. We get the existing 30 scenarios and the judge plumbing for free, just orchestrated by CI. The autogpt JWT auth already exists (`providers/sdk/autogpt-auth.ts`) and `data/autogpt-endpoint.yaml` exists.

**Obstacles regardless:**

- AgentProbe has no LICENSE file at the repo root. Confirm with org before treating it as ours-to-fork.
- AgentProbe's runtime is Bun + TS; our backend is Python + Poetry. Either we accept polyglot (Option C) or we reimplement (Option A).
- AgentProbe's eval pipeline assumes a conversational SUT. Dream's eval needs to probe internal state (which memories surfaced, what was demoted) without going through a chat surface. The architecture mismatch means we can lift *scenarios and rubrics* easily, but the *runner* is wrong-shape for half of P0.6.
- AgentProbe uses OpenRouter for both persona and judge models. We'd likely use Anthropic direct or our existing Bedrock provider. Trivial port but worth noting.

Recommendation: **Option A for retrieval/demotion/ratification/cost/latency metrics; Option C for the conversational memory-pack scenarios** so we get to reuse all 30 scenarios without rewriting the runner.

## 9. Gaps for P0.6

What AgentProbe gives us:

- Conversational memory scenarios with `fresh_agent` resets — covers retention, distillation, rigidity, abstention, temporal, cross-domain, compositional, introspection, hygiene, long-tail, negative-test patterns
- LLM judge with structured output, prompt caching, dimension-level evidence + reasoning
- Per-scenario named failure modes for diagnostic reporting
- Per-turn latency and per-call token usage in the persistence schema
- Multi-session orchestration with cold-boot semantics

What AgentProbe **does not** give us and we have to build:

1. **Warm-context retrieval relevance (golden query → expected memory hits).** No code path expects "the retrieval system returned set R; expected set was G; score precision@k / recall@k / NDCG." AgentProbe scores the *final conversational response*, not the *retrieved memory set*. We need a Python harness that calls `fetch_warm_context()` directly with a curated query, compares the returned `MemoryEnvelope` IDs against a golden ID list, and emits standard IR metrics. There's no analog in AgentProbe.

2. **Demotion precision (was this `superseded` flip correct?).** No concept in AgentProbe. The closest scenarios (`mem-negative-forget-on-request`, the rigidity pack) test *behavior after demotion*, not the demotion decision itself. We need a labeled set of (memory_before, demotion_event, ground_truth_correct?) triples, then a scorer over the dream-pass logs. Either human label, LLM judge, or both — the AgentProbe per-dimension judge pattern is the right shape but the dataset doesn't exist.

3. **Ratification rate (% of `tentative` flipped to `active` within 7d).** This is an aggregation over our production telemetry, not a benchmark in the AgentProbe sense. Built in `backend/copilot/monitoring/instrumentation.py` per the P0 spec, not in eval. AgentProbe contributes nothing here. We need: a Redis hash `mem:hits:{user_id}` (per `p0-spec.md:336`), a daily rollup, and a baseline + alert.

4. **Web-fact-check P/R against curated ground truth.** Needs a curated set of (claim, current_truth) pairs. The "old but stable" vs "stale and wrong" split called out in `p0-spec.md:260` is the dataset — but it doesn't exist yet. AgentProbe has zero web-fact-check scenarios or rubrics. The judge-rubric pattern transfers; the data doesn't.

5. **Cost per dream pass.** AgentProbe persists per-call token counts in `target_events.usage_json` and per-call latency in `target_events.latency_ms`, but it never aggregates these to a $-per-scenario or $-per-run number. We need a rollup that joins `target_events` to a price table per model. We can copy the schema shape; we have to write the rollup.

6. **p50/p95 latency.** Same shape: AgentProbe has the raw per-call latency. Aggregation to percentiles per scenario / per pass / per phase is something we build. Not hard, but not free.

7. **Graphiti-specific retrieval evals.** No mention of Graphiti, FalkorDB, episodes, or graph traversal anywhere in AgentProbe (`grep -i graphiti|falkordb|episode src tests data` returns zero hits in source). The compositional scenario explicitly *anticipates* graph vs flat differentiation in its tester note (`data/multi-session-memory.yaml:1356-1360`) but the eval scores agent output, not graph structure. Anything probing "did the graph return the right entity?" or "did 1-hop demotion stay 1-hop?" we write from scratch.

8. **Provenance / `dream:web_verified:{ts}:{url}` checking.** AgentProbe has no scenario that asserts on memory metadata. Our memory envelope provenance check is a unit-test/assertion problem, not a benchmark problem.

9. **Repro determinism for graph state.** AgentProbe scenarios use `crypto.randomUUID()` for the pinned user_id per iteration (`run-suite.ts:1220`), which is good for isolation but means every run starts from a clean slate. For Graphiti-backed evals we may want to *seed* a memory graph from a fixture and then run the warm-context probe against it. That's a fixture format AgentProbe doesn't have.

10. **CI integration.** AgentProbe runs locally with `bun run test`. To run the memory pack on every PR we need either docker-compose orchestration (boot autogpt_platform, point AgentProbe at it, run scenarios, collect SQLite results) or the reimplementation path. Either way, the wiring is ours.

Bottom line: **lift the conversational scenarios and rubrics and the judge pattern; build the quantitative retrieval/demotion/ratification scorers ourselves.** P0.6 ends up being ~60% conversational pack (AgentProbe) + ~40% in-process IR scorers, telemetry rollups, and Graphiti-aware probes (us). The conversational half is the high-effort curation work — handing that to us for free is the big win. The IR-scoring half is mostly straightforward Python with NumPy and existing platform instrumentation.
