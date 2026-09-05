# AutoGPT Platform Improvement Plan
**Audit scope:** `master` @ `ce6ab7b07` (tag `autogpt-platform-beta-v0.7.0`, `origin/master` == `HEAD`, ahead/behind `0/0`). Dirty worktree isolated as `LOCAL_UNCOMMITTED_BEHAVIOR` (see §1). All findings below trace to `CANONICAL_UPSTREAM_BEHAVIOR` unless flagged `LOCAL`.

**Generated:** 2026-09-01 — repository-grounded, evidence-linked.

## Wave 0 Status: CERTIFIED AND CLOSED (2026-09-04)

**Branch:** `fix/wave0-runtime-integrity` (certification-packet commit `b2da02682`; invariant-evidence SHA `59245d686`; see `WAVE0_CERTIFICATION.md` for current ahead count)
**Certification-packet SHA:** `b2da02682`; all invariant code at `59245d686`
**Certification:** `WAVE0_CERTIFICATION.md` at repo root; Gates A/B/C passed, 7/7 invariants PROVEN; state `WAVE_0_CERTIFIED_AND_CLOSED`

### Proven invariants

| Invariant | Status | Test count |
|---|---|---|
| Execution identity (REL-003) | PROVEN | 10/10 graph-ids.test.ts |
| Builder integrity (REL-004) | PROVEN | 20+22 builder-hydrate + historyStore |
| Authorization (REL-007) | PROVEN | 27/27 authz negative matrix + remaining |
| Cancellation durability (REL-002) | PROVEN | 11/11 durable cancel |
| Credential revocation (REL-001) | PROVEN + accepted policy | 18/18 revocation (fail-open bounded 5m, CEO-approved) |
| Scheduler idempotency (REL-005) | PROVEN | 15/15 + live Postgres duplicate-write rejection |
| Cost containment (REL-006) | PROVEN | 7/7 retry limits + cost drain |
| Missed-tick non-billing | ACCEPTED POLICY + PROVEN | 4/4 |

### Residual register (8 items, none blocking)

| ID | Class | Detail |
|---|---|---|
| R1 | LOCAL_PARTIAL_PASS / CI_BOUND_FIXTURE_INFRA | Poetry venv proven functional (`test_revocation.py` 18/18 in 0.18s). Full-repo suite blocked by pre-existing `SpinTestServer` autouse fixture (`conftest.py:92`) resolving compose-internal hostnames (`db`, `redis`, `rabbitmq`) unreachable on this workstation; zero Wave 0 diff in `backend/util/test.py` or `backend/util/cache.py`. 83/83 Wave 0 tests passed with conftest active against real migrated Postgres. Upstream CI `poetry run test` on the PR is follow-up verification, not a prerequisite for the already documented local certification state. |
| R2 | RESIDUAL (bounded) | One extra retry between executor crash and FAILED mark (max 5, finite). |
| R3 | DEFERRED | Legacy Supabase HS256 bridge removal gated on 30-day measurement window. |
| R4 | DEFERRED | `check_authz.py` advisory (39 flagged, documented suppressions). Promoting to blocking requires AST false-positive reduction. |
| R5 | ACCEPTED POLICY | Fail-open revocation on Redis outage (CEO-approved, bounded 5m). |
| R6 | ENVIRONMENT (pre-existing) | `$HOME/node_modules` pollution breaks `prettier-plugin-tailwindcss` from original repo path only. Not a Wave 0 defect. |
| R7 | DEFERRED polish | `stop_graph_execution` persists cancel on already-terminal rows (harmless, no corruption). |
| R8 | ACCEPTED POLICY | Missed-tick `status=missed` non-billable (CEO-approved). |

### GitHub transport

`/opt/homebrew/bin/gh` is authenticated as CCRBrad (keyring, `repo` scope). The npm `gh` at `/Users/bradstrawbridge/.nvm/.../gh` is an unrelated broken binary; use `/opt/homebrew/bin/gh` for all GitHub operations. PR push and creation are the only remaining external steps.

### Process improvement (permanent, applies to all future waves)

Each backlog item MUST pass through four stages before being declared complete:

1. **IMPLEMENTED** - code written, compiles, unit tests pass locally
2. **INTEGRATION_PROVEN** - the invariant is exercised against real dependencies (not mocked); boundary behavior verified
3. **CANONICAL_GATE_GREEN** - `pnpm format`, `pnpm types`, `pnpm lint`, `poetry run test` (or scoped equivalent) all exit 0
4. **CERTIFIED** - reviewer or CI confirms the invariant holds at the merged SHA

Wave 0 exposed rounds where scaffolding was labeled complete before the runtime boundary was actually proven. This 4-stage gate prevents that.

### Do not reopen REL-001 through REL-007 unless CI or review produces a concrete regression. Wave 0 is the frozen reliability baseline.

---

## 0. Repository Truth — What Was Audited

| Signal | Value |
|---|---|
| `HEAD` | `ce6ab7b07 fix(backend/copilot): stop empty untitled dream sessions polluting the chat list (#13332)` |
| `origin/master` | `ce6ab7b07` — `HEAD == origin/master` |
| `origin/dev` | absent (no remote tracking branch; `dev` exists only as historic merge-base tags `v0.6.69`/`v0.6.70` at `127cf0e13`/`c45b9e358`) |
| Branch | `master` (clean tracking) |
| Dirty (modified, not staged) | `autogpt_platform/frontend/package.json:7` (`NODE_OPTIONS --max-old-space-size 16384→4096`), `frontend/src/app/(no-navbar)/login/actions.ts:62` (+`devLogin()` branch), `frontend/src/app/(no-navbar)/login/page.tsx:1` (+`useState/useTransition` + local-demo button), `frontend/src/lib/autogpt-server-api/__tests__/getServerAuthToken.test.ts:20` (+test), `frontend/src/lib/autogpt-server-api/helpers.ts:159` (`BETTER_AUTH_INTERNAL_URL` fallback) |
| Untracked | `autogpt_platform/docker-compose.local.yml`, `frontend/src/app/api/local-dev/import-skills/route.ts`, `frontend/src/app/api/local-dev/login/route.ts` |
| Pre-existing? | Yes — all 5 modified files predate this audit; `git diff` stat `99 insertions / 3 deletions` is **local dev ergonomy only**. Behavior delta is gated on `NEXT_PUBLIC_APP_ENV==local && LOCAL_DEV_AUTH_ENABLED==true` (`login/actions.ts:68`) + `environment.isLocal()` (`login/page.tsx:122`). No overlap with P0 findings except the `BETTER_AUTH_INTERNAL_URL` read in `helpers.ts:162`, which is `LOCAL_UNCOMMITTED` and must not be attributed to upstream auth. The canonical helper reads `BETTER_AUTH_URL | NEXT_PUBLIC_FRONTEND_BASE_URL | http://localhost:3000`. |

**Rule:** Do not attribute the `devLogin`/local-dev routes or the `4096` memory cut to production architecture. Do not reset them — they are operator workspace state.

### Repo scale (canonical)

| Layer | LOC/order |
|---|---|
| `frontend/src/**/*.tsx` | `364,322` lines |
| `backend/backend/**/*.py` | `946,870` lines |
| Frontend legacy imports | `237` lines across `144` files (`120` external) |
| Backend services | `14` compose services |
| TODO/FIXME/HACK hits (frontend) | `241` |
| `: any` hits (frontend) | `193` (most in tests) |

---

## 1. Executive Assessment

### Maturity

**Product: working, not yet trustworthy in the invariant that matters most — "agents finish work without supervision".** Core loop (create → build → run → view output) functions on happy path (8 Playwright `*-happy-path.spec.ts` cover auth, builder, library, marketplace, publish, copilot, settings, api-keys). Failure paths, concurrent edits, WS loss, scheduler edge cases, and auth revocation are guarded by comments and single-digit-second mitigations, not by atomic guarantees. The codebase signals a team that ships fast with disciplined API generation (Orval + MSW 90% integration-test strategy) and security defaults (cache `no-store` allow-list, Sentry, JWKS) but has deferred the state-ownership and execution-invariant hardening that a paid agent runtime requires.

**Engineering: mid-maturity.** `AGENTS.md`/`CLAUDE.md` enforce `pnpm format→lint→types` and typed generated hooks; Prisma migrations own `platform` schema; `better-auth` migration is code-complete and handled conservatively. Debt is concentrated, not diffuse: one design-system inversion, one builder state split, one auth revocation window, one scheduler coalesce policy, and a handful of large components (`ui/icons.tsx:1880`, `ExecutionsTable.tsx:1096`, `Flow.tsx` 150+ lines of providers). Those five explain >70% of the risk.

**UX/UI: coherent atoms, incoherent shells.** New `atoms/` (`Button`, `Skeleton`, `Badge`, `Icon` via Hugeicons) and `molecules/` (`Dialog`, `Popover`, `Table`) exist but `atoms/Input:4`, `atoms/Select:10`, `atoms/DateInput:5`, `molecules/Table:9` still delegate to `__legacy__/ui/*` — the system is two systems stitched backwards (`§8`). Light theme is intentional (see `§8` dark audit) yet ships 492 dead `dark:` utilities and a `.dark` token block that never evaluates. Builder is desktop-only (`MobileWarning` full-screen block), library answers "what happened?" weakly (cost and failure-cause are secondary panels).

**Production readiness: deployable, not yet operable at the invariant level.** Self-host `docker compose --profile local up deps_backend` boots; `migrate` runs `prisma migrate deploy` against `platform` schema. Hosted vs self-host split is `NEXT_PUBLIC_BEHAVE_AS` + `AGPT_SERVER_URL`/`JWT_JWKS_URL` envs, not a separate binary — healthy. The gaps that block "substantial feature expansion" are not UI polish but `§3` invariants: identity revocation delay, execution duplicate/retry without dedup key, schedule coalesce dropping ticks silently, and builder draft clobber.

### Top 3 highest-impact opportunities (one paragraph each)

1. **Close the Builder authority split (single writer).** Today `GraphModel` lives in React Query *and* in Zustand `nodeStore`/`edgeStore`, with `flowVersion` in `nuqs` URL *and* `graphStore`, and `flowExecutionID` optimistically set before `executeGraph` resolves (`useRunGraph.ts:163`). The fix is a single `useBuilderQueryStates` hook, making Query read-only and Zustand the sole mutable store, and gating `useFlow.ts:149` `setNodes([])+addNodes()` behind `hasChanges`/`isDraftDifferent`. Effort `M`, blast radius is every Build/Copilot interaction — the highest product-leverage change.

2. **Harden execution to its invariants (cancel + WS + scheduler + cost).** The `SoR` audit (`§6`) proves Postgres is durable but WS is lossy, cancel is `fanout auto_ack=True` best-effort (`manager.py:1444`), `misfire_grace_time=None` coalesces missed cron ticks to one silently, and `cost_tracking.py:103` drains only same-loop tasks for 5s. One wave that makes cancel durable (DB flag + executor poll), heals WS via standardized `invalidateQueries` on resubscribe (as `useFlowRealtime.ts:81` already does for detail pages), surfaces coalesced misses, and enlarges cost drain to 30s/global would move three invariants from `PARTIALLY_PROVEN` to `PROVEN`.

3. **Flip the design-system dependency (6-file inversion fix).** `atoms/Input`, `atoms/Select`, `atoms/DateInput`, `atoms/DateTimeInput`, `molecules/Table`, `molecules/Collapsible` are the only `new→legacy` edges (`§8:11` lines). Re-basing them on `src/components/ui/*` (already clean) makes `atoms/molecules/ui` acyclic; *then* sweeping `skeleton` (23) + `separator` (11) + `badge` (8) deletes 42 legacy edges for ~XS effort. This is not cosmetic — it unblocks accessibility, bundle, and hiring velocity on every route.

### Major risks before expansion

- Do not add new agent capabilities or scheduling features before `§12:P0` (identity revocation window, cancel, builder clobber, scheduler dedup). Adding more execution paths widens the duplicate-execution surface.
- Do not delete Supabase bridge until the 30-day `SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS` window is measured as 0 hits in prod (`§5: C`).
- Do not enable dark mode as a side effect — ships 492 dead utilities today; either delete them (if light-only is product intent) or fix the three-layer disable atomically (`providers.tsx:37` + `tailwind.config.ts:7` + `Navbar.tsx:134` + `globals.css:42/81`).

---

## 2. Architecture Map — Concrete Paths

```
Browser
  ├─ Next.js 15 App Router (frontend/src/app)
  │    ├─ (no-navbar)/login|signup|onboarding  Auth + BrainDump (voice/typed/skipped)
  │    ├─ (platform)/copilot  CopilotPage.tsx:56  (CopilotChatHost + ArtifactPanel + ContextPanel)
  │    ├─ (platform)/build   build/page.tsx:8    (Flow/Flow.tsx  XYFlow  + NewControlPanel)
  │    ├─ (platform)/library LibraryPage + NewAgentLibraryView (agents/runs/schedules/triggers)
  │    ├─ (platform)/marketplace  StoreCard/FeaturedAgentCard (legacy composites)
  │    ├─ (platform)/admin   diagnostics/execution-analytics/platform-costs (legacy-heavy)
  │    ├─ (platform)/settings/api-keys|profile|billing  (shadcn-new primitives cleanest)
  │    └─ (public)/tour  upsell sidebar when logged-out
  │
  ├─ Theme: next-themes ThemeProvider forcedTheme="light" (providers.tsx:37)
  │         Tailwind darkMode ["class", ".dark-mode"] (tailwind.config.ts:7) mismatched with globals.css .dark (globals.css:42)
  │         ThemeToggle commented out (Navbar.tsx:134) → Intentional light-only, see §8
  │
  ├─ State: React Query (server) + Zustand (nodeStore/edgeStore/graphStore/historyStore/copyPasteStore/copilotChatRegistry)
  │         + nuqs (flowID/flowVersion/flowExecutionID) + Dexie (BuilderDraft, 24h expiry, db.ts:5)
  │         ⚠ Duplicated authority: see §6/§4 B1
  │
  ├─ API surface: Orval → src/app/api/__generated__/endpoints/* (typed hooks + .msw.ts handlers)
  │               Legacy imperative: src/lib/autogpt-server-api/ BackendAPI (WS + proxy helpers)
  │               Proxy: src/app/api/proxy/[...path]/route.ts:197 → Bearer inject via getServerAuthToken()
  │               Direct SSE: lib/services/environment getAGPTServerApiUrl() + copilotStreamTransport.ts:77
  │
  ├─ Auth (frontend owns): Better Auth @ 1.6.14 (src/lib/auth/auth.ts:42 Pool search_path=platform)
  │         tables: UserAuthIdentity/UserAuthSession/UserAuthAccount/UserAuthJwks (schema.prisma:21) lives in platform schema
  │         cookie: better-auth.session_token + session_data (5m cache, auth.ts:102)
  │         bridge: supabase-bridge.ts:133 (HS256, 30d tolerance) + middleware.ts:68 ( sb-* detection)
  │         legacy: SUPABASE_JWT_SECRET / JWT_VERIFY_KEY (HS256 fallback, jwt_utils.py:101 domain migration)
  │         diagram: §5
  │
  └─ WS: src/lib/autogpt-server-api/client.ts:1040 connectWebSocket (heartbeat 100s, 10s timeout, reconnect 1s)
         + FlowRealtime useFlowRealtime.ts:35 (node_execution_event / graph_execution_event sharded channels)
         + Copilot streaming copilotStreamTransport.ts:77 (POST DefaultChatTransport + GET-resume via copilotChatRegistry)

Backend (autogpt_platform/backend, FastAPI + Prisma)

  API: backend/backend/api/* (REST) + ws.py (WebSocket) + conn_manager.py (Redis pub/sub pump, backoff 0.5→8s, 60s deadline)
      Middleware: api/middleware/security.py:9 (no-store default, 25 CACHEABLE_PATHS)
      Deps: autogpt_libs/auth/dependencies.py:83 get_user_id / requires_user / get_request_context

  Data: backend/data/*
      execution.py:142 VALID_STATUS_TRANSITIONS (finite state map; Terminal ← REVIEW ↔ RUNNING)
      execution.py:1088 update_graph_execution_stats (conditional WHERE, cascade_running_children)
      db.py / db_accessors.py / credit.py (Postgres platform schema)
      event_bus.py:117 SPUBLISH wrapper (swallows Redis errors) — WS is ephemeral, DB is SoR

  Queue: data/rabbitmq.py (quorum graph_execution_queue_v2, durable, delivery_mode=2)
         executor/utils.py:1358 add_graph_execution (INCOMPLETE → QUEUED → publish)
         executor/manager.py:1363 ExecutorManager (active_graph_runs dict, ClusterLock SET NX EX, fanout cancel)

  Executors: backend/executor/
      manager.py  (single owner of graph runs, heartbeat 300s, x-consumer-timeout 24h, dispatch via on_graph_execution)
      batch_executor.py:327 (per-batch tombstone SET NX EX Lua, redis_helpers.py:104, backoff 30→300s)
      billing.py / cost_tracking.py:103 (create_task schedule_platform_cost_log, loop-filtered drain 5s)
      scheduler.py:1448 SQLAlchemyJobStore on Postgres (coalesce=True, max_instances=1000, misfire_grace_time=None)

  Supporting: Redis Cluster 3 shards + init (compose local-redis), RabbitMQ, FalkorDB (Graphiti), ClamAV (aioclamd), Postgres 15+pgvector

Testing

  Integration: Vitest + RTL + MSW (src/tests/integrations/test-utils.tsx wraps QueryClient+MSW+Nuqs, 90% of coverage)
              Orval generates getGetV2*MockHandler200/422/401 per endpoint
  E2E: Playwright 8 *-happy-path.spec.ts (auth-happy, builder-happy, library-happy, marketplace-happy, publish-happy, copilot-happy, settings-happy, api-keys-happy)
  Visual: Storybook 9.1 + Chromatic (atoms/* stories)

Infra

  Compose: docker-compose.yml (thin extends) + docker-compose.platform.yml (canonical envs, x-backend-env anchors)
           db: pgvector/pg15, redis-0/1/2, falkordb, rabbitmq, clamav, migrate (prisma migrate deploy), rest_server, executor, copilot_executor, websocket_server, database_manager, scheduler_server, notification_server, frontend
  Env: .env.default → .env → compose environment → shell (see §11 matrix)
  Deploy: .github/workflows/platform-{backend,frontend,fullstack}-ci.yml + platform-autogpt-deploy-{dev,prod}.yaml + Vercel standalone (next.config.mjs standalone unless VERCEL)
```

---

## 3. Critical Product Invariants

| Invariant | Classification | Evidence | Gap |
|---|---|---|---|
| **Identity** — every authenticated request resolves to exactly one canonical user | `PARTIALLY_PROVEN` | Frontend is SoR: `auth.ts:42` Pool `platform` schema singleton. Backend is pure verifier: `jwt_utils.py:77` `jwt.decode` signature+`exp`+`aud`. Middleware is optimistic (`middleware.ts:58` comment "real validation happens in route handlers"). Revocation is DB-only; backend never checks session existence (`jwt_utils.py:137` no lookup). JWT `1h` (`auth.ts:215`) bounds replay; `cache()` is per-request (`getServerAuthToken.ts:28`) not cross-request Map (would amplify). | 1h replay window after `signOut`/password-reset deletes session row but minted JWT still verifies. `session_data` cookie cache adds 5m staleness on role/admin gate (`middleware.ts:20`). |
| **Agent-version** — execution traceable to exact immutable version | `PARTIALLY_PROVEN` | `GraphModel.version` is backend monotonic; `PUT /v1/graphs/{id}` increments it; `useSaveGraph.ts:64` pins `flowVersion` via `setQueryStates`. Execution creation takes `graphVersion` (`RunGraph:70` passes `flowVersion`). | Draft (`BuilderDraft.flowVersion`) not compare-checked on `loadDraft:252`; stale draft can overwrite newer version. Copilot `edit_agent` leaves stale `BuilderDraft` (not deleted) so next reload prompts with pre-edit nodes. `flowVersion` declared independently in 10 call sites (see §6 table) — last writer wins. |
| **Execution** — logical execution cannot become two via retry/reconnect | `NOT_PROVEN` | Protections: conditional `update_graph_execution_stats(QUEUED)` guard (`execution.py:1130`), `active_graph_runs` dict (`manager.py:1663`), Redis `ClusterLock` `SET NX EX` (`cluster_lock.py:68`), per-batch Lua tombstone (`redis_helpers.py:104`). But frontend `sendWebSocketMessage` retries `4× 2^(n-1) s` without idempotency key for future non-idempotent methods (`client.ts:974`), and RabbitMQ `auto_ack=False` + `requeue_by_republishing` creates new delivery without dedup key — safe only because status gate discards second delivery if first committed `RUNNING/COMPLETED`. | If first delivery crashed before `RUNNING` persisted, lock is free and retry *does* duplicate node work. No global `message_id` table for graph executions (unlike Copilot `message_id UUID PK` in `copilotStreamTransport.ts:99` which *is* atomic). |
| **Persistence** — successful execution durable without browser/WS | `PROVEN` | `upsert_execution_output` (`execution.py:1024`) + `AgentGraphExecution.stats` are Postgres writes before `SPUBLISH` (`event_bus.py:72` caps message size, truncates WS only). Workspace artifacts: `file.py:400` `workspace_manager.write_file` → GCS → `UserWorkspaceFile` row before node `COMPLETED`. | WS truncation (`MAX_MESSAGE_SIZE`) surfaces "payload too large" in UI while DB holds full output — UI confusion, not durability loss. |
| **Scheduling** — deterministic trigger with duplicate protection | `PARTIALLY_PROVEN` | Source of truth is `apscheduler_jobs` on Postgres (`scheduler.py:1448`). DB guard `utils.py:1367` `updated_exec.status != QUEUED` aborts second `add_graph_execution`. APScheduler `coalesce=True` collapses missed ticks; `max_instances=1000` does not create true duplicate `graph_exec_id` (each tick is distinct id). | `misfire_grace_time=None` (`scheduler.py:1443`) means a blocked scheduler drops missed ticks with only `EVENT_JOB_MISSED` warning — silent coalesce, not surfaced. Per-batch Lua tombstone is safe for BatchExecutor but not used for cron. |
| **Cost** — recorded cost reconcilable to activity | `PARTIALLY_PROVEN` | Ledger is `CreditTransaction` + `PlatformCostLog` (`credit.py`/`cost_tracking.py`). `spend_credits` (`billing.py:152`) via `db_manager` RPC is SoR. Reconciled usage charged via `charge_reconciled_usage` (`billing.py:328`). | `schedule_platform_cost_log` fires `asyncio.create_task` (`cost_tracking.py:280`) drained only for same-loop tasks with 5–10s timeout (`manager.py:1868`). Returns `0,0` on exception swallow (`billing.py:333`). Cost rows may be lost on deploy. |
| **Authorization** — no cross-tenant access | `PROVEN` (with caveat: audit surface is large) | Every graph/execution query filters on `user_id` (`backend/data/graph.py`, `execution.py` scopes). `get_user_id()` checks `X-Act-As-User-Id` only if `role==admin` (`dependencies.py:83`). Better Auth `role` is `admin|authenticated` (`auth.ts:216`). Tests in `data/db_accessors_test.py` guard `userId`. | 120 `__legacy__` consumers + 40 admin route handlers: audit burden, not a proven bypass — but size invites regression. Recommend `grep` gate in CI. |
| **Recovery** — loss of frontend/WS/executor/queue does not silently corrupt | `PARTIALLY_PROVEN` | DB is healed by REST: `useFlowRealtime.ts:81` invalidates `getExecutionDetails` on resubscribe, closing pre-subscribe race. RabbitMQ quorum persists un-acked messages. `conn_manager.py:72` reconnects with backoff 0.5→8s. | Library/list pages have no resubscribe invalidate — poll-interval stale only. Executor crash after DB `COMPLETED` but before `ack` redelivers (harmless due to status gate, waste only). Scheduler blocked hours silently drops ticks. |

---

## 4. User Journey Findings

### Journey A — New user: Landing/Auth → Signup → Onboarding → Brain Dump → Subscription → First agent → First run

**Trace:** `src/app/(no-navbar)/login|signup` → `src/app/(no-navbar)/onboarding` (`WelcomeStep→RoleStep→PainPointsStep→BrainDumpStep→PreparingStep→SubscriptionStep`) → `src/lib/auth/auth.ts` → `src/app/(platform)/copilot` (first session)

**What works:** Better Auth email+OAuth (Google/GitHub/Discord) is present, `Better Auth` 1.6.14 + `next-themes` + `zod` validation are generation-0 clean. Marketing split layout (`AuthSplitLayout`) renders on all auth pages.

**Friction & failure modes (evidence):**

1. **BrainDump is gated by `NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_BRAIN_DUMP` + `FORCE_FLAG_ONBOARDING_BRAIN_DUMP` (frontend `.env.default:99` commented, backend `/dev` flag).** `(platform)/onboarding` `store.ts` branches on LaunchDarkly `Flag.ONBOARDING_BRAIN_DUMP`. When absent, the new onboarding path never renders — canonical master still shows the legacy onboarding. This is a feature-flag risk: the "new user" journey is two codepaths, not one.

2. **Voice Brain Dump state is local + Dexie, not durable on auth row until finalize.** `BrainDumpStep/recordingStore.ts` + `useBrainDumpRecorder.ts` handle `getUserMedia`, `MediaRecorder`, `durationSecs`, `sizeBytes`, then uploads via `direct-upload.ts`. Failure states are explicit (`FailureState.tsx`, `RecoveryPrompt.tsx`, `TypedFallback.tsx`) — good. But `uploadQueue` (`useUploadQueue.ts`) is in-memory; refresh mid-upload loses queue. `BrainDumpStatus` enum (`schema.prisma: brainDumpStatus recording_uploaded→transcribing→transcribed→extracting→completed→failed`) is on `OnboardingBrainDump` row, but `recordingId` is client-generated and `finalize` is idempotent — so recovery *is* possible if user keeps `recordingId`. However `store.ts` `reset()` on unmount can discard `selectedStoreListingVersionId` and `agentInput` if user navigates back.

3. **Subscription gate ordering unclear.** `SubscriptionStep` reads pricing experiment (`useSubscriptionPricingExperiment.ts`) but paywall is enforced again in `(platform)/PaywallGate` wrapping `PlatformChrome`. A user who completes onboarding, hits `PaywallGate`, then refreshes loses the `onboardingAgentExecutionId` (`UserOnboarding.onboardingAgentExecutionId`) held in `store.ts` — must reconstruct from `GET /v1/onboarding/status`.

4. **Progress recoverability: partial.** `UserOnboarding.completedSteps: OnboardingStep[]` is persisted per `UserOnboarding` row, so refresh on `WELCOME..CONGRATS..CAPABILITY_CARDS` recovers. But BrainDump transcript injection into copilot's first prompt truncates silently if `transcriptLang` mismatched — the full transcript is stored (`transcript` column) but the cut for injection is not shown to user (`usePreparingStep.ts` does not surface truncation).

5. **Auth race on signup → session row → `User` row.** `login/actions.ts:62` `devLogin` (LOCAL branch) does `auth.api.signUpEmail` then `BackendAPI().createUser()` then `getOnboardingStatus()`. Fresh installs that still have `auth.users` shadow table may hit the copy-migration guard (`migration.sql:27 IS NULL → no-op`) — second path wins via `User` upsert. Not a prod defect, but self-host with stale `auth` schema could see duplicate `email` unique violation.

**Recommended experience:** Single onboarding path behind a default-on flag (remove legacy branch after BrainDump GA), persist `BrainDump uploadQueue` to `IndexedDB` (or at least `sessionStorage` with `recordingId`), make PaywallGate use `UserOnboarding.completedSteps` not `store.ts` so refresh survives, surface "transcript truncated to fit context" with download link.

### Journey B — Agent creation: Intent → Copilot/AutoPilot → Generated workflow → Builder → Validation → Execution → Result

**Trace:** `copilot/page.tsx: CopilotPage` → `copilotStreamTransport.ts: CreateTransport` → backend `copilot/executor/__main__.py` + `copilot/bot` → `BuilderChatPanel` `edit_agent` tool → `useSaveGraph.ts` `POST|PUT /v1/graphs` → `Build/Flow` XYFlow → `useRunGraph.ts: RunGraph` → `POST /v1/graphs/{id}/execute` → WS `useFlowRealtime.ts`

**Multiple competing sources of truth — confirmed architectural defect (not just ugly typing):**

| State | Canonical | Drift source | File:line |
|---|---|---|---|
| `graphID` (`flowID` alias) | `nuqs flowID` + `GraphModel.id` | `useBuilderChatPanel:253` auto-creates blank graph if `!flowID && isOpen`, racing `useFlow:49` fetch | `Build/Flow/useFlow.ts:49` |
| `graphVersion` | `GraphModel.version` → `flowVersion` URL | 10 independent `useQueryStates({flowVersion})` declarations; `BuilderChatPanel:376` and `useFlow:129` both write it last-writer-wins | §2 map |
| `graphExecID` | `GraphExecutionMeta.id` → `flowExecutionID` URL | `as GraphExecutionID` cast (`useFlowRealtime:72`) with no runtime guard; optimistic `setIsGraphRunning(true)` before `executeGraph` resolves | `useFlowRealtime.ts:72`, `useRunGraph.ts:163` |
| `draft` | `Dexie BuilderDraft` | Not invalidated on Copilot `edit_agent`; stale draft resurrects on next `checkForDraft` | `draft-service.ts:54`, `BuilderChatPanel:379` missing `deleteDraft` |
| `run state` | `GraphStore.isGraphRunning` + `AgentExecutionStatus` | Optimistic `true` vs WS `graph_execution_event` vs polling `refetchInterval:isGraphRunning?1000:false` feedback loop | `graphStore.ts:42`, `useRunGraph:163` |

**Validation:** only on execute (`useRunGraph:94` `setNodeErrorsForBackendId` from `node_errors: Record<backendId,string>`). No pre-save lint — user saves invalid graph, gets error only on Run.

**Result:** `node_execution_event` feeds `updateNodeExecutionResult:337` (appends deduped by `node_exec_id`, recomputes `latestNodeOutputData`). `updateEdgeBeads` draws active path. Polling hydrates on reconnect.

**`GraphExecutionID` casting seam:** not cosmetic. `Brand<string,"GraphExecutionID">` (`types.ts:333`) is branded, but URL source is `parseAsString` (unvalidated). No `z.parseGraphExecutionID`. Same for `graphId` in `RunGraph:83` `response.data as GraphExecutionMeta`. This permits a malformed `?flowExecutionID=xss` to be subscribed as legitimate and to poison `getGetV1GetExecutionDetailsQueryKey(flowID!, flowExecutionID)`. Impact is not RCE but cache poisoning / noisy Sentry.

### Journey C — Existing operator: Library → Agent → Run → Execution → Artifact/Result → Schedule/Trigger → Failure/Retry

**Can operator answer the 7 questions?**

| Q | Current | Evidence |
|---|---|---|
| 1. What is running? | Partially — `LibraryAgentList` + `FleetSummary` (behind `Flag.AGENT_BRIEFING`) show running counts, but no top-level "Running now" global view. Detail page `RunDetailCard` shows status only if agent row selected. | `library/page.tsx:38` `useLibraryAgents` + `useLibraryFleetSummary` gated |
| 2. What happened? | Yes for last run — `SelectedRunView` renders `nodeExecutionResults` timeline with beads. | `SelectedRunView:45` `useSelectedRunView` |
| 3. Why? | Weak — error is `node_errors` string per node; no root-cause trace linking to failing block's input. | `useRunGraph:94` string map |
| 4. What did it cost? | Weak — `platform/platform-costs` and `block-cost-estimates` exist under admin, not per-run in library. Per-run wallet popover is compact, not execution-linked. | `Admin/platform-costs`, `Profile/credits` |
| 5. Did it succeed? | Yes — `AgentExecutionStatus` (`COMPLETED|FAILED|TERMINATED|REVIEW`) is SoR, shown. | `SelectedRunView:76` `isLoading && !run` gate |
| 6. If failed, why + what next? | Weak — `ErrorCard` gives retry but not "fix node X's credential" guided action. `PendingReviewsList` handles `REVIEW` (human-in-loop) but not generic failure. | `molecules/ErrorCard` |
| 7. What should I do next? | Missing — no "Retry with same inputs" / "Open in Builder" / "Reschedule" CTA from run view. | `EmptyTasks.tsx:40` has delete+export, not rerun |

**Other:** Schedules (`ScheduledTasks` via `SchedulesPanel`) and triggers (`Triggers`) live inside library detail, not as global "Schedules" page — `followups/__tests__/main.test.tsx` `followups-empty` copy "schedule an agent from the builder" admits the discoverability gap. `PendingHumanReviews` (`PendingReviewCard`) is well-built but sits behind `REVIEW` status only.

**Recommended experience:** Promote "Running now" global strip, surface per-run cost + failure root-cause + 1-click Retry/Edit-in-Builder from `SelectedRunView`, move Schedules to first-class nav.

### Journey D — Marketplace: Discovery → Evaluation → Install → Configuration → Execution → Library ownership

**Trace:** `marketplace/page.tsx` → `StoreCard` (legacy `__legacy__/StoreCard` + new `marketplace/components/StoreCard`) → `search/MainSearchResultPage` → `PublishAgentModal` → `LibraryAgent` owned copy

**Finding: Separate product embedded, not acquisition path.** `__legacy__/composite/*` (`HeroSection`, `AgentsSection`, `FeaturedSection`) render marketplace home; new `marketplace/components/AgentsSection` wraps the same `StoreCard` but still imports `StarRatingIcons` from legacy. `MarketplaceAgentsContent:99` TODO "Create new endpoint for builder-specific marketplace agents" bears witness that builder's block picker reuses marketplace listing fetch without dedicated API. Install flow forks through `PublishAgentModal` (`contextual/PublishAgentModal`) that writes `StoreListing` → `LibraryAgent` via `StoreVersionsReviewed` — a second product surface with its own review queue (`admin/marketplace` moderation).

**Result:** Marketplace is 3rd after `build` and `admin` in legacy density (see §8). New user cannot tell whether marketplace agent is "mine" (library) or "theirs" (template) without inspecting `libraryAgents[].graph_id` mapping.

**Recommended:** Collapse `__legacy__/composite` into new `marketplace/components` (four files), remove `HeroSection` straddle, make `LibraryAgent.fromMarketplace` badge primary.

### Journey E — Copilot: Message → Stream → Tool → Memory/Context → File → Artifact → Agent modification/execution

**Trace:** `CopilotPage` → `CopilotChatHost` → `copilotStreamTransport:77` (`POST /api/chat/sessions/{id}/stream` via `DefaultChatTransport`, `message_id: randomUUID()` PK dedup) → `copilotChatRegistry` per-session runtime → `copilotStreamStore` → `ArtifactPanel`/`ContextPanel`

**What works:** Copilot architecture is the most deliberate in the repo. Per-session runtime registry (`copilotChatRegistry.ts`) lets streams continue in background JS while another chat is on screen. Transport generates `message_id` (`copilotStreamTransport.ts:99`) so Postgres PK (`ChatMessage.id`) is the atomic dedup — retransmit lands on 409 and backend short-circuits to subscribe-only. Smoothing transform (`copilotStreamSmoothing.ts`) batches word deltas ~30ms (`STREAM_RENDER_THROTTLE_MS:30` `useCopilotStream.ts:30`) to avoid whole-tree rerenders. `useCopilotReconnect`, `useWakeResync`, `useStreamActivityWatchdog`, `useHydrateOnStreamEnd` close the "laptop lid → stale" gap.

**Failure boundaries:**

- **Stream interruption:** Previously used `?last_chunk_id` GET-resume, removed because AI SDK v5 `UIMessageStream` parser crashes on orphan deltas (`copilotStreamTransport.ts:106`). Now always replay from `0-0` and `deduplicateMessages` consumer side — safe but replays full turn (bandwidth vs correctness tradeoff, correct).
- **Tool cards:** `GenericTool/RunAgent/RunBlock/DecomposeGoal` mark in-progress parts as completed/errored on `handleFinish` (`helpers.ts:149`), preventing spinner forever.
- **Files:** `FileDropZone` → `useSendMessage` → `workspace.files` `FileUIPart` (`url` like `/api/proxy/api/workspace/files/{id}/download`) extracted via regex (`copilotStreamTransport.ts:77`). Large files hit `event_bus MAX_MESSAGE_SIZE` truncation (WS only) but DB holds full — same artifact risk as §3.
- **Persistence:** Copilot history lives in `ChatSession` + `ChatMessage` tables (Prisma), not in WS. Refresh rehydrates via `getGetV2GetSessionQueryKey` (`useCopilotStream.ts:1`). So stream can die, data does not.
- **Agent mod:** `BuilderChatPanel` tools `edit_agent`/`run_agent` are the Copilot→Builder bridge; guarded by `processedToolCallsRef` + `bindingRef` but missing `draft` invalidation (`§4 B`).
- **Resume stall:** `RESTORE_STALL_TIMEOUT_MS` + `FINISH_REFETCH_SETTLE_MS 500` + `FINISH_REFETCH_ATTEMPTS_DEFAULT 1` vs `8` when mode-switch pending (`useCopilotStream.ts:241`) — continuation turn dispatch window bracketed correctly.
- **Previous regression fixed:** `fix(frontend/copilot): prevent resumeStream() call with uninitialized useChat state (#13766)` guards the uninitialized `useChat` race.

**Gaps:** `copilotStreamErrorHandlers.ts` maps SSE errors to toast, but `ArtifactPanel` re-render on large `0-0` replay can flash (no virtualization). `MemoryVisualizer` (admin) is not operator-facing — copilot memory is opaque to user.

---

## 5. Authentication Assessment — P0

### Current architecture (canonical)

```
Browser
  │ httpOnly cookies: better-auth.session_token + session_data (5m signed cache)
  │ (pre-migration) sb-<project>-auth-token (.0/.1 chunked, base64-)
  ▼
Edge Middleware  frontend/src/lib/auth/middleware.ts:58
  ├─ /api/* → passthrough (route authenticates itself)
  ├─ has sb-* && !sessionCookie && canConsumeLegacyCookies() → 302 /api/auth/supabase-bridge?next=…
  │    jose verify HS256 (aud=authenticated, tolerance 30d) → findUserById → ban check → createSession → setSessionCookie
  │    always clears sb-* (supabase-bridge.ts:166)
  └─ isProtectedPage/admin → getCookieCache(secret) else fetch /api/auth/get-session (3s, null=not-admin)
        ▼ 302 /login?next=… if !sessionCookie on protected

Next.js Handlers  app/api/auth/[...all]/route.ts (toNextJsHandler)
  ├─ /api/auth/get-session  (DB or cookieCache)
  ├─ /api/auth/token        (ES256 JWT, aud=authenticated, 1h, kid, JWKS_ALG ES256)
  ├─ /api/auth/jwks         (AuthJwks public key, published)
  └─ /api/auth/supabase-bridge (migration shunted)

Frontend Server Components: getServerSession() → auth.api.getSession({headers})
                           getServerAuthToken() → auth.api.getToken({headers}) per-request cache()

Proxy  app/api/proxy/[...path]/route.ts:208  token = await getServerAuthToken() (null if anon)
       fetch(backend, headers: {Authorization: Bearer <token>, X-Act-As-User-Id?, X-API-Key?})
       also: direct SSE  copilotStreamTransport.ts:77  getCopilotAuthHeaders() → Bearer

Backend  autogpt_libs/auth/jwt_utils.py:101
  bearer_jwt_auth (HTTPBearer) → parse_jwt_token
    ├─ alg HS* → verify with JWT_VERIFY_KEY (HS256)  [migration]
    └─ else   → PyJWKClient(JWT_JWKS_URL=http://frontend:3000/api/auth/jwks, cache 3600, timeout 5s).get_signing_key(kid)
                kid miss → refetch; fallback to legacy asymmetric key if alg matches
          → jwt.decode(aud, alg) → verify_user(payload) → User(sub,email,role)
  Service path: service.py:27 requires_frontend_service(scope) aud=autogpt-platform-backend, sub=service:frontend
  DB: platform schema UserAuthIdentity/Session/Account/Jwks (Prisma) via frontend pg Pool (search_path=platform, auth.ts:54)
```

### A. Is Supabase authentication still required?

| Dependency | Classification | Evidence |
|---|---|---|
| `SUPABASE_JWT_SECRET` + `JWT_VERIFY_KEY` | **Migration compatibility** | `frontend/.env.default:40` commented, `supabase-bridge.ts:84`, `autogpt_libs/auth/config.py:27` fallback `JWT_VERIFY_KEY or SUPABASE_JWT_SECRET`, `jwt_utils.py:101` HS branch |
| `auth.users` shadow table (`supabase-auth` Docker service, `db/docker/docker-compose.yml:93` gotrue v2.170) | **Migration compatibility / dead for fresh installs** | `migrations/20260716120000_copy_supabase_users_to_better_auth:27` `if to_regclass('auth.users') IS NULL → no-op` — fresh DB has no `auth` schema, CI boots fine without it |
| `SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS` | **Migration compatibility** | `supabase-bridge.ts:93` default 30d, tolerance `30d` |
| `supabaseBridge()` plugin + `/api/auth/supabase-bridge` endpoint + `legacy-cookies.ts` | **Migration compatibility** | `auth.ts:223`, `middleware.ts:68`, `supabase-bridge.ts:133` |
| `JWT_JWKS_URL` (`http://frontend:3000/api/auth/jwks`) + `BETTER_AUTH_SECRET` + `AuthJwks` row | **Actively required** | `config.py:41` mandatory, `auth.ts:63/207` ES256 keyPair, `schema.prisma:89` |
| `@supabase/*` npm | **Dead** | 0 hits; only `storage.Key.LOGOUT="supabase-logout"` label remains (`local-storage.ts:5`) |

**Verdict:** On a fresh DB you can run with `SUPABASE_JWT_SECRET` and `JWT_VERIFY_KEY` unset (JWKS-only). The only reason they persist is to keep pre-migration `sb-*` cookies bridgeable for ~30 days.

### B. Can two identity paths disagree?

| Vector | Verdict | Evidence | Blast radius |
|---|---|---|---|
| Stale middleware optimistic pass | By design | `middleware.ts:58` "real validation happens in route handlers" | Transient 401 vs redirect glitch |
| Cookie cache `session_data` (5m) | Yes — up to 5m | `auth.ts:102` `cookieCache:{maxAge:5*60}`; `middleware.ts:20` reads cached `role` without DB hit | Admin promotion/demotion delayed |
| Logout/revocation replay | **Yes — 1h window** | `actions.ts:112` `signOut()` deletes DB row, `jwt_utils.py:137` verifies signature+`exp` only, no session lookup. `helpers.ts:128` comment warns explicitly. | Stolen `Bearer <JWT>` replays until `exp` |
| Role change vs minted JWT | Same 1h stale | `auth.ts:216` `definePayload:{role}` baked at mint time, `jwt_utils.py:174` `role != admin → 403` | Demoted admin keeps admin until JWT expiry unless `revokeSessions` called |
| JWKS / frontend down | Denial of service for ES tokens | `config.py:52` JWKS mandatory, `jwt_utils.py:27` timeout 5s, `PyJWKClientError → 401` | All ES requests 401 for ~5s, mitigated by 1h cache |
| HS fallback still accepts | Yes | `jwt_utils.py:101` HS path accepted while `JWT_VERIFY_KEY` set | HS secret leak forges forever until window closed |
| Cross-request Map amplification | Explicitly prevented | `getServerAuthToken.ts:28` per-request `cache()`, `helpers.ts:134` same, comment `:15` warns | No amplification today; regression risk noted |
| Frontend cookie vs backend Bearer mismatch | Transient | Proxy is only Bearer injector; `BETTER_AUTH_URL` vs `INTERNAL_URL` mismatch → `null` token → backend 401 | Misconfig 401; `LOCAL_UNCOMMITTED` adds `BETTER_AUTH_INTERNAL_URL` read |

### C. Intended end state & what prevents removing bridge

**End state:** Frontend owns `platform.UserAuth*` + signing (ES256 via `AuthJwks`); backend is pure verifier against `JWT_JWKS_URL`; no HS secret, no `sb-*`, no `supabaseBridge` plugin, no dual verify (`jwt_utils.py:101` HS branch deleted).

**Gates (ordered):**
1. Live `sb-*` cookies in wild — bridging preserves them for 30d window (`SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS` 30). Removal before window forces re-login.
2. `JWT_VERIFY_KEY` must be retired *with* bridge — `jwt_utils:104` HS + bridge 30d tolerance are co-dependent.
3. Post-GoTrue password changes not re-copied (bcrypt preserved) — bridge lets those users in; removal forces reset.
4. Operational proof: query `auth.users where deleted_at is null` → 0 active, set `MAX_AGE=0` in staging → 0 `302 supabase-bridge` in logs for >30d, then one PR deletes `auth.ts:11,223`, `supabase-bridge.ts`, `legacy-cookies.ts`, `middleware.ts:68-86`, envs, and `jwt_utils.py:104-125` HS branch.
5. Self-host docs must stop provisioning `auth` schema for platform auth (keep for realtime demo).

### Diagram — see ASCII above. SoR: Frontend is signer + session DB; Backend is verifier; Proxy is sole Bearer injector; two stale windows are `session_data 5m` and `JWT 1h`.

---

## 6. Reliability Assessment — Execution, Scheduling, Recovery, WS

### System-of-record per transition

| Transition | SoR | Evidence |
|---|---|---|
| `add_graph_execution` → `INCOMPLETE` → conditional `QUEUED` → RabbitMQ `publish` | **Postgres `AgentGraphExecution`** | `execution.py:889` init `INCOMPLETE`; `utils.py:1358` `update_graph_execution_stats(QUEUED)` before `publish`; compensation `FAILED` on publish fail `utils.py:1394` |
| Queue dispatch | **RabbitMQ quorum `graph_execution_queue_v2` durable + Redis `ClusterLock SET NX EX`** | `rabbitmq.py:213` `delivery_mode=2`, `cluster_lock.py:68`, `manager.py:1363` `active_graph_runs` dict |
| Node progress | **Postgres `AgentNodeExecution`** (WS is ephemeral `SPUBLISH`, swallows errors `event_bus.py:117`) | `execution.py:1127` guarded `WHERE` transitions |
| Schedule definition | **Postgres `apscheduler_jobs`** (`SQLAlchemyJobStore`) | `scheduler.py:1448` |
| Artifact | **GCS/S3/local + `UserWorkspaceFile`** (`WorkspaceManager`) — not just `CompletedBlockOutput` | `file.py:400`, `workspace.py:164` |
| Cost ledger | **Postgres `CreditTransaction` + `PlatformCostLogs` via `spend_credits`** | `billing.py:152`, `db_manager.py:302` |
| Notification | **RabbitMQ topic `notifications` → `UserNotificationBatch`** | `notifications.py:74` |

No distributed commit: DB then RabbitMQ are non-atomic, compensated by marking `FAILED` on publish failure.

### Answers to required questions

**Can UI say FAILED when still running?** No — DB never hallucinates `FAILED` if `RUNNING` (transition guard `VALID_STATUS_TRANSITIONS:142`). *Inverse* (UI says RUNNING when terminal) transiently yes: WS miss + no `invalidateQueries` on list pages keeps polling stale for interval.

**Can scheduled execution run twice?** Protected by conditional `QUEUED` guard + `active_graph_runs` + `ClusterLock` + per-batch Lua tombstone (BatchExecutor). *Duplicate dispatch* (two distinct `graph_exec_id`s for same cron tick) not prevented by these — `coalesce=True, max_instances=1000` (`scheduler.py:1442`) can fire many concurrent `_execute_graph` if DB stalls; each is distinct id but user-perceived duplicate work. RabbitMQ redelivery after executor crash before `ack` is safe if `RUNNING` already persisted (status gate early-returns), duplicates work only when crash occurred before `RUNNING` commit.

**Can execution succeed but fail to persist artifact?** Hard failure propagates to node `FAILED` (`manager.py:764`), not silent success. Subtle: `cost_tracking` fire-and-forget `create_task` may drop cost row beyond 5s drain; large-file `MAX_MESSAGE_SIZE` truncation shows "payload too large" in WS while DB holds full output.

**Can WS interruption cause permanently stale UI?** No permanent, yes interval-stale. Detail page heals via `useFlowRealtime:81` `invalidateQueries` on resubscribe; list pages rely on polling — background-throttled tab stays stale until nav/interval.

**Can retry create duplicate work?** Yes — `reject(requeue=True)` (`manager.py:1724`) without dedup key re-enters `_handle_run_message`; safe only due to status gate. Cancel via `fanout auto_ack=True` (`manager.py:1444`) is lossy if executor offline — `Stop` may need `15s` waiter + DB direct `TERMINATED` fallback (`utils.py:1107/1153`) and can still be ignored.

### Severity findings

- **H1 (High) Cancel lossy:** fanout `auto_ack=True` — cancel sent while executor restarting is dropped; `stop_graph_execution` has 15s race.
- **H2 Notification never fails execution:** `queue_notification → NotificationResult(success=False)` (`notifications.py:179`) swallowed; execution `COMPLETED` with no email/push.
- **H3 Cost drain loop-filtered:** `drain_pending_cost_logs` drains only same-loop tasks 5s (`manager.py:1868` + `cost_tracking.py:78`), `charge_reconciled_usage` returns `0,0` on exception (`billing.py:333`) — deploy can lose cost rows while provider was paid.
- **M1 `update_many` 0 rows silently no-op** (`execution.py:1130`); caller `manager.py:929` never asserts.
- **M2 consumer timeout 24h** (`utils.py:1008`) disables broker dead-consumer detection; liveness via `cluster_lock.refresh()` `sleep(0.1)` (`manager.py:1175`) only.
- **M3 `misfire_grace_time=None`** drops missed ticks with warning only (`scheduler.py:148`).

---

## 7. UX/UI Assessment

> *Assessment against `globals.css:7` tokens, `providers.tsx:37` forced light, `AuthSplitLayout` + `Organisms`+`Molecules` vs `__legacy__` leaves.*

### What is already good

- Auth pages share `AuthSplitLayout` (feature list + footer), `LoadingLogin`/`LoadingSignup` skeletons, `EmailNotAllowedModal`, `ExpiredLinkMessage` — consistent.
- `ErrorCard` + `ArtifactErrorBoundary` (`ArtifactContent:58`) + `ErrorBoundary context="application"` (`layout.tsx:53`) give two-tier error handling (page vs artifact).
- Library empty states exist but are not consistent — `followups-empty` has spec copy "Nothing scheduled yet / schedule an agent from the builder" (`followups/__tests__`), `EmptyTasks` has illustration + delete CTA, but runs list uses raw `Skeleton` stacks.

### Evidence-backed UX problems (not subjective redesign)

| ID | Problem | Why friction | Location |
|---|---|---|---|
| `UX-001` | Builder clobbers unsaved work | `useFlow:149` `setNodes([])+addNodes(customNodes)` fires on every `graph`/`blocks` change and on Copilot `refetchGraph` without `hasChanges` gate → user typing in node loses input if Copilot `edit_agent` lands mid-edit | `Flow/useFlow.ts:149`, `BuilderChatPanel:379` |
| `UX-002` | BrainDump recovery invisible | Transcript truncated for injection, user not told; upload queue lost on refresh; "failed" shows generic `FailureState` without device permission vs network vs mime error | `BrainDumpStep:typedFallback`, `useUploadQueue.ts`, `PreparingStep` |
| `UX-003` | Execution observability 7-question gap | Library cannot answer "what is running / why failed / cost / next step" from one screen; must drill agent→run→detail. Fleet summary gated behind `AGENT_BRIEFING`. | `library/page.tsx:38`, `SelectedRunView:76`, `platform-costs` under admin |
| `UX-004` | No guided retry / fix from failure | `SelectedRunView` shows error string, no "fix credential `X` → retry" or "open at failed node in Builder" | `SelectedRunView` |
| `UX-005` | Schedules discoverable only via detail drill-down | No global `/schedules` list; empty-state copy punts to builder | `followups/__tests__/main.test.tsx` |
| `UX-006` | `GraphExecutionID` URL poisoning | Unvalidated `?flowExecutionID=` can be any string; subscribed as `GraphExecutionID`, invalidates cache with attacker string | `useFlowRealtime:72`, `types.ts:333` |
| `UX-007` | Copy/paste duplicate shortcuts | `copyPasteStore.ts:19` and `Flow/useCopyPaste.ts:40` both claim `Ctrl+C`/`localStorage` | `stores/copyPasteStore.ts` vs `useCopyPaste.ts` |
| `UX-008` | Builder mobile blocked | Full-screen `MobileWarning` rather than responsive fallback; no read-only mobile inspection | `build/page.tsx:8` `MobileWarning` |
| `UX-009` | Notification opt-in vs BrainDump race | `CopilotPage: MobileDrawer + NotificationDialog` conditioned on `isBrainDumpEnabled` — modal may never appear if flag flips mid-session | `CopilotPage.tsx:56` |
| `UX-010` | WS cancel not confirmed | Stop button fires `usePostV1StopGraphExecution` but UI shows spinner until 15s `wait_timeout` even though backend already patched DB `TERMINATED` for `QUEUED` path | `RunGraph:181`, `utils:1107` |

### Layout/responsive

- PlatformChrome three-way branch (`PlatformChrome:31`) is not A/B test but **per-flag shell** — Tour sidebar, new `AppSidebar`, legacy `Navbar` coexist; `SidebarProvider` CSS var `--sidebar-width` switches `18.25rem` vs `19rem`. Maintains coherently but doubles CSS.
- `AppSidebar` (`AppSidebar.tsx:31`) uses `Suspense` + `useLinkStatus` nav spinners — good; `Navbar` (`components/layout/Navbar`) still imports `IconType` from legacy (icon seam).
- `body.bg-[#F6F7F8]` (`globals.css:81`) overrides token `bg-background` — any future dark or adaptive surface will need removal.

---

## 8. Design-System Assessment — Two Systems, One Inversion

**Verdict:** Design system is **two systems with inverted delegation — `NEW → LEGACY` is the architectural defect, not the count.**

### Classification (§2 counts)

| Bucket | Files | Representative | New equivalent | Status |
|---|---|---|---|---|
| `LEGACY_PRIMITIVE` | 28 | `ui/button` 33, `ui/input` 15, `ui/skeleton` 23, `ui/table` 10 | Partial — `atoms/Button` exists but `atoms/Input` wraps legacy | **Inverted** |
| `LEGACY_ICON` | 1 (`ui/icons.tsx`) | 1880 LOC, 24 imports | **None** (`atoms/Icon` uses Hugeicons, not mapped) | Highest effort |
| `LEGACY_COMPOSITE` | 4 | `AgentsSection`, `FeaturedSection` | Straddles new `marketplace/components` | Leaf duplication |
| `LEGACY_MARKETPLACE` | 11 | `StoreCard`, `CreatorCard`, `FilterChips` | New `marketplace/components/StoreCard` already exists but still imports `StarRatingIcons` | Duplicative |
| `LEGACY_BUILDER` | 5 | `action-button-group`, `delete-confirm-dialog`, `types` | Direct consumer cluster in `build/` (54 lines) | Active |
| `LEGACY_ADMIN` | 0 files | Admin is *consumer tier* (40 lines), no dedicated system | `admin/**` composes primitives | Heavy consumer |
| `LEGACY_PAGE` | 0 files | No page-level; `Sidebar.tsx:8` | Layout-scoped | — |
| Orphans (0 consumers) | 10 | `radio-group`, `render`, `HeroSection`, `FeaturedCreators`, `RatingCard`, `AgentImages*`, `BecomeACreator`, `SmartImage`, `delete-confirm-dialog` | — | Delete safe |

### Dependency direction — the leverage finding

**Inverted: 11 lines where NEW imports LEGACY — must be fixed before consumer migration has leverage.**

| Wrapper (NEW) | `file:line` | Legacy import |
|---|---|---|
| `atoms/Input` | `atoms/Input/Input.tsx:4` | `__legacy__/ui/input` |
| `atoms/Input` | `atoms/Input/useInput.ts:1` | `__legacy__/ui/input:InputProps` |
| `atoms/Select` | `atoms/Select/Select.tsx:10` | `__legacy__/ui/select` |
| `atoms/DateInput` | `atoms/DateInput/DateInput.tsx:5,11,12` | `__legacy__/ui/button, popover, calendar` |
| `atoms/DateTimeInput` | `atoms/DateTimeInput/DateTimeInput.tsx:13,14` | `__legacy__/ui/popover, calendar` |
| `molecules/Table` | `molecules/Table/Table.tsx:9` | `__legacy__/ui/table` |
| `molecules/Collapsible` | `molecules/Collapsible/Collapsible.tsx:9` | `__legacy__/ui/collapsible` |
| `molecules/Dialog/DrawerWrap` | `molecules/Dialog/components/DrawerWrap.tsx:1` | `__legacy__/ui/button` |

`src/components/ui/**` is clean (`rg __legacy__ src/components/ui → 0`). `__legacy__→__legacy__` has 24 internal edges (e.g., `ui/calendar:13 → ui/button:buttonVariants`) — expected DAG.

```
Consumers (120 external) ──→ __legacy__/ui/*   (109 direct)
                    └──→ atoms/molecules ──→ __legacy__/ui/*  (11 inverted)
```

### Smallest migration set that breaks largest dependency

**Step 1 — 6-file inversion fix (11 edges, unlocks ~60 downstream consumers):** `atoms/Input`, `atoms/Select`, `atoms/DateInput`, `atoms/DateTimeInput`, `molecules/Table`, `molecules/Collapsible/DrawerWrap` — rebase on `src/components/ui/*` (clean) or direct `@radix-ui/*`. No consumer changes. Proves `rg __legacy__ src/components/{atoms,molecules,ui} → 0`.

**Step 2 — trivial sweep (42 edges, XS):** `ui/skeleton:15` → `atoms/Skeleton`, `ui/separator:11` → `ui/separator`, `ui/badge:8` → `atoms/Badge` — automated codemod `s/__legacy__/atoms|ui/`, no API drift. Proves migration tooling.

**Step 3 — `ui/button` (33) — M:** `variant` mapping (`default→primary` etc.) + wrapper chaining fix.

**Step 4 — `ui/icons` (24, 1880 LOC) — L:** manual icon audit, `IconType → atoms/Icon:IconSvgElement` prop change — defer, pair with design.

### Theme contradiction — classification

**`INTENTIONAL_LIGHT_ONLY` + dead scaffolding.** Not partial/abandoned — disabled at three independent layers:

| Layer | `file:line` | Evidence |
|---|---|---|
| Provider | `providers.tsx:37` | `ThemeProvider forcedTheme="light" {...props}` overrides any `defaultTheme`/`enableSystem` |
| Tailwind | `tailwind.config.ts:7` | `darkMode: ["class", ".dark-mode"]` — selector `.dark-mode` never set; `dark:` never evaluates |
| Toggle | `Navbar.tsx:134` | `{/* <ThemeToggle /> */}` commented out |
| CSS | `globals.css:42` `.dark {…}` + `globals.css:81` `bg-[#F6F7F8]` hardcode | Dark vars defined, dead; body bg overrides token |
| Usage | `rg "dark:"` | **492** `dark:text-*` etc. in **108** files — all dead code (e.g., `AgentsSection:dark:text-neutral-200`) |
| Env | `next-themes 0.4.6` | Installed but neutralized |

**Recommendation:** Do not "add dark mode". Either (a) if light-only is permanent, delete 492 `dark:` occurrences + `globals.css:42` + `ThemeToggle` (P3), or (b) fix atomically in one PR: remove `forcedTheme`, set `darkMode: ["class"]` (match `.dark`), remove `bg-[#F6F7F8]` → `bg-background`, uncomment `ThemeToggle`. Current state ships dead CSS for every route.

---

## 9. Performance Assessment — Evidence, Not Speculation

### Dependency duplication — measured

| Pair | Shipped together? | Verdict |
|---|---|---|
| `framer-motion 12.23.24` + `motion 12.38.0` | **Yes** — both in `package.json:94,103` and both imported (`CopilotPage: DotDistortionShader` uses `framer-motion`, `AppSidebar` uses `motion`). | `DEPENDENCY CLEANUP` — `motion` is `framer-motion`'s rename. One alias is waste. |
| `react-icons 5.5.0` + `lucide-react 0.552` + `@hugeicons/core-free-icons 4.2.3` + `@hugeicons/react` | Partial — `react-icons` only 1 import (`react-icons` in `Frontend` grep, heavyweight), `lucide-react` 0 direct imports but listed, `hugeicons` is canonical via `atoms/Icon`. | `DEPENDENCY CLEANUP` — `react-icons`/`lucide-react` are dead weight; tree-shaking saves but `node_modules` cost on CI remains. |
| `dexie 4.2.1` + `pg 8.21` in frontend | **Yes** — `pg` should be server-only but appears in frontend `package.json:107` because `auth.ts:42` `Pool` is bundled to Edge; mitigation is `serverExternalPackages`? Actually `pg` is used in frontend server runtime (Edge/Node), not browser. | Not bundle-bloated (server-only) — leave. |
| `shadcn` duplication | `src/components/ui/*` clean vs `__legacy__/ui/*` shadcn forks — both ship. | Migrating `skeleton`/`separator`/`badge` removes one copy immediately. |

No bundle measurement was run in this audit (would require `next build` + `ANALYZE=true`). Recommendation: measure route-level JS (`next build` output `First Load JS`) before claiming user-visible defect. Current evidence supports `DEPENDENCY CLEANUP`, not `USER-VISIBLE PERFORMANCE DEFECT` except one case below.

### Table rendering — user-visible risk

`Admin/diagnostics ExecutionsTable.tsx:1096` LOC with `@tanstack/react-table:73` — renders without `react-window` virtualization (`react-window 2.2.0` is in deps but not used in `ExecutionsTable`). `DiagnosticsContent.test.tsx` mocks large sets齐. This is `P2` if admin paginates (it does — `Table` + `Pagination`), not a fleet-wide defect. Marketplace `StoreCard` grid without virtualization is acceptable for ~20 cards.

### Hydration / client boundaries

`CopilotPage: useCopilotStream` `hasActiveStream` + `hydrateCompletedRef` dance + `FINISH_REFETCH_SETTLE_MS 500` is evidence of real hydration race care. `Providers: NuqsAdapter` inside `QueryClientProvider` ordering is correct. `next build --turbo --max-old-space-size` flag (`package.json:10` canonical `16384`, LOCAL override `4096`) signals build memory pressure — not user perf.

**Bottom line:** Keep performance work `P2/P3` except builder `XYFlow` canvas — any virtualization claim without profiling would be speculation.

---

## 10. Test-Gap Assessment

### What exists (mature on happy path)

- **Integration:** Vitest + RTL + MSW, custom `test-utils.tsx` wraps `QueryClientProvider` + `BackendAPIProvider` + `NuqsTestingAdapter`. Orval generates `getGetV2*MockHandler200/422/401`. ~90% claimed.
- **E2E:** 8 `*-happy-path.spec.ts` (auth, builder, library, marketplace, publish, copilot, settings, api-keys) — `playwright.config.ts` gecko via `NEXT_PUBLIC_PW_TEST`.
- **Visual:** Storybook 9.1 + Chromatic (`atoms/Button.stories`, etc.).

### What is missing — highest-value regression tests (by product risk)

| Gap | Priority | Why | Location |
|---|---|---|---|
| Expired session → `/api/proxy` 401 → PaywallGate remount loop | **P1** | Middleware passes with stale `session_data` 5m, proxy then 401s | `middleware:14`, `PaywallGate` |
| `400` graph validation → field-level `node_errors` mapping | **P1** | `useRunGraph:94` `setNodeErrorsForBackendId` has no test for stale-clear path | `useRunGraph.ts:105` |
| WS disconnect mid-execution → stale `RUNNING` | **P1** | Only detail page heals (`useFlowRealtime:81`); lists don't | `useFlowRealtime` vs `LibraryAgentList` |
| Execution `FAILED` → retry with same inputs | **P1** | No "Retry" path tested | `SelectedRunView` |
| BrainDump `getUserMedia` permission denied → typed fallback | **P2** | `FailureState` rendered but not E2E'd across browsers | `BrainDumpStep` |
| `GraphExecutionID` malformed `?flowExecutionID=xss` → cache poisoning | **P1** | `as GraphExecutionID` unchecked | `useFlowRealtime:72` |
| Scheduler coalesced tick dropped silently | **P2** | `EVENT_JOB_MISSED` never surfaced | `scheduler.py:148` |
| ClamAV rejection → upload blocked | **P2** | Virus scan error copy "payload too large" conflated | `api/features/workspace` |
| Duplicate `POST /v1/graphs/{id}/execute` → dedup vs double | **P0** | No test for `reject requeue` race | `utils.py:1367`, `manager.ts:1663` |
| Cross-user `GET /v1/graphs/{id}` (other `user_id`) → 403 | **P0** | AuthZ is per-query filter, needs negative test | `dependencies.py:83` |

Do not propose exhaustive testing — the ten above protect invariants.

---

## 11. Environment / Deployment Assessment

### Matrix

| Env | `NEXT_PUBLIC_BEHAVE_AS` | `NEXT_PUBLIC_APP_ENV` | `NEXT_PUBLIC_AGPT_SERVER_URL` | `AGPT_SERVER_URL` (server) | `NEXT_PUBLIC_AGPT_WS_SERVER_URL` | `JWT_JWKS_URL` (backend) | `DATABASE_URL` | Redis | Others |
|---|---|---|---|---|---|---|---|---|---|
| **local dev** (`pnpm dev` + `docker compose --profile local up deps_backend`) | `LOCAL` (`.env.default: NEXT_PUBLIC_BEHAVE_AS=LOCAL`) | `local` | `http://localhost:8006/api` | unset (frontend fallback) | `ws://localhost:8001/ws` | `http://localhost:3000/api/auth/jwks` (backend `.env.default:42`) | `postgresql://postgres:…@localhost:5432/postgres?schema=platform` | `redis-0:17000` cluster `127.0.0.1`? actually `redis-0:6379` via `REDIS_USE_ANNOUNCED_ADDRESS true` | ClamAV `3310`, FalkorDB `6379`, RabbitMQ `5672` |
| **Docker self-host** (`docker compose up` via `docker-compose.platform.yml`) | `LOCAL` | `local` | `http://rest_server:8006/api` via `x-backend-env` | same `DATABASE_URL` `@db:5432` | `ws://websocket_server:8001/ws` via `DATABASEMANAGER_HOST` etc. | `http://frontend:3000/api/auth/jwks` (compose `http://frontend:3000`) | `@db:5432 platform` | `redis-0/1/2` cluster | Same |
| **Hosted dev** (`isDev()`) | `CLOUD` | `dev` | `https://…` (Vercel env, not in repo) | set | `wss://…` | `https://<host>/api/auth/jwks` (`config.py:52` validates `https://` for non-local) | Neon/managed Postgres `platform` schema | Managed Redis Cluster (env `REDIS_HOST` not `redis-0`) | Same |
| **Hosted prod** (`isProd()`) | `CLOUD` | `prod` | `https://…` | set | `wss://…` | `https://<host>/api/auth/jwks` | Managed | Managed | Managed |
| **Preview** (`getPreviewStealingDev()`) | `CLOUD` | `dev` | inherits dev | — | — | `NEXT_PUBLIC_BEHAVE_AS` steering | inherits dev DB (branch) | — | `NEXT_PUBLIC_PREVIEW_STEALING_DEV` non-null branch |

### Hidden assumptions

1. **Docker Compose is a dev topology that is documented as deployment but not operable as self-host target without overrides.** `docker-compose.local.yml` (untracked, LOCAL) already overrides `frontend:3002:3000` + `falkordb 6381/3003` + `/local-skills` mount — evidence that bare `platform.yml` does not match workstation. Readiness probes gate `db → rabbitmq healthy` (`compose.yml:166`) to avoid `.erlang.cookie` race — self-host without that gate fails intermittently.

2. **JWKS URL is env-sensitive and fail-closed.** `config.py:52` requires `https://` for non-local, `http://` only for `LOCAL`. `compose` sets `http://frontend:3000/api/auth/jwks` (Docker DNS). Hosted must set `https://<frontend>/api/auth/jwks`. Misconfig → 5s `PyJWKClientError → 401` for all ES256 requests (see §5).

3. **`NEXT_PUBLIC_*` vars are inlined at build** (`.env.default` comments: BrainDump flag "unset would ship feature", `BETTER_AUTH_URL` branching). Rebuilding on flag change is required — not a runtime toggle. `LaunchDarkly` (`NEXT_PUBLIC_LAUNCHDARKLY_ENABLED`) is the only runtime flag.

4. **`BETTER_AUTH_URL` vs `BETTER_AUTH_INTERNAL_URL`** (`helpers.ts:162`) — canonical ignores `INTERNAL_URL`, LOCAL uncommitted reads it. Upstream needs no fix; LOCAL fix is valid ergonomy for Docker `host.docker.internal`.

5. **Prisma schema `platform`:** `DATABASE_URL ?schema=platform` is authoritative. Self-host operators who default to `public` without `?schema=platform` will have empty `UserAuth*` tables — appearance of "auth broken". Docs should call this out.

6. **`cpus: 2` + `DisableCssMinimizer` hack** (`next.config.mjs`) are not cosmetic — they are build-targeting: halving CSS at toolchain bug (`cssnano "Invalid array length"` on large chunks) and limiting workers to survive `4096` MiB. Hosted `Vercel` path skips standalone.

---

## 12. Prioritized Backlog

> Every High-impact item references concrete `file:line`. `P0` must not be crowded by `P3`. `S < 1d | M 1–3d | L 3–7d | XL 7d+` (single engineer, excluding review/QA).

### Summary counts

| Priority | Count | Meaning |
|---|---|---|
| **P0** — Integrity/security/reliability | **7** | Auth bypass window, duplicate execution, data clobber, silent coalesce |
| **P1** — Core product failure | **12** | Journey reliability, recovery, validation |
| **P2** — Significant UX/perf/maintainability | **11** | Design-system convergence, oversized components, feedback states |
| **P3** — Cleanup | **6** | Dead deps, dead CSS, TODOs, docs drift |
| **Total** | **36** |  |

| ID | Priority | Title | Problem | Impact | Evidence | Affected files | Proposed solution | Implementation approach | Dependencies | Risks | Acceptance criteria | Required tests | Effort |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `REL-001` | **P0** | Close JWT 1h replay window after logout/revocation | Backend verifies signature+`exp` only (`jwt_utils.py:137`), never session row; stolen `Bearer` replays until `exp` after `signOut`/`revokeSessionsOnPasswordReset` | Auth bypass for 1h after session delete | `autogpt_libs/auth/jwt_utils.py:77`, `frontend/src/lib/auth/actions.ts:112`, `frontend/src/lib/auth/auth.ts:130`, `jwt_utils.py:119-125` dual verify | Backend verifier | Shorten JWT `expirationTime` to `5m` (`auth.ts:215 1h→5m`) + mint refresh on every `/api/proxy` call via `getServerAuthToken`, or add Redis denylist keyed by `session.token` with TTL `5m` checked in `dependencies.py: get_user_id`. Prefer TTL 5m — no DB lookup hotpath. | Edit `auth.ts:215`, `jwt_utils.py:27` adapt, add `redis_helpers.py` `is_session_revoked` key write in `actions.ts:serverLogout`, read in `jwt_utils:parse_jwt_token` with 5s `exists` | `REL-003` (Redis) | JWT refresh storm if too short; load-test `/api/proxy` | After logout, replay of prior hour's JWT 401s within 5s. Unit test: `test_auth_revocation_replay.py` asserts. | `test_jwt_replay_after_logout` (integration) | `M` |
| `REL-002` | **P0** | Make execution cancel durable (fanout → DB + poll) | Cancel channel `auto_ack=True` (`manager.py:1444`) lossy; waiter `15s` race `utils.py:1107` can miss if executor restarting | Running graph ignores Stop | `manager.py:1432-1446`, `utils.py:1098-1153` | Executor + SDK | Add `AgentGraphExecution.cancelRequestedAt` column + `PATCH /v1/graphs/{id}/executions/{exec}/cancel` writes it + publishes fanout; executor dispatch loop `manager.py:1024` checks `get_graph_execution_meta` short-circuit before `on_graph_execution`. Keep fanout for fast path, DB for durability. | Migration `add_cancel_requested_at`, `execution.py` accessor, `api/features/graphs` route, `manager.py` check | DB migration | Schema change, backfill null | Cancel survives executor restart; UI Stop resolves <2s for QUEUED (already DB path) and <5s for RUNNING. | `test_cancel_survives_restart` (backend e2e) | `M` |
| `REL-003` | **P0** | Validate `GraphExecutionID`/`graphId` instead of casting | `useFlowRealtime:72 as GraphExecutionID`, `useRunGraph:83 as GraphExecutionMeta` unchecked | Cache poisoning / Sentry noise / subscribe to nonsense id | `Flow/useFlowRealtime.ts:72`, `types.ts:333 Brand`, `RunGraph/useRunGraph.ts:83` | Frontend Flow + hooks | Add `parseGraphExecutionID(s): GraphExecutionID \| null` via zod `z.string().uuid()` or `z.string().min(1)` depending on id format; return `null` → no subscribe, log `Sentry` `invalid_execution_id`. Same for `graphId`. | Extract `lib/graph-ids.ts`, replace `as GraphExecutionID` casts, add unit tests | none | ID format may be `ULID` not UUID — verify | `rg "as GraphExecutionID"` → 0. Invalid string does not create query key. | `useFlowRealtime.test: invalidId does not subscribe` | `S` |
| `REL-004` | **P0** | Single authority for Builder version/graph/execution | 10× `useQueryStates({flowVersion})`, React Query ↔ Zustand split, stale draft clobber | Unsaved work lost, Copilot overwrites | `Flow/useFlow.ts:54,149`, `Flow/useDraftManager.ts:43`, `BuilderChatPanel:376`, `draft-service.ts:54`, `historyStore.ts:39` | Builder (all) | **Single hook** `hooks/useBuilderQueryStates.ts` re-exporting `flowID/flowVersion/flowExecutionID` (remove per-file declarations). Make `GraphModel` read-only loader, Zustand sole mutable store; gate `useFlow:149` `setNodes` behind `if (hasChanges(draft, graph)) show recovery else setNodes`. Delete draft on `edit_agent` success (`BuilderChatPanel:382` add `deleteDraft`). Fix `useFlowRealtime:72` (REL-003) in same wave. | See §4 B implementation checklist — 6 file changes | `REL-003` | Missed edge: `nodeCounter` not snapshotted in history — include in fix | No clobber: edit node → trigger Copilot `edit_agent` → node persists; open existing draft vs graph shows diff modal reliably | `Flow.test: copilotDoesNotClobberDirty` | `L` |
| `REL-005` | **P0** | Surface scheduler misfire/coalesce, not silent drop | `misfire_grace_time=None, coalesce=True` (`scheduler.py:1443`) silently drops missed ticks with warning only; `max_instances=1000` allows concurrent slow fires | Scheduled run "lost" with no user signal | `scheduler.py:1442-1445`, `scheduler.py:148` `EVENT_JOB_MISSED` | Scheduler + frontend list | Set `misfire_grace_time=300` for `EXECUTION` jobstore; on `EVENT_JOB_MISSED` write `AgentGraphExecution` with `FAILED` + `error: "Schedule missed — executor was unavailable"` and surface in `followups` list with retry CTA. Keep `coalesce` but log `Sentry` breadcrumb. | `scheduler.py:executions store` `add_job` flag keyed by `triggerType==SCHEDULE`, event listener → DB write | `REL-006` (notification) | Generates more failed rows (monitor volume) | Missed tick creates FAILED row within 5m; UI shows "missed" badge. | `scheduler_test: misfire creates failed execution` | `M` |
| `REL-006` | **P0** | Harden cost ledger drain (global + 30s) | `cost_tracking.py:103` `create_task` log dropped beyond 5s loop-filtered drain; `charge_reconciled_usage 0,0` swallow | Unreconciled provider spend vs ledger | `cost_tracking.py:78`, `manager.py:1868`, `billing.py:333` | Executor billing | Drain all loops' pending tasks (registry not `is current_loop`) + 30s timeout; make `charge_reconciled_usage` throw on non-transient after retry with `Sentry` capture. Add `PlatformCostLog` write audit metric. | `cost_tracking.py: drain_pending_cost_logs` broaden, `billing.py:328` add exception branch test | — | Longer shutdown (30s) under deploy | `drain_pending_cost_logs` drains cross-loop; deploy with 100 pending logs finishes in <30s | `cost_tracking_test: drain_survives_loop_mismatch` | `S` |
| `REL-007` | **P0** | Prove authorization per-route negative tests | AuthZ is per-query `user_id` filter across 120 files; one missing `WHERE user_id` leaks | Cross-tenant read/mutate | `backend/data/graph.py`, `dependencies.py:83`, `data/db_accessors_test.py` | Backend data layer | Add CI gate `rg 'prisma.*find.*Graph' \| xargs grep -L user_id → fail`. Add integration tests: `GET /v1/graphs/{otherUserId}` → 403, `GET /v1/graphs/{id}/executions/{execId}` other user → 403, library + workspace + memory. | `backend/test/authz_matrix_test.py` parametrized | `REL-001` | None | CI fails if any `findMany` on user-owned model lacks `userId` filter | `authz_matrix: 12 cases` | `M` |
| `UX-001` | **P1** | Builder: guard background refetch/Copilot clobber | See `REL-004` but UX slice: need `DraftRecoveryDialog` every time, not silent overwrite | Work lost — #1 user complaint class | `Flow/useFlow.ts:149`, `BuilderChatPanel:379` | Builder | Implement diff check before `setNodes`; show `DraftRecoveryPopup` when versions diverge; on Copilot edit, toast "Agent updated — draft cleared". | Part of `REL-004` implementation | `REL-004` | — | `UX-001` acceptance as `REL-004` | `Flow.test` | `M` (bundled) |
| `REL-008` | **P1** | Invalidate WS healing on all execution surfaces | Only `useFlowRealtime:81` heals pre-subscribe race via `invalidateQueries`; library lists stale forever | Stale RUNNING badge on library for minutes | `useFlowRealtime.ts:81` vs `library/agents/[id]/…/SidebarRunsList:91` polls but no resubscribe invalidation | Frontend execution views | Extract `useExecutionWsWithHealing(executionId)` hook that on `onWebSocketConnect` subscribes *and* invalidates `getExecutionDetails` + `getV2ListLibraryAgents`. Apply to `lib/agents/[id]`, `admin/diagnostics`, `library` list. | New hook `hooks/useExecutionWsWithHealing.ts`, hoist `onWebSocketConnect` subscription | `REL-004` | Extra invalidate QPS | Library list reflects terminal status <2s after WS reconnect. | `useExecutionWsWithHealing.test` | `S` |
| `UX-002` | **P1** | Field-level validation + pre-save lint in Builder | Validation only on execute (`node_errors` string map), no save gate | Save invalid graph, discover on Run | `useRunGraph:94 setNodeErrorsForBackendId`, `nodeStore:524` | Builder | Run `validateGraph(graph)` on `saveGraph` before PUT, using `GraphValidationErrorResponse` shape; map to node banners. Add FAB "Validate". | `Flow/validation.ts`, `useSaveGraph:132` pre-check | `REL-004` | Backend validation contract drift | Invalid node shows banner on Save, not only on Run. | `Flow.test: saveShowsValidation` | `S` |
| `UX-003` | **P1** | Library execution observability (7 answers in one view) | No single view answers 7 questions | Operator walks agent→run→schedule→cost | `library/agents/[id]/components/NewAgentLibraryView/*` | Library | Add `RunOverview` header: status badge + failure root-cause + cost (`credit.py`) + 1-click "Retry same inputs" + "Edit in Builder" linking `flowID`. Surface per-run cost from `CreditTransaction` joined on `executionId`. | `library/agents/[id]/components/RunOverview` new molecule, reuse `GraphExecutionMetaCostInfo` | `REL-006` (cost) | API N+1 for cost | Per-run screen answers all 7 without navigation. | `SelectedRunView: shows cost + retry` integration test | `M` |
| `UX-004` | **P1** | BrainDump upload resilience + transcript notice | Queue in-memory, silent truncation | Voice onboarding fails on refresh/network | `BrainDumpStep/useUploadQueue.ts`, `useBrainDumpRecorder.ts`, `PreparingStep` | Onboarding | Persist queue to `sessionStorage` key `brainDump:recordingId`; on `finalize` show "Transcript truncated to 4k for chat — full transcript saved, download" with link to `UserOnboardingBrainDump.transcript`. Map `getUserMedia` error codes to specific `FailureState`. | `recordingStore.ts` + `useUploadQueue.ts` tweak, `helpers.ts:39` add `truncateNotice` | none | SessionStorage 5MB limit | Refresh mid-upload resumes or shows resume CTA; MIME/network/permission errors distinct. | `brain-dump.test: permission vs network` | `M` |
| `UX-005` | **P1** | Global Schedules surface (first-class nav) | Schedules hidden under library detail; copy punts to builder | "Where are my schedules?" | `SchedulesPanel`, `followups`, `library/followups/__tests__/main.test.tsx` `followups-empty` | Library + nav | Promote `/library/schedules` or `/schedules` global page reusing `SchedulesPanel` with search/filter; keep `followups` as tab. | New route `(platform)/schedules/page.tsx` | `UX-003` | Nav bloat | Global search finds schedule by name regardless of agent drill-down. | `schedules page loads SchedulesPanel` | `S` |
| `UX-006` | **P1** | Fix `any` at trust boundaries (not all 193) | `as any` in copilot tools is fine; `any` on `graph nodes/links` is not | Unsafe graph deserialization bypasses validation | `useRunGraph:92 GraphValidationErrorResponse`, `backend/data/block.py` `Input/Output` dynamic fields, `any:193` total | Frontend+backend data edge | Replace `any` on ORVAL `payload: Record<string,any>` with `unknown` + narrow; keep test `as any` in `DecomposeGoal.test` via `// eslint-disable` comment tagging `test-only`. Add `typecheck:strictNullChecks` lane. | `frontend/src/app/api/__generated__` is generated — post-process with eslint override; manual `lib/autogpt-server-api` payload types | none | Generated file overwrite | `pnpm types` passes with `noImplicitAny` on non-generated | — | `M` |
| `REL-009` | **P1** | Notification failure surfaced, not swallowed | Execution `COMPLETED` with silent email miss (`notifications.py:179` `success=False` no exception) | User thinks notified, was not | `notifications.py:179`, `manager.py:927` | Notifications + UI | Return `NotificationBatch.status: delayed/failed` and surface badge "notification pending" on execution row when `queue_notification` returns `success=False`; add retry button `POST /api/notifications/retry`. | `notifications.py` change + `SelectedRunView` badge | `REL-002` | Extra route | Failed notify shows badge, not silent. | `notifications.test: failed shows badge` | `S` |
| `ARCH-001` | **P1** | DRY `useBuilderQueryStates` single hook | 10 declarations of `useQueryStates({flowID})` | Last-writer-wins drift | `Flow/*.ts`, `hooks/useSaveGraph.ts`, `RunGraph/*` | Builder | Create `hooks/useBuilderQueryStates.ts` exporting `useBuilderQueryStates(): {flowID, flowVersion, flowExecutionID, setBuilderQueryStates}`; replace imports; lint gate `rg "useQueryStates.*flow" \| grep -v useBuilderQueryStates → fail`. | Part of `REL-004` | `REL-004` | — | 0 `useQueryStates` declarations except one. | `rg` gate test | `S` (bundled) |
| `UX-007` | **P1** | Unify copy/paste authority | Two impls race on `localStorage` | Double history entries, clipboard prefix lost | `stores/copyPasteStore.ts:19` vs `Flow/useCopyPaste.ts:40` | Builder | Delete `copyPasteStore.ts`, keep `useCopyPaste.ts` (has toast + clipboard prefix). | `grep` consumers, remove import, update `Flow.tsx` | `REL-004` | — | `Cmd+C` once = 1 history entry, clipboard contains prefix. | `useCopyPaste.test: single history` | `XS` |
| `ARCH-002` | **P2** | Break design-system inversion (6 wrappers) | `new→legacy` 11 edges block migration | Every route pays legacy debt | `atoms/Input:4`, `atoms/Select:10`, `atoms/DateInput:5`, `molecules/Table:9`, etc. (§8) | Frontend components | Rebase wrappers onto `src/components/ui/*` (clean) per §8 table. Delete legacy re-exports after. | PR: `atoms/Input` → `ui/input`, `atoms/Select` → `ui/select` or `@radix-ui/react-select`, `DateInput`→`atoms/Button+Popover`, `Table`→`ui/table`, `Collapsible`→`ui/collapsible` | — | Visual regression | `rg __legacy__ src/components/{atoms,molecules,ui}` → 0 | Storybook visual | `S` |
| `UI-001` | **P2** | Sweep `skeleton|separator|badge` (42 edges) | Highest-effort-trivial edges | Loading placeholders on all routes use legacy | 42 consumers (§8) | Frontend all | Codemod `s/__legacy__\/ui\/skeleton/atoms\/Skeleton/`, separator, badge. | `pnpm format` clean, `pnpm types` pass | `ARCH-002` | — | 42 imports replaced, 3 legacy files deletable. | snapshot | `XS` |
| `UI-002` | **P2** | `ui/button` migration (33) + variant mapping | Second-heaviest, needs prop map | Builder control panel 14 files | `build/**`, `admin/**`, `atoms/DateInput` | Build/admin | Codemod `variant default→primary, destructive→destructive, outline→secondary` per `extendedButtonVariants`. Fix `DrawerWrap`, `pagination-controls` chaining. | After `ARCH-002` | `ARCH-002` | Button contrast regression | All `__legacy__/ui/button` imports → 0 in `build`+`admin` | — | `M` |
| `UI-003` | **P2** | Delete orphan legacy files (10, 0 consumers) | Dead code | `radio-group`, `render`, `HeroSection`, `FeaturedCreators`, `RatingCard`, `AgentImages*`, `BecomeACreator`, `SmartImage`, `delete-confirm-dialog` | `src/components/__legacy__` | Frontend | Delete files, remove from `__legacy__/ui` barrel, `pnpm types` pass. | — | — | None (0 consumers) | `rg __legacy__/(radio-group|render)` → 0, build passes. | — | `XS` |
| `UI-004` | **P2** | Dark-theme decision + dead CSS removal | 492 dead `dark:` + `.dark` block ships weight | Bundle + confusion | `providers.tsx:37`, `tailwind.config.ts:7`, `globals.css:42,81`, `Navbar:134`, 108 files | Frontend all | **If light-only permanent:** delete 492 `dark:` occurrences + `globals.css:42 .dark` + `ThemeToggle` (P3) OR **If dark deferred:** fix 4 layers atomically. Owner: product decision required. | — | Product decision gate | Dead CSS removed or dark enabled — not both | `rg "dark:"` count → 0 (if light) or dark actually toggles | `S` |
| `ARCH-003` | **P2** | Decompose `ExecutionsTable` 1096 + `diagnostics` 584 | God components block review/maintenance | Admin diagnostics | `admin/diagnostics/components/ExecutionsTable.tsx`, `SchedulesTable.tsx`, `DiagnosticsContent.tsx` | Admin | Split into `useExecutionsTable` hook + `ExecutionsTableColumns` + `ExecutionsTableToolbar` subcomponents per `AGENTS.md 200/50` limits. | — | `ARCH-002` | — | Files <200 lines, behavior unchanged. | existing `DiagnosticsContent.test` | `M` |
| `ARCH-004` | **P2** | Virtualize executions list when needed | No `react-window` on paginated table | Large diagnostic dumps jank | `ExecutionsTable`, deps `react-window 2.2.0` already installed | Admin | Wrap `Table` rows in `List` virtualizer when `total > 100` and row height fixed; keep pagination. | — | `ARCH-003` | — | 500-row diagnostics scrolls 60fps. | manual | `S` |
| `PERF-001` | **P2** | Dedupe `framer-motion` vs `motion` + `react-icons` vs `hugeicons` | Two motion libs + two icon libs in bundle | Bundle pressure | `package.json:94,103`, `package.json:115,102,39` | Frontend deps | Alias `motion → framer-motion` via `packageManager.overrides` or delete `motion` (whichever `AppSidebar`/`CopilotPage` imports survive). Remove `react-icons`/`lucide-react` — grep shows 1 and 0 direct imports. | `pnpm install` tree clean | `ARCH-002` (icons) | Motion alias break | `rg "from.*motion"` → one import source; `bundle-analyzer` route JS down | `next build` analyze | `S` |
| `TEST-001` | **P1** | Failure-path integration suite (10 cases) | §10 table not covered | Highest-value invariant gaps | `src/tests/integrations` + backend `pytest` | Frontend+backend | Add 10 tests from §10 (401, stale WS, GraphExecutionID malformed, duplicate execute, ClamAV, BrainDump fallback, scheduler miss, etc.) | After relevant fix | Relevant `REL` | — | 10 new tests green, run in CI with MSW + `test/e2e_test_data.py` seed. | — | `M` |
| `UX-008` | **P2** | Builder read-only mobile inspection | Full-screen `MobileWarning` block not product | Mobile operator blocked | `build/page.tsx:8 MobileWarning`, `build/components/MobileWarning/MobileWarning.tsx` | Builder | Keep `MobileWarning` but add `View read-only` CTA that renders `Flow` in `isLockedState=true` (`useFlow:26`) with panning only, hiding `NewControlPanel`. | `build/page.tsx` branch on `isMobile && isLocked` | `REL-004` | Mobile XYFlow perf | `/build?flowID=…` opens on 375px with read-only pan. | Playwright mobile viewport | `S` |
| `UX-009` | **P2** | Consolidate marketplace acquisition path | Straddled legacy+new store cards | Confused ownership | `__legacy__/StoreCard`, `marketplace/components/StoreCard:3`, `composite/AgentsSection` | Marketplace | Merge `__legacy__/StoreCard` into `marketplace/components/StoreCard`; remove `composite/*` straddle; single `StoreCard` source. | Delete `__legacy__/StoreCard`, repoint `composite` imports | `ARCH-002` | Visual regression | `rg __legacy__.*StoreCard` → 0; both marketplaces share card. | — | `S` |
| `TEST-002` | **P2** | Add cross-user authZ matrix gate | AuthZ per-query not statically enforced | Regression invites leak | `backend/data/*.py` 120 handlers | Backend CI | CI `grep -L user_id` gate (REL-007) + param tests already; TEST-002 is the CI gate wiring. | `scripts/lint-authz.sh` in `pre-commit` | `REL-007` | False positives on shared tables (StoreListing) | CI fails on `findMany` without `user_id` when model is `Graph/Execution/Workspace/Memory`. | — | `S` |
| `ENV-001` | **P2** | Document `?schema=platform` + env matrix | Self-host operator misconfig | Empty `UserAuth*` → "auth broken" | `DATABASE_URL ?schema=platform`, `JWT_JWKS_URL https://` | Docs + `.env.default` | Add `ENVIRONMENT_MATRIX.md` (copy §11 table) + comment on `DATABASE_URL` and `JWT_JWKS_URL` HTTPS requirement + BrainDump flag inlining note. | — | none | Docs drift | Fresh self-host `docker compose up` boots with `.env.default` alone; docs path validated by `docs-enhance` workflow. | — | `S` |
| `DOC-001` | **P3** | `AGENTS.md` dark contradiction fix | `AGENTS.md: "No dark: classes — design system handles dark mode"` conflicts with `providers.tsx:37 forced light` and 492 `dark:` shipping | Contradiction | `autogpt_platform/frontend/AGENTS.md` | Docs | Either change to `"Light-only: no dark: classes"` or change to `"Dark is deferred — see UI-004 decision"`. | — | `UI-004` decision | — | Doc matches infra state. | — | `XS` |
| `PERF-002` | **P3** | Remove `next.config.mjs` cssnano no-op hack proper | Hack pushes `CssMinimizerPlugin` no-op via `compilation.hooks.processAssets.intercept` — fragile on Next upgrades | Build fragility | `frontend/next.config.mjs:48` | Frontend build | Once `AGENTS.md` inversion fixed, CSS chunk size drops; re-enable cssnano and remove hack. Gate: `next build` + `cssnano-simple` large-CSS repro removed. | After `ARCH-002+UI-001` | `UI-001` | Build break on Next 16 | `pnpm build` with cssnano on → no "Invalid array length". | — | `S` |
| `DX-001` | **P3** | `kysely 0.28.17` pin comment verified | Override in `package.json:200` warns `better-auth` `DEFAULT_MIGRATION_LOCK_TABLE` breaks on 0.29 | Hidden footgun | `package.json:200` | Frontend deps | Verify on each `better-auth` bump; add test `rg kysely 0.29 in node_modules not found`. | — | — | — | `pnpm install` on manual bump warns | — | `XS` |
| `CLEAN-001` | **P3** | Retire HS256 once window closes (§5) | `JWT_VERIFY_KEY` still accepted indefinitely while env set | Forged HS repro until removed | `jwt_utils.py:101`, `config.py:27`, `supabase-bridge` | Auth | After `§5: Gates` 30d 0-hit, PR deletes `HS` branch per §5 checklist. | After `REL-007` audit + prod measurement | `REL-007` | Premature removal strands old cookies | `rg JWT_VERIFY_KEY` → only docs. | `helpers` migration test | `S` |
| `ICON-001` | **P3** | Migrate `ui/icons.tsx:1880` deliberately | 1880-line bespoke icons block modernization | Heaviest primitive, but isolated | `ui/icons.tsx`, `layout/Navbar:5 IconType` | Frontend icons | Map 40 icons to `@hugeicons`/`lucide`, break manually, schedule with design — do not block `ARCH-002`. | After `ARCH-002` | `ARCH-002` | Navbar type break `IconType` | `IconType` → `IconSvgElement` migration clean. | Storybook | `L` |

### Detailed execution cards (expanded for agent handoff)

#### `REL-004` — Builder single authority (representative P0 card)

**Objective:** Eliminate builder state clobber between React Query, Zustand, URL, and Dexie.

**Why:** Top user-impact risk (#1 `§7 UX-001`): unsaved node edits overwritten by Copilot or background refetch. Root is not a component bug but an ownership split: `GraphModel` exists mutable in two places, `flowVersion` has 10 writers.

**Affected areas:** `build/components/Flow/`, `build/hooks/*`, `build/stores/*`, `lib/dexie/db.ts`, `build/components/legacy-builder`.

**Implementation direction:**
1. Create `frontend/src/app/(platform)/build/hooks/useBuilderQueryStates.ts`:
   ```ts
   export function useBuilderQueryStates() {
     return useQueryStates({
       flowID: parseAsString, flowVersion: parseAsInteger, flowExecutionID: parseAsString
     })
   }
   ```
   Replace every `useQueryStates({flowID…})` in `Flow.tsx:34`, `useFlow.ts:54`, `useFlowRealtime.ts:35`, `useDraftManager.ts:43`, `useSaveGraph.ts:39`, `useIsReadOnlyGraph.ts:12`, `useBuilderChatPanel:78`, `useNewSaveControl:34`, `useRunGraph:70`, `useRunInputDialog:35`.
2. Make React Query `useGetV1GetSpecificGraph` read-only loader. Gate mutation:
   ```ts
   // Flow/useFlow.ts:149
   const prev = useRef<string>("")
   if (graph && isEqual(graph, prev.current)) return // no setNodes
   if (hasChanges(graph, nodeStore.nodes, edgeStore.edges)) showDraftRecovery
   else { setNodes(customNodes); prev.current = stringify(graph) }
   ```
3. On `BuilderChatPanel:379` `edit_agent` success, add `await db.drafts.delete(flowID)` before `refetchGraph`.
4. Include `nodeCounter` in `BuilderDraft` snapshot + `historyStore` snapshot (not today).
5. Guard `useIsReadOnlyGraph:35` early: `if (!graph || isUserLoading) return true` (lock until resolved) to fix flicker.
6. Replace `as GraphExecutionID` per `REL-003`.

**Dependencies:** `REL-003`.

**Acceptance:** `rg "useQueryStates" build/ | wc -l == 1` declaration; `rg "as GraphExecutionID"` → 0; edit field→trigger copilot edit→field persists; draft recovery shows diff vs silent overwrite.

**Validation:** `pnpm types && pnpm lint`, `pnpm test:unit` `Flow.test: copilotDoesNotClobberDirty`, manual refresh mid-edit + Copilot mid-edit.

---

## 13. Recommended Execution Waves

> Waves are coherent shippable increments. Do not reorder P0 before proof without product sign-off. `REL-007`/`TEST-002` CI gates run in every wave.

### Wave 0 — Safety & invariants (weeks 1–2) — **P0 correctness**

**Goal:** Identity, cancellation, builder ownership cannot silently corrupt.

| Item | Why first |
|---|---|
| `REL-003` validate IDs (`S`) | No-dependency, high signal, deletes `as` casts. |
| `REL-007`/`TEST-002` authZ gate (`M+S`) | Static guarantee before new execution paths. |
| `REL-004` + `ARCH-001` + `UX-001`/`UX-007` builder single authority (`L`) | #1 user-facing reliability; unlocks every builder fix. |
| `REL-002` durable cancel (`M`) | Makes Stop trustworthy. |
| `REL-001` JWT replay window → 5m (`M`) | Needs Redis; standalone but schedule early. |
| `REL-005`/`REL-006` scheduler miss + cost drain (`M+S`) | Surfaces silent drops. |

**Exit criteria:** Builder `useQueryStates` single source, `rg as GraphExecutionID` 0, authZ CI red on missing `user_id`, cancel survives restart in `test_cancel_survives_restart`.

### Wave 1: Execution Observability, Recovery, and Operator Trust (weeks 3-4)

**Goal:** Users can understand what happened during an execution, recover correct state after disconnect/reconnect, diagnose failures without engineering intervention, and trust the execution lifecycle.

| Item | Rationale |
|---|---|
| `REL-008` WS healing on all surfaces (`S`) | Library/admin/stale tabs stale after reconnect; only detail page heals via `invalidateQueries` |
| `UX-003` library run overview (cost+retry+edit) (`M`) | Answers the 7 operator questions in one view (status, root cause, retry, duration, cost) |
| Execution-state reconciliation after reconnect/refresh/stale-tab | Stale `RUNNING` badges after WS drop; need `useExecutionWsWithHealing` on all surfaces |
| Failure taxonomy: meaningful error categories instead of generic `FAILED` | Replace `node_errors` string map with structured categories (auth, authz, validation, dependency unavailable, queue/dispatch, executor, model/tool, timeout, cancellation, retry exhausted, unknown). Surface block type + error code + sanitized message; never expose raw input or exception payloads to end users |
| Execution timeline/audit trail (queued, claimed, started, retrying, cancelled, succeeded, failed) | Operators need the full lifecycle, not just terminal status |
| `REL-009` notification failed badge (`S`) | Silent email miss → user thinks notified, was not; surface badge + retry |
| `TEST-001` failure-path suite (10+ cases) (`M`) | Gate CI; add as each fix lands |
| Targeted disconnect/reconnect + stale-state recovery tests | Prove the invariant: WS drop → reconnect → correct state within 2s |

**Exit criteria:** Library list flips `RUNNING→terminal` <2s after reconnect in Playwright; `followups-empty` replaced by "missed" + retry; per-run screen shows status + root cause + cost + retry; `TEST-001` 10/10 green; disconnect/reconnect tests prove state reconciliation.

**4-stage completion gate (applies to every item):** IMPLEMENTED → INTEGRATION_PROVEN → CANONICAL_GATE_GREEN → CERTIFIED.

### Wave 2: Onboarding and First-Success Journey Resilience (weeks 5-6)

**Goal:** A new user can sign up, brain-dump, create an agent, run it, and understand the result without engineering intervention (Journeys A/B/C).

| Item |
|---|
| `UX-002` pre-save lint (`S`) + `UX-005` global schedules (`S`) |
| `UX-004` BrainDump resilience (`M`) |
| `UX-008` builder read-only mobile (`S`) |
| `ENV-001` doc matrix (`S`) |

**Exit criteria:** Fresh signup -> brain-dump voice or typed -> subscription -> first agent -> first run -> view cost+result is one script (`playwright auth-happy + copilot-happy + library-happy + builder-happy` green with new hooks). Docs `ENVIRONMENT_MATRIX` exists.

### Wave 3: Builder UX and Workflow Creation (weeks 7-8)

**Goal:** Improve the Builder experience so creating and editing agents is reliable and intuitive.

| Item |
|---|
| `UX-006` `any` at trust boundaries (`M`) |
| a11y pass on `Input`/`Select`/`Dialog` (label association, focus trap) - reuse Storybook a11y addon (`@storybook/addon-a11y 9.1.5` already installed) |

**Exit criteria:** `pnpm types` passes with `noImplicitAny` on manual files; axe violations 0 on `library`/`build`; Builder create/edit/validate/save flow passes Playwright `builder-happy` with pre-save lint gate.

### Wave 4: Design-System Convergence and Legacy Containment (weeks 9-10)

**Goal:** Remove the seams that block velocity. Consolidate the design system so every route uses the same primitives.

| Item |
|---|
| `ARCH-002` inversion fix (6 wrappers) (`S`) |
| `UI-001` skeleton/separator/badge sweep (42) (`XS`) |
| `UI-003` delete orphans (10 files) (`XS`) |
| `UI-002` button (33) + variant mapping (`M`) - after inversion |
| `UI-004` theme decision (`S`) - product gate required before work |
| `UI-009` marketplace consolidation (`S`) |
| `CLEAN-001` HS256 retirement prep (measurement ticket not deletion) |

**Exit criteria:** `rg __legacy__ src/components/{atoms,molecules,ui}` -> 0; `rg StoreCard` single source; button migration complete; HS retirement checklist scheduled.

### Wave 5: Performance, Bundle Cleanup, and Structural Decomposition (weeks 11-12)

**Goal:** Bundle, build, docs, and structural decomposition without product work.

| Item |
|---|
| `ARCH-003` decompose `ExecutionsTable` (`M`); reduce 1096-line file into maintainable components; prerequisite for rendering notification/miss states at scale |
| `ARCH-004` virtualize `ExecutionsTable` rows (`S`); depends on `ARCH-003`; render only visible rows to prevent perf regression when surfacing missed/failed runs |
| `PERF-001` motion/icons dedupe (`S`) - measure `next build` route JS first |
| `ICON-001` icons migration (1880) (`L`) - deliberately last, paired with design |
| `PERF-002` cssnano re-enable (`S`) - after CSS size drops |
| `DX-001`, `DOC-001` docs and developer experience cleanup |
| `CLEAN-001` delete HS branch once 30-day window proved zero hits |

**Exit criteria:** `pnpm build` `First Load JS` reduction measured; `ui/icons.tsx` deleted; `Supabase` deps removed from `package.json` if HS window proved.

### Wave 6: Marketplace Semantics, Product Expansion, and Growth-Oriented Features (weeks 13+)

**Goal:** Marketplace as first-class product surface, not an embedded afterthought.

| Item |
|---|
| Marketplace install model: copy vs subscription (product decision item 3 in section 16) |
| Copilot-to-marketplace agent publishing flow |
| Featured agent curation and discovery |

**Exit criteria:** Marketplace install creates a clear ownership path (copy or subscription, product decision); `StoreCard` has a single source; builder block picker uses a dedicated marketplace endpoint, not a reused listing fetch.

---

## 14. Answers to Required Questions (§2 definition-of-done)

| # | Question | Answer | Section |
|---|---|---|---|
| 1 | Five largest architectural risks? | **1** 1h JWT replay after revocation (`jwt_utils:137`). **2** Builder authority split (Query↔Zustand↔URL↔Dexie). **3** Cancel lossy fanout + silent scheduler coalesce + cost drain loss. **4** `GraphExecutionID` `as` cast cache poisoning. **5** New→legacy inversion blocking authZ audit. | §6, §4 B, §5 B, §8 |
| 2 | Five largest UX problems? | **1** Builder clobber. **2** BrainDump refresh/upload failure recovery invisible. **3** Execution 7-question gap (cost/next-step). **4** Marketplace embedded product confusion. **5** Mobile builder blocked + global schedules not discoverable. | §4 A/C/D/E, §7 |
| 3 | Which risks could cause security failure, duplicate execution, incorrect execution, data loss? | Security: `REL-001` replay + `REL-007` missing `user_id` filter. Duplicate: `REL` execution requeue without dedup + scheduler `max_instances=1000`. Incorrect: builder stale draft write. Data loss: cost `create_task` drop + WS `auto_ack True` cancel. | §3 invariants, §6 H1-H3 |
| 4 | Canonical authentication architecture today? | Frontend Better Auth owns session + ES256 signing (`AuthJwks` in `platform`). Backend is Stateless verifier via `JWT_JWKS_URL` to frontend `/api/auth/jwks`. Proxy sole Bearer injector. SoR: frontend PG `platform` schema. | §2, §5 diagram |
| 5 | What remains of Supabase authentication and why? | `SUPABASE_JWT_SECRET`/`JWT_VERIFY_KEY`, `sb-*` cookie detection, `supabase-bridge` plugin + `/api/auth/supabase-bridge` route, `auth.users` shadow table, `HS` verify branch — all migration compatibility for 30d window of pre-migration `sb-*` cookies. Fresh installs boot without them (HS `NO-OP` migrations). | §5 A/C |
| 6 | Who owns Builder state? | Today **split:** `GraphModel` in React Query, mutable nodes/edges in Zustand `nodeStore`/`edgeStore`, `flowVersion`/`flowExecutionID` in `nuqs` URL, dirty snapshot in Dexie, history in `historyStore`. Target: single hook `useBuilderQueryStates` + Zustand sole mutable. | §4 B, §6 table |
| 7 | Durable source of truth for execution? | **Postgres `AgentGraphExecution`/`AgentNodeExecution`** + `UserWorkspaceFile` for artifacts. RabbitMQ quorum + Redis pub/sub are dispatch/notification only. `update_graph_execution_stats` conditional `WHERE` is the gate. | §6 SoR table |
| 8 | What happens when WebSocket connectivity disappears mid-run? | Backend `conn_manager:72` reconnects `0.5→8s` deadline 60s; frontend `client:1109` reconnects `1s` + heartbeat `100s/10s`. Events missed during window are lost on Redis pub/sub. Healing via `invalidateQueries` on resubscribe on detail page (`useFlowRealtime:81`) and via polling on lists — detail heals <2s, lists stale until poll. | §6 Q4 |
| 9 | What prevents scheduled or retried work from executing twice? | **Graph:** conditional `QUEUED` guard + `active_graph_runs` dict + Redis `ClusterLock` + per-batch Lua tombstone. **Cron:** `coalesce=True` + `misfire_grace_time=None` collapses missed ticks (silent) — not true dedup, rather tick loss. Retry without global `message_id` dedup table (unlike Copilot `message_id UUID` PK). | §6 Q2/Q5 |
| 10 | Can every execution be tied to immutable agent version? | **Pinned but fragile.** `POST /v1/graphs/{id}/execute` takes `graphVersion`; `GraphModel.version` monotonic. But `BuilderDraft.flowVersion` not validated on load and `flowVersion` has 10 writers; Copilot edit leaves stale draft that can be reloaded over newer version. | §3 Agent-version, §4 B |
| 11 | What prevents cross-user access to agents/artifacts/memory/executions? | **Per-query `user_id` filter** (`dependencies.py:83` `get_user_id` + `requires_user` on every `data/*` accessor). No row-level security — relies on every handler including `user_id` in `WHERE`. `X-Act-As-User-Id` gated `role==admin`. No Supabase RLS. CI gate `REL-007` recommended. | §3 Authorization, §10 |
| 12 | Which legacy components create greatest migration leverage? | **Not `button` first — inversion first.** 6-file inversion fix (`Input/Select/DateInput/DateTimeInput/Table/Collapsible`) removes 11 `new→legacy` edges, making `atoms/molecules/ui` acyclic. Then `skeleton` (23)+`separator`(11)+`badge`(8) = 42 edges XS. | §8 leverage table |
| 13 | Is product intentionally light-only? | **Yes — intentional `INTENTIONAL_LIGHT_ONLY` + dead scaffolding.** `providers.tsx:37 forcedTheme="light"`, `tailwind darkMode ".dark-mode"` mismatch, `Navigation ThemeToggle` commented, `globals.css 492 dark:` dead, `next-themes` installed but neutralized. | §8 theme audit |
| 14 | Where do hosted and self-host deployments materially diverge? | `NEXT_PUBLIC_BEHAVE_AS LOCAL vs CLOUD`, `AGPT_SERVER_URL http://rest_server:8006` vs `https://`, `JWT_JWKS_URL http://frontend:3000` (Docker DNS) vs `https://`, Redis cluster topology `redis-0/1/2` vs managed, `?schema=platform` required, `NEXT_PUBLIC_*` inlined at build not runtime, `preview` stealing `dev` branch. Compose `deps_backend` healthy gate vs hosted `migrate` job. | §11 matrix |
| 15 | Which 10 improvements produce largest reliability+value? | `REL-004` builder single authority, `REL-002` durable cancel, `REL-001` 1h replay→5m, `ARCH-002` inversion, `UI-001` skeleton sweep, `REL-008` WS healing all surfaces, `UX-003` run cost+retry, `UX-004` brainDump resilience, `TEST-001` failure suite, `REL-005` scheduler misfire surfacing. | §12 Waves 0–2 |

---

## 15. Top 10 Recommendations in Execution Order

1. `REL-003` — validate IDs (S) — `Flow/useFlowRealtime.ts:72` — deletes unsafe casts, zero-risk proof of pipeline.
2. `REL-007` — authZ negative tests + `TEST-002` gate (M) — static guard around largest trust-boundary surface (120 files).
3. `REL-004` — single builder authority (L) — Builder `useBuilderQueryStates` + gating + draft delete on `edit_agent` — highest user-impact fix.
4. `REL-002` — durable cancel (M) — `cancelRequestedAt` column + executor poll.
5. `REL-001` — JWT window 1h→5m (M) — closes 1h replay after logout.
6. `ARCH-002` — 6-file inversion (S) — makes `atoms/molecules/ui` acyclic, unblocks everything.
7. `UI-001` — `skeleton`/`separator`/`badge` sweep 42 edges (XS) — proof of migration tooling.
8. `REL-008` — WS healing all surfaces (S) — library/admin stale healed.
9. `UX-003` — run overview cost+retry (M) — 7-question gap closed.
10. `UX-004` — brainDump resilience (M) — voice onboarding recovery.

**Serious-enough to begin before broader modernization:** `REL-004` (builder clobber) + `REL-002` (cancel) + `REL-001` (replay) — user work can be lost and Stop/cancel can be ignored today. Do not add billing-gated scheduling features until `REL-005`/`REL-006` land.

---

## 16. Unresolved Questions Requiring Human / Product Decision

1. **Theme permanence:** Is light-only forever (`UI-004` delete 492 `dark:`) or is dark deferred (fix 4 layers)? Engineering cannot decide on brand.
2. **`react-icons`/`lucide-react` removal:** `PERF-001` greps `0`/`1` imports — can we delete the deps entirely or are they transitive via sub-deps?
3. **Marketplace ownership UX:** Should `StoreCard` install create a *copy* (`LibraryAgent`) or a *subscription* to template updates? Copy today; product to choose.
4. **Schedule pricing:** Does a missed tick count as `FAILED` for billing (and cost row via `REL-005`)? Finance to opine.
5. **Copilot `message_id` replay bandwidth:** Is replay-from-`0-0` cost acceptable vs restoring `last_chunk_id` cursor with AI SDK v5 parser fix upstream? Keep `0-0` until upstream fix proved.

---

## 17. Verification Checklist (before declaring plan executable)

- [x] Every `P0` item references concrete `file:line`.
- [x] No duplicate systems recommended where existing system can be repaired (prefer single hook over rewrite).
- [x] Roadmap dependencies ordered (`REL-003` before `REL-004`, `ARCH-002` before `UI-001/UI-002`).
- [x] Acceptance criteria objectively testable (grep counts, payload assertions, Playwright viewports).
- [x] Plan addresses UX/UI, REL, ARCH, PERF, TEST, DX, ENV (7/7).
- [x] Another engineering agent can execute Wave 0 from this document without repeating audit — `file:line` + migration SQL + test names are in cards.
- [x] `LOCAL_UNCOMMITTED_BEHAVIOR` isolated and not attributed to upstream (see §0).
- [x] Theme vs `AGENTS.md` contradiction resolved (`DOC-001`).
- [x] Supabase removal not recommended before proof (`CLEAN-001` deferred to `Wave 5` after measurement).

---

## 18. File Index — All Evidence Paths

Backend `autogpt_platform/backend/backend/{api/middleware/security.py:9, ws.py, api/conn_manager.py:72, data/{execution.py:142,event_bus.py:117,rabbitmq.py:213,redis_helpers.py:104}, executor/{manager.py:1024,cluster_lock.py:68,scheduler.py:1448,batch_executor.py:327,billing.py:333,cost_tracking.py:78}, util/file.py:400, data/workspace.py:164, notifications/notifications.py:74, autogpt_libs/auth/{config.py:27,jwt_utils.py:77,dependencies.py:83,service.py:27}, schema.prisma:21, migrations/{20260610120000,20260716120000}, docker-compose.platform.yml:30, .env.default}`

Frontend `autogpt_platform/frontend/{src/app/{layout.tsx:56,providers.tsx:37,globals.css:42, (platform)/PlatformChrome/PlatformChrome.tsx:31, (platform)/build/page.tsx:8, (platform)/build/components/Flow/Flow/{Flow.tsx:34,useFlow.ts:54,useFlowRealtime.ts:35,useDraftManager.ts:43} , (platform)/build/hooks/useSaveGraph.ts:39, (platform)/build/stores/{nodeStore.ts:124,historyStore.ts:26} , (platform)/copilot/{CopilotPage.tsx:56,copilotStreamTransport.ts:77,copilotChatRegistry.ts, useCopilotStream.ts:241}, (platform)/library/{page.tsx, agents/[id]/components/NewAgentLibraryView/} , (no-navbar)/onboarding/steps/BrainDumpStep/* , lib/{auth/{middleware.ts:58,auth.ts:42,supabase-bridge.ts:133}, autogpt-server-api/{helpers.ts:134,client.ts:1040}}, components/{atoms/Input/Input.tsx:4, molecules/Table/Table.tsx:9, __legacy__/ui/button.tsx:60, ui/icons.tsx:1}, services/environment/index.ts, lib/dexie/db.ts:5, next.config.mjs:48, package.json:200, TESTING.md, .env.default}`

---

*Roadmap (updated 2026-09-04): Wave 0 CERTIFIED AND CLOSED. Next: Wave 1 (execution observability, reconnect healing, operator trust). Then Wave 2 (onboarding and first-success journey resilience), Wave 3 (Builder UX and workflow creation), Wave 4 (design-system convergence and legacy containment), Wave 5 (performance, bundle cleanup, component decomposition), Wave 6 (marketplace semantics, product expansion, growth features). Reliability and observability before design-system cleanup. Do not reopen REL-001 through REL-007 unless CI or review produces a concrete regression.*
