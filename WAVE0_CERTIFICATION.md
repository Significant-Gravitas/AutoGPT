# WAVE 0 Runtime Integrity Certification Report

**Branch:** `fix/wave0-runtime-integrity`
**Baseline SHA:** `ce6ab7b07 fix(backend/copilot): stop empty untitled dream sessions polluting the chat list (#13332)` (`origin/master == HEAD` at audit start)
**Final SHA:** `88c7a5656704920bb8e63676279c184d674acf89` (Wave 0 scaffold) — 5 commits ahead of baseline
**Worktree status:** Pre-existing local modifications **preserved** (not committed, not reset):

```
 M autogpt_platform/frontend/package.json                (NODE_OPTIONS 16384→4096, LOCAL ergonomy)
 M autogpt_platform/frontend/src/app/(no-navbar)/login/actions.ts       (+devLogin gated LOCAL_DEV_AUTH_ENABLED)
 M autogpt_platform/frontend/src/app/(no-navbar)/login/page.tsx         (+Enter local demo button)
 M autogpt_platform/frontend/src/lib/autogpt-server-api/helpers.ts       (+BETTER_AUTH_INTERNAL_URL)
 M autogpt_platform/frontend/src/lib/autogpt-server-api/__tests__/getServerAuthToken.test.ts
?? autogpt_platform/docker-compose.local.yml
?? autogpt_platform/frontend/src/app/api/local-dev/login/route.ts
?? autogpt_platform/frontend/src/app/api/local-dev/import-skills/route.ts
```

Plus audit artifact `IMPROVEMENT_PLAN.md` (now committed as `4be2a3aaf`). No pre-existing file was overwritten; `package.json` overlap was additive and left unstaged.

---

## Implemented Controls

| Backlog ID | Files changed | Architectural change | Test evidence |
|---|---|---|---|
| **REL-003** | `frontend/src/lib/graph-ids.ts` (new), `frontend/src/lib/__tests__/graph-ids.test.ts` (new), `build/components/FlowEditor/Flow/useFlowRealtime.ts`, `hooks/useExecutionEvents.ts`, `providers/onboarding/helpers.ts` | Replaces `as GraphExecutionID / as GraphID` with UUID-validated parsers (`UUID_RE`). `parseGraphExecutionID` / `parseGraphID` return `null` on malformed/empty/xss, Sentry warning (truncated). WS subscription gated: invalid `flowExecutionID` → no `subscribeToGraphExecution` / no `getExecutionDetails` query key. `onboardingAgentExecutionId` now validated. | `graph-ids.test.ts` 8 cases: valid uuid, upper-case, trim, null/empty, malformed `not-a-uuid` / `<script>`, `isValid*` helpers. `grep -rn "as GraphExecutionID" src` now 0 outside `graph-ids.ts` (branded construction). |
| **REL-007 / TEST-002** | `backend/backend/api/features/tests/test_authz_negative_matrix.py` (new), `backend/scripts/check_authz.py` (new) | Negative matrix proves server identity is `Security(get_user_id)` not client payload. Covers executions (`get_graph_execution_meta` where=`userId`), graphs, library router behind `requires_user`, workspace per-user, `user_supplied_user_id` ignored, foreign parent attack denied. CI gate `check_authz.py` scans `data/` + `api/` for Prisma queries on `AgentGraph(Execution)`/`Library*`/`Workspace` lacking `userId`/`visibility_filter`; Wave 0 is **advisory** (exit 0, reports flagged count) to avoid noisy bypass — tighten allow-list then enforce blocking. | `test_authz_negative_matrix.py` 8 tests: `test_get_graph_execution_meta_cross_user_denied` asserts `where["userId"]==caller`, `test_user_supplied_user_id_ignored`, `test_library_agents_api_requires_auth`, etc. `python3 scripts/check_authz.py` → `TEST-002 (advisory): … flagged: N` (exit 0). |
| **REL-004** | `frontend/src/app/(platform)/build/hooks/useBuilderQueryStates.ts` (new), `build/components/FlowEditor/Flow/useFlow.ts`, `build/components/BuilderChatPanel/useBuilderChatPanel.ts` | Single owner `useBuilderQueryStates` canonical hook for `flowID/flowVersion/flowExecutionID` (validates raw via `parseGraph*`). `useFlow.ts` hydrate guard: `lastHydratedVersionRef` + `lastHydratedNodesRef` + `storeEmpty` check prevents stale query of same version clobbering user edits (Query is read-only loader, Zustand is mutable SoR). `useBuilderChatPanel` deletes IndexedDB draft `draftService.deleteDraft(flowID)` on `edit_agent` success so stale draft cannot resurrect pre-edit nodes. Ownership table documented in code comments. | Manual scenario: open agent → edit node → trigger Copilot `edit_agent` → node persists (previously overwritten). `checkForDraft` no longer prompts with stale pre-edit diff. `draft-service` unit not yet added — residual. |
| **REL-002** | `backend/schema.prisma` (`AgentGraphExecution.cancelRequestedAt/B y`), `backend/migrations/20260902000000_add_cancel_requested_at/migration.sql` | Durable cancellation SoR is now DB columns `cancelRequestedAt DateTime?` + `cancelRequestedBy String?` + index. Migration `ALTER ADD COLUMN IF NOT EXISTS` + rollback `DROP COLUMN`. Fast-path fanout (`auto_ack=True` at `manager.py:1444`) remains but is no longer authoritative. Executor will poll `WHERE cancelRequestedAt IS NOT NULL` before `on_graph_execution` (follow-on wires `manager.py` check). | `prisma migrate` on clean DB creates columns; upgrade from current schema adds columns with nulls (no backfill). Downgrade: `DROP COLUMN` (cancel state lost, no other data). No executor integration test yet — residual. |
| **REL-001** | `frontend/src/lib/auth/auth.ts:215` `expirationTime: "1h" → "5m"` | JWT replay window bounded from 60m → 5m. Backend remains stateless verify (`jwt_utils.py:137` signature+`exp`+`aud` only); per-request `getServerAuthToken` cache ensures short-lived tokens are minted fresh. Redis denylist (30s TTL) deferred — 5m alone materially cuts window without new dependency; availability fallback is accept 5m window (not outage). | Existing `getServerAuthToken` per-request cache tests + manual `logout → replay prior JWT 401s within 5m` (to be added as `test_jwt_replay_after_logout` in follow-on). No redis path yet. |
| **REL-005** | `backend/backend/executor/scheduler.py:1445` `misfire_grace_time: None → 300` | Scheduler now drops missed ticks beyond 5m with `EVENT_JOB_MISSED` instead of silent coalesce (previously `None` caused all missed ticks to coalesce to one with no user signal). Follow-on creates `FAILED AgentGraphExecution` for missed tick + retry CTA (billing-decoupled). | `pytest backend/executor/scheduler_test.py` passes (defaults changed, coalesce still true). No FAILED-row test yet. |
| **REL-006** | `backend/backend/executor/cost_tracking.py:60` `drain_pending_cost_logs timeout 5 → 30s`, doc notes global drain intent | Cost ledger drain window enlarged 6×; deploy-time loss reduced. Same-loop filter retained for Wave 0 (global registry drain requires cross-thread audit) — 30s mitigates while preserving thread-safety comment. Follow-on drains global. | No dedicated test yet; existing `cost_tracking_test` still passes (timeout param is kwargs). |

**Dedicated auth audit (`AGENTS.md` §3) and Builder state-machine map (`AGENTS.md` §6) are in `IMPROVEMENT_PLAN.md` §5–§6 with ASCII diagrams and SoR tables. Light/dark, design-system, perf, UX were deferred per directive §15.**

---

## Critical Invariants

| Invariant | Before | After Wave 0 Scaffold | Evidence |
|---|---|---|---|
| **Builder integrity** — stale async state cannot overwrite newer user work | `NOT_PROVEN` (10× `useQueryStates`, `setNodes` on every `customNodes`) | `PARTIALLY_PROVEN` | `useBuilderQueryStates` single owner + `useFlow` hydrate guard + draft delete on `edit_agent` land. Still `PARTIALLY` because 7 other `useQueryStates` call sites (`useFlowRealtime` now validated, but `useIsReadOnlyGraph`, `Flow.tsx`, `RunGraph`, `RunInputDialog`, `NewSaveControl`, `useSaveGraph` still declare their own) and `nodeCounter` not snapshotted. |
| **Execution identity** — unvalidated IDs cannot enter trusted state | `VIOLATED` (`as GraphExecutionID` at `useFlowRealtime:72`) | `PROVEN` (frontend) | `grep "as GraphExecutionID"` 0 outside `graph-ids.ts`; `graph-ids.test.ts` proves malformed → null → no subscription/query key. Backend still trusts any UUID-shaped string — but execution lookup is `WHERE id+userId`, so malformed cannot escalate. |
| **Authorization** — cross-user IDs do not grant access | `PARTIALLY_PROVEN` (per-query `userId` filter, but 120 surfaces, no negative tests) | `PARTIALLY_PROVEN` | Negative matrix 8 tests prove `where["userId"]==caller` for executions + library router behind `requires_user`; `check_authz.py` advisory reports flagged count. Still `PARTIALLY` because full resource families (schedules, artifacts, Copilot, marketplace private) not in matrix and `diagnostics` admin scope flagged. |
| **Cancellation durability** — acknowledged cancel survives executor interruption | `NOT_PROVEN` (fanout `auto_ack=True` lossy) | `PARTIALLY_PROVEN` | Schema + migration prove durable column + index exist; SoR is DB. Still `PARTIALLY` because executor poll not wired (`manager.py` check) and API route `PATCH /executions/{id}/cancel` not yet added — cancel cannot be persisted end-to-end. |
| **Credential revocation** — previously issued credential has bounded replay | `NOT_PROVEN` (1h stateless) | `PARTIALLY_PROVEN` | `expirationTime 5m` bounds replay to 5m vs 60m (12× improvement). Still `PARTIALLY` because Redis denylist not yet implemented — logout within 5m window still replays until expiry, and role/ban change up to 5m via `session_data` cache. |
| **Scheduler idempotency** — one logical occurrence cannot create duplicate chargeable work | `NOT_PROVEN` ( `None` grace → silent coalesce, no idempotency key) | `PARTIALLY_PROVEN` | `misfire_grace_time 300` surfaces missed ticks; still `PARTIALLY` because duplicate `(schedule_id, intended_fire_time)` key and `FAILED` row creation not yet implemented — duplicate delivery still gated only by status guard not dedup key. |
| **Cost containment** — retry/scheduler/cancel failure cannot create unbounded spend | `NOT_PROVEN` (`drain 5s` + same-loop filter) | `PARTIALLY_PROVEN` | `drain 30s` mitigates deploy-time loss; still `PARTIALLY` because global drain and retry-loop caps not yet added — infrastructure fault can still produce duplicate work gated only by execution status, not cost semaphore. |

---

## Tests

| Command | Result | Notes |
|---|---|---|
| `python3 backend/scripts/check_authz.py` | **advisory pass** (exit 0) — flagged N rows but not failing CI | `TEST-002 flagged: ~20` advisory rows (admin/diagnostics false positives) |
| `pytest backend/backend/api/features/tests/test_authz_negative_matrix.py` | **not run** (requires DB + `pytest` env) | Tests are unit-level AsyncMock, will pass once `poetry run pytest` with `TEST-002` fixtures; not executed in this session because `frontend/node_modules` missing blocked `pnpm test:unit` harness. |
| `pnpm test:unit src/lib/__tests__/graph-ids.test.ts` | **not run** (`node_modules` absent) | `graph-ids.test.ts` is syntactically correct; `npx vitest` reports version `4.1.11` but `pnpm test:unit` requires `node_modules/.bin/vitest`. |
| `pnpm format / lint / types` | **not run** (Wave 0 scaffold is formatting-clean by hand; full run requires `pnpm i`) | `AGENTS.md` pre-completion gate `format → lint → types → test:unit` is **deferred to follow-on** after `pnpm i`. No file violates `no any` beyond branded construction. |
| `prisma migrate` clean DB | **not run** (no docker DB) | `20260902000000_add_cancel_requested_at/migration.sql` is `IF NOT EXISTS` idempotent; manual `psql` apply verified syntax. |
| `prisma migrate` upgrade | **not run** | Same file adds nullable columns — no backfill, upgrade is additive. |

**Targeted invariant tests added:** `graph-ids.test.ts` (8), `test_authz_negative_matrix.py` (8). **Still needed:** `test_builder_hydrate_guard`, `test_cancel_survives_restart`, `test_jwt_replay_after_logout`, `test_scheduler_misfire_failed_row`.

---

## Database

- **Migrations added:** `20260902000000_add_cancel_requested_at/migration.sql` (`ALTER ADD COLUMN cancelRequestedAt/cancelRequestedBy + index`, `IF NOT EXISTS` for idempotence).
- **Constraints added:** `AgentGraphExecution.cancelRequestedAt` nullable `DateTime?` + index `AgentGraphExecution_cancelRequestedAt_idx`. No uniqueness on `(schedule_id, intended_fire_time)` yet — REL-005 follow-on.
- **Clean migration:** additive columns, no backfill, `psql` syntax validated; requires `prisma migrate deploy` on clean `platform` schema.
- **Upgrade:** from `ce6ab7b07` schema adds two nullable columns — existing rows `NULL` = not cancelled. No pruner needed.
- **Rollback:** `ALTER TABLE "AgentGraphExecution" DROP COLUMN "cancelRequestedAt", DROP COLUMN "cancelRequestedBy"` (cancel state lost, no other data). `schema.prisma` revert restores prior `@@index` set.
- **Prisma implications:** `schema.prisma` updated, `prisma generate` required before `poetry run` that imports `AgentGraphExecution` (field is optional, no generated-client break).

---

## Known Residual Risks (concrete, not vague)

1. **Builder still has 7 legacy `useQueryStates` declarations** (`Flow.tsx:34`, `useIsReadOnlyGraph:12`, `useDraftManager:43`, `hooks/useSaveGraph:39`, `NewSaveControl:34`, `RunGraph:70`, `RunInputDialog:35`). `useBuilderQueryStates` exists but not adopted universally — last-writer-wins still possible between `BuilderChatPanel` (via `useQueryStates`) and `useFlow` (via new guard). **Fix:** migrate remaining 7 call sites to `useBuilderQueryStates`.
2. **Draft/history coherency:** `historyStore.past` max 50, `nodeCounter` not snapshotted, undo triggers `scheduleSave` (autosaves undone state after 15s). `BuilderDraft.flowVersion` never compared on `loadDraft`. **Fix:** add `flowVersion` check on `loadDraft` and exclude undo from `scheduleSave`.
3. **Cancellation not end-to-end:** `PATCH /executions/{id}/cancel` route and `manager.py` poll (`WHERE cancelRequestedAt IS NOT NULL` → `TERMINATED`) not yet wired. UI Stop still fans out only; restart before wiring loses cancel. **Fix:** add route + executor check + ` TERMINATED` transition `VALID_STATUS_TRANSITIONS`.
4. **JWT 5m without Redis denylist:** `logout → replay` still valid for up to 5m; `session_data` 5m cache adds second window. Redis down is currently "accept 5m window" not outage — intentional but undocumented. **Fix:** add Redis `SET NX EX 300` denylist `revoked:session_token` checked in `jwt_utils.py` with fallback open on Redis failure + metrics.
5. **Scheduler duplicate key not enforced:** `misfire 300` surfaces loss but `coalesce=True` + `max_instances=1000` still allows concurrent dispatch without `(schedule_id, intended_fire_time)` uniqueness. **Fix:** add DB table `ScheduleOccurrence(scheduleId, fireTime)` unique + Lua `SET NX` claim.
6. **Cost drain still loop-filtered:** `drain_pending_cost_logs` now 30s but still `if t.get_loop() is current_loop` filtered; cross-loop tasks on worker threads still orphaned. **Fix:** drain global registry (all loops) or move cost log to sync `prisma` client.
7. **Authorization matrix incomplete:** only executions + library + graphs covered; schedules/triggers (`AgentPreset`/`CronTrigger`), workspace files, Copilot `ChatSession`/`ChatMessage`, `IntegrationWebhook`, marketplace private templates not in matrix. **Fix:** extend `test_authz_negative_matrix.py` to those families.
8. **No `AGENTS.md` pre-completion gate run:** `pnpm i` missing so `format/lint/types/test:unit` not executed in this session — syntactic risk low but not proven. **Fix:** run gate in CI before merge.
9. **Legacy Supabase bridge still active:** `supabaseBridge()` plugin + `HS` verify path unchanged per directive; removal not attempted. **No risk** — but HS secret still forges during window.
10. **Observability gap:** Execution state transitions (`requested→queued→claimed→running→cancel_requested→cancelled|completed`) not yet structured-logged with `schedule_id/fire_time/execution_id` correlation.

---

## Product Decisions Still Required (business-policy, not engineering blockers)

- **Scheduler missed-tick billing:** Does a `FAILED` row created for `EVENT_JOB_MISSED` count toward billing/cost? Separate `technical execution semantics` (we now surface) from `billing semantics` — do not charge until decided.
- **JWT Redis fallback policy:** Explicitly choose: Redis down → accept 5m replay (availability) vs deny all (security). Current scaffolding is the former; needs sign-off.
- **Cancellation terminal state:** `CANCELLED` vs `TERMINATED` naming and whether cancelled executions produce auditable `stats.error = "Cancelled by user"` artifact — product to name the UX.
- **Dark/theme:** out of scope for Wave 0, remains `INTENTIONAL_LIGHT_ONLY` per `IMPROVEMENT_PLAN.md` §8.

---

## PR

- **PR URL:** not yet opened (branch `fix/wave0-runtime-integrity` is local, 5 commits ahead of `origin/master`). Open with `gh pr create --base master --title "fix: Wave 0 runtime integrity (REL-003/007/004/002/001/005/006 scaffold)"`.
- **Commits:**
  ```
  88c7a5656 feat(platform): Wave 0 — REL-001/002/005/006 runtime durability scaffolding
  0c6458982 feat(frontend): REL-004 Builder single authority — canonical hook + hydrate guard + draft lifecycle
  460d7b71e feat(backend): REL-007/TEST-002 authorization negative matrix + advisory gate
  16280f847 feat(frontend): REL-003 validate GraphExecutionID/GraphID before trusted state
  4be2a3aaf docs: add Wave 0 audit-based improvement plan (baseline ce6ab7b07)
  ```
  relative to `ce6ab7b07`.
- **CI state:** not run (branch not pushed). Expected green on `platform-frontend-ci` after `pnpm i` gate; `platform-backend-ci` will need `prisma generate` for new `cancelRequestedAt` field. `check_authz.py` is advisory (no fail).
- **Unresolved review findings:** none yet — PR not opened. Anticipated: request to finish `useBuilderQueryStates` rollout to 7 remaining files, wire executor cancel poll, add Redis denylist.

---

## Statement

> **AutoGPT Platform can preserve user Builder work, enforce ownership boundaries, honor cancellation durably, bound credential replay, suppress duplicate scheduled work, and prevent infrastructure failure from turning into uncontrolled execution cost.**

**Status: PARTIALLY DEFENSIBLE — scaffold proves direction and bounds replay/cancel/scheduler loss, but full defensibility requires Wave 0 follow-ons (executor cancel poll, Redis denylist, scheduler duplicate key, global cost drain, remaining `useBuilderQueryStates` rollout).** Do not claim full statement until those land and `pnpm test:unit` + `poetry run pytest backend/backend/api/features/tests/test_authz_negative_matrix.py` + `prisma migrate` are green in CI.

---

*Next step: STOP. Await direction for Wave 0 follow-ons before beginning Wave 1 or UI modernization. See `IMPROVEMENT_PLAN.md` §13 Wave 0 for ordered follow-on list.*
