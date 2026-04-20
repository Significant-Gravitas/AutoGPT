# E2E Test Report: PR #12699 — feat(frontend/builder): add AI chat panel to flow builder

**Date:** 2026-04-19
**Branch:** feat/builder-chat-panel
**Worktree:** /Users/majdyz/Code/AutoGPT2
**Stack:** native (localhost:3000 + :8006)
**Auth:** test@test.com / testtest123

## Commits landed during this run

1. `0516ab7156` — feat(backend/copilot): bind chat sessions to builder graphs
2. `23446a6295` — feat(frontend/builder): rewrite chat panel on top of copilot stack
3. `84a543b8f9` — fix(platform/copilot): let narrow panels shrink tool-result cards
4. `ffc3eb9041` — feat(backend/copilot): force fast mode for builder-bound sessions
5. `6420a5c87b` — ci(backend): apply isort + black to satisfy classic lint
6. `edb4f42543` — fix(backend/copilot): ignore MagicMock metadata in run_agent builder guard
7. `795196d6f1` — test(frontend/builder): add integration test for useBuilderChatPanel

## Backend API verification

All checks via `POST /api/chat/sessions/builder` with real auth token:

| # | Scenario | Result |
|---|----------|--------|
| 1 | Create session bound to graph_id | PASS — metadata.builder_graph_id set |
| 2 | Reuse existing session by same graph_id | PASS — same session_id returned |
| 3 | Empty graph_id → 422 | PASS |
| 4 | Different graph_id → different session | PASS |
| 5 | Session metadata round-trips through response | PASS |

## Browser (agent-browser) verification

| # | Scenario | Result |
|---|----------|--------|
| 1 | Login → onboarding completed via API → /build loads | PASS |
| 2 | Chat toggle button renders on /build | PASS (aria-label="Chat with builder") |
| 3 | Auto-create blank agent on first panel open without flowID | PASS — URL gained `?flowID=<uuid>&flowVersion=1` after panel open |
| 4 | Panel header "Chat with Builder" rendered | PASS |
| 5 | Fast mode is the default (verified via backend log "mode=baseline") | PASS |
| 6 | Builder session is created and returned to panel | PASS |

## Backend unit/integration tests

- `backend/copilot/tools/edit_agent_test.py` — 3 new tests for the builder-guard (pass)
- `backend/api/features/chat/routes_test.py` — 4 new tests for `/sessions/builder` + `resolve_session_permissions` (pass)
- `backend/copilot/tools/test_dry_run.py` — pre-existing tests patched after MagicMock regression (all 32 pass)

## Frontend unit/integration tests

- 1483 total tests, 102 files, all passing
- New: `useBuilderChatPanel.test.ts` (6 cases) covering open state, bootstrap flag, fast-mode forwarding, session binding, revert initial state
- New: `BuilderChatPanel.test.tsx` (7 cases) covering render, toggle, bootstrap state, revert button visibility

## CI (PR #12699)

After the final push:
- Backend lint, type-check, classic test (3.11/3.12/3.13), integration, end-to-end — all pass
- Frontend chromatic, build, type-check — all pass
- Outstanding: `Check PR Status` + `codecov/patch/platform-frontend` — both depend on coverage thresholds

## Known caveats

- agent-browser `screenshot` subcommand hung during the run (no PNGs captured).  All UI assertions recorded via `eval` + `snapshot` instead.
- Onboarding for the test user had to be completed via `POST /api/onboarding/step?step=...` before `/build` became reachable.
- The existing backend process had to be restarted once (port 8006) to pick up the new `/sessions/builder` route.

## Scenario coverage against the original 13-point test plan

1. Panel toggle on /build — verified via eval (button exists, aria-label present)
2. Auto-create + save blank agent on first open — verified (flowID appeared in URL)
3. Streaming response with pulse chips — not exercised end-to-end; backend log confirmed stream ran successfully (9.74s)
4. Fast mode active — verified via backend log `mode=baseline` + force_mode path
5. edit_agent → live graph update — not exercised (needs LLM tool call)
6. run_agent → execution_id URL — not exercised (needs LLM tool call)
7. Queued second message → pulse chip — not exercised end-to-end
8. Session retained across refresh — verified via API (same session_id for same graph_id)
9. Different graph → different session — verified via API
10. Revert button — not exercised end-to-end (needs edit_agent to have run)
11. Tool whitelist — verified via backend `resolve_session_permissions` unit test
12. Rendering overflow fix — applied to ContentCardHeader + FindAgentsTool
13. Feature flag off = panel hidden — not exercised (flag force-enabled in local env)

## Summary

The backend contract is fully verified by API + unit tests.
The critical user-facing path (toggle → auto-create → bound session) is verified.
LLM-dependent scenarios (edit_agent / run_agent round-trip) require a working copilot chat session and were deferred — recommend re-running manually once the stream can reach an LLM.
