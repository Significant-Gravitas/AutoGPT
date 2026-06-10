---
name: feature-planner
description: Produce an architecturally sound implementation plan for an AutoGPT Platform change (backend, frontend, blocks, or cross-cutting). Design for the class of problems, not the single instance. Use when you need a plan that will survive /review-feature-plan without bandaids or workarounds.
user-invocable: true
args: "[task description] — feature, bug fix, or enhancement; may reference issues, endpoints, blocks, or UI."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Feature Planner

Produce an implementation plan for the AutoGPT Platform. Design for the class, not the instance. Never propose bandaids, workarounds, or shortcuts — everything lives in its architecturally correct place.

This skill produces the plan only. The plan-review loop belongs to the caller — when invoked from `/feature-implementer` or `/block-implementer`, the orchestrator owns the loop. When invoked standalone, run `/review-feature-plan` against the plan yourself and iterate until clean.

## Input

A task description: new feature, bug fix, or enhancement. May reference GitHub issues, Linear tickets, API endpoints, blocks, or UI behavior.

## Process

Complete every step. Do not skip any.

### Step 1: Route to domain guides

Determine which guides apply and read each that does **before** planning:

| Guide | When it applies |
|-------|----------------|
| `/add-block` | New blocks or provider integrations (authoritative checklist) |
| `autogpt_platform/backend/AGENTS.md` | Any backend change — code style, layering, testing rules |
| `autogpt_platform/frontend/AGENTS.md` + `frontend/CONTRIBUTING.md` | Any frontend change — component structure, data fetching, design system |
| `docs/platform/block-sdk-guide.md` | Block SDK details (ProviderBuilder, costs, credentials) |
| `docs/platform/new_blocks.md` | Block edge cases (webhooks, files, error handling) |
| `docs/platform/contributing/oauth-integration-flow.md` | New OAuth providers |
| `docs/platform/workspace-media-architecture.md` | File handling, `store_media_file()`, virus scanning |
| `autogpt_platform/backend/TESTING.md` / `frontend/TESTING.md` | Test strategy for the affected surface |

### Step 2: Trace an analogous existing feature

Name an existing feature most similar to the task and trace it end to end — list the actual file paths visited, from entry point (route / block / page) through service, data, and tests. This is how the plan inherits the codebase's patterns instead of inventing new ones. A plan without a real trace is rejected by `/review-feature-plan`.

### Step 3: Reuse building blocks

Consult the existing infrastructure before proposing any new helper:

| Module | What lives there |
|--------|-----------------|
| `backend/util/request.py` (`Requests`) | SSRF-safe HTTP — never raw `httpx`/`requests` for user-influenced URLs |
| `backend/util/file.py` (`store_media_file`) | Media ingest/output, virus scanning, workspace adaptation |
| `backend/sdk/` | Block SDK: `ProviderBuilder`, credentials, costs, webhooks |
| `backend/util/settings.py` | Config and `Secrets` — env-backed, never ad hoc `os.environ` |
| `backend/data/` | Data access layer — user-scoped queries live here, nowhere else |
| `frontend/src/app/api/__generated__/` | Generated API hooks (Orval) — regenerate with `pnpm generate:api`, never extend `BackendAPI` |
| `frontend/src/components/` | Design system (atoms/molecules/organisms) — never `__legacy__` |
| `frontend/src/services/feature-flags/` | LaunchDarkly flag helpers |

Require justification for every new helper: name the existing modules checked and why none fit.

### Step 4: Place logic in the correct layer

- API routes in `backend/api/features/<area>/` stay thin; business logic in services; data access in `backend/data/` with explicit user-ID scoping.
- The executor (`backend/executor/`) owns run semantics — graph/queue behavior never leaks into routes or blocks.
- Frontend: render logic in `.tsx`, behavior in `use*.ts` hooks, pure logic in `helpers.ts`. No business rules in the frontend that the backend must also enforce.
- Schema changes are auth-aware: if a request/response schema is broadened, the plan must show the auth path still holds end to end (who can now send/see what?).
- Prisma schema changes include migration impact: backward compatibility with running deployments, indexes for new query patterns.

### Step 5: Write the plan with every mandatory section

1. **Problem & class coverage** — what class of problems this solves; name at least three concrete cases in the class (or justify why this is genuinely a single case).
2. **Best fix, not merely plausible** — alternatives considered and why this approach wins.
3. **Analogous trace** — the feature traced in Step 2, with file paths.
4. **Building blocks** — existing modules reused; new helpers with justification.
5. **File-by-file changes** — every file to create or modify, with what changes and why.
6. **Data model & migration impact** — Prisma changes, migration safety, or "none".
7. **Auth & security path** — user-ID scoping for `data/` changes, schema-broadening auth check, input validation at boundaries, secrets handling, SSRF.
8. **Test plan** — TDD: which failing test is written first (`@pytest.mark.xfail` backend, integration test frontend); colocated `*_test.py` / `__tests__/`; snapshot updates expected; frontend patch coverage (codecov patch coverage is required to merge).
9. **Rollout** — feature flag if the work spans multiple PRs; PR split if there are multiple concerns (each PR has one purpose).
10. **Out of scope** — what this plan deliberately does not touch.

## Output

Return the full plan in the structure above. Do not start implementing. Do not edit files.
