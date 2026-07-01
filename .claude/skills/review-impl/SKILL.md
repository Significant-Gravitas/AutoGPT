---
name: review-impl
description: Review an implementation in scope — an uncommitted diff, a just-finished agent change, a commit, or named files — for missing or wrong behavior in the AutoGPT Platform. Findings-only architecture and correctness review across backend API, executor, blocks, data/Prisma, and frontend changes.
user-invocable: true
args: "[scope] — diff ref, commit, or file list; defaults to the working tree diff against the branch base. Also provide the original task description and plan if available."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Review Implementation

Review for gaps: things that are missing or wrong. Do not spend findings on style nits, CI-enforced formatting, or a diff recap.

## Workflow

1. Identify the changed surface from the diff, commit, or named files.
2. **Build the evidence map before judging.** For each changed function: read the function itself, at least one caller and one callee, sibling implementations of the same pattern, and the tests that cover it. Do not review from the diff hunks alone — most real findings live in the code the diff *didn't* touch.
3. Classify the surface area: backend API, data/Prisma, executor/queue, blocks, frontend/UI, shared libs, or build/CI/docs.
4. Apply only the relevant lenses below.
5. If the change claims to fix a bug or add behavior, verify it with a discriminating test or runtime check — not just "the code looks right." A fix whose test passes both before and after the change is not verified.
6. Report findings only. Silence means LGTM.

Every verdict must answer: **is this the best fix, or merely plausible?** If a simpler or more idiomatic approach exists in a sibling, that is a finding.

Skip checks CI already enforces: `poetry run format` / `poetry run lint`, `pnpm format` / `pnpm lint` / `pnpm types`, and block registration validation (`test_available_blocks`).

## Universal Lenses

- **Class vs single case:** Does the change cover the reusable class? Name at least three examples in the class. If only one exists, flag a special-case smell.
- **Sibling coverage:** If one site in a class changed, name the siblings that needed the same treatment and verify they were handled or intentionally unaffected.
- **Test adequacy:** Tests must exercise the failure path, not only the happy path. A bug fix must include a test that fails without the fix. Mocks target where the symbol is **used**, not where it's defined; `AsyncMock` for async.
- **Edge cases:** Empty inputs, pagination boundaries, concurrent executions, TOCTOU in file access and credit charging, partial failures mid-`yield`, eliminated/deleted users, retries causing double effects.
- **Idiomatic code:** New duck typing (`hasattr`/`isinstance` dispatch), dicts where Pydantic models belong, linter suppressors, fresh `any` types, inner imports, wildcard exception handling that swallows errors.

## Surface-Specific Lenses

### Backend API

- `Security()` (not `Depends()`) for auth dependencies; input validation at the boundary.
- Changes in `backend/data/*.py` must scope queries by user ID — verify the check exists or the exemption is justified.
- Broadened request/response schemas: walk the auth path end to end — who can now send or see what they couldn't before?
- SSE endpoints: `data:` lines for frontend-parsed events (matching the Zod schema), `: comment` lines for heartbeats.
- New cacheable endpoints must be deliberate: the security middleware default is no-store; additions to `CACHEABLE_PATHS` need justification.
- Error messages sanitize paths (`os.path.basename()`); secrets never logged.

### Data / Prisma / Redis

- Migrations are backward compatible with running deployments (old code + new schema during rollout).
- New query patterns have supporting indexes.
- Redis: cluster-safe single-key operations only — locks via `SET NX`, atomic compare-and-set via single-key Lua; `transaction=True` on pipelines.
- Credit/billing mutations are atomic and idempotent under retry.

### Executor / Queue

- Graph execution semantics stay in `backend/executor/` — not leaked into routes or blocks.
- Messages are idempotent under redelivery; failure paths release resources (DB connections, locks, queue acks).
- Long-running work doesn't hold DB transactions open.

### Blocks

- Block ID is a real UUID4 and unique; class name ends in `Block`; every Input/Output field is a `SchemaField`; boolean inputs have defaults.
- `test_input` / `test_output` / `test_mock` triple present; **mocks mirror the real API's response shape** — a mock that returns a convenient shape the real API doesn't produce is a finding.
- HTTP via the SSRF-safe `Requests` wrapper, never raw `httpx`/`requests` for user-influenced URLs.
- Files via `store_media_file()` with `for_block_output` for outputs.
- Credentials wired end to end: provider `_config.py`, credentials field, test credentials, and (for OAuth) handler registration + `Secrets` env vars.
- Costs registered (`with_base_cost` or `merge_stats` for provider-billed usage).
- Inputs/outputs compose with other blocks in the graph editor — types that nothing can connect to are a finding.

### Frontend

- Generated API hooks (`@/app/api/__generated__/`), never new `BackendAPI` / `src/lib/autogpt-server-api` usage.
- Design system components, never `src/components/__legacy__/`.
- Render logic in `.tsx`, behavior in `use*.ts`, pure logic in `helpers.ts`; function declarations for components.
- New pages/features have integration tests in `__tests__/` (Vitest + RTL + MSW) — codecov patch coverage is required to merge, so missing tests are a blocking finding.
- Protected routes added to `lib/supabase/middleware.ts`.

## Output

Numbered findings, each with: the lens violated, file:line evidence, why it's wrong (not how to restyle it), and severity (blocker / should-fix / nice-to-have). If a full pass over every applicable lens yields nothing, return exactly: **"Implementation review clean — no findings."**

Do not fix anything. Do not commit.
