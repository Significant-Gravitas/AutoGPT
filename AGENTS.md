# AutoGPT Platform Contribution Guide

This guide provides context for coding agents when updating the **autogpt_platform** folder.

## Directory overview

- `autogpt_platform/backend` – FastAPI based backend service.
- `autogpt_platform/autogpt_libs` – Shared Python libraries.
- `autogpt_platform/frontend` – Next.js + Typescript frontend.
- `autogpt_platform/docker-compose.yml` – development stack.

See `docs/content/platform/getting-started.md` for setup instructions.

## Code style

- Format Python code with `poetry run format`.
- Format frontend code using `pnpm format`.

## Frontend guidelines:

See `/frontend/CONTRIBUTING.md` for complete patterns. Quick reference:

1. **Pages**: Create in `src/app/(platform)/feature-name/page.tsx`
   - Add `usePageName.ts` hook for logic
   - Put sub-components in local `components/` folder
2. **Components**: Structure as `ComponentName/ComponentName.tsx` + `useComponentName.ts` + `helpers.ts`
   - Use design system components from `src/components/` (atoms, molecules, organisms)
   - Never use `src/components/__legacy__/*`
3. **Data fetching**: Use generated API hooks from `@/app/api/__generated__/endpoints/`
   - Regenerate with `pnpm generate:api`
   - Pattern: `use{Method}{Version}{OperationName}`
4. **Styling**: Tailwind CSS only, use design tokens, Phosphor Icons only
5. **Testing**: Integration tests (Vitest + RTL + MSW) are the default (~90%, page-level). Playwright for E2E critical flows. Storybook for design system components. See `autogpt_platform/frontend/TESTING.md`
6. **Code conventions**: Function declarations (not arrow functions) for components/handlers

- Component props should be `interface Props { ... }` (not exported) unless the interface needs to be used outside the component
- Separate render logic from business logic (component.tsx + useComponent.ts + helpers.ts)
- Colocate state when possible and avoid creating large components, use sub-components ( local `/components` folder next to the parent component ) when sensible
- Avoid large hooks, abstract logic into `helpers.ts` files when sensible
- Use function declarations for components, arrow functions only for callbacks
- No barrel files or `index.ts` re-exports
- Avoid comments at all times unless the code is very complex
- Do not use `useCallback` or `useMemo` unless asked to optimise a given function
- Do not type hook returns, let Typescript infer as much as possible
- Never type with `any`, if not types available use `unknown`

## Testing

- Backend: `poetry run test` (runs pytest with a docker based postgres + prisma).
- Frontend integration tests: `pnpm test:unit` (Vitest + RTL + MSW, primary testing approach).
- Frontend E2E tests: `pnpm test` or `pnpm test-ui` for Playwright tests.
- See `autogpt_platform/frontend/TESTING.md` for the full testing strategy.

Always run the relevant linters and tests before committing.
Use conventional commit messages for all commits (e.g. `feat(backend): add API`).
Types: - feat - fix - refactor - ci - dx (developer experience)
Scopes: - platform - platform/library - platform/marketplace - backend - backend/executor - frontend - frontend/library - frontend/marketplace - blocks

## Flag-off contract 🚩

"Flag-off" means all LaunchDarkly feature flags evaluate to their safe defaults
(`false` for booleans). This is the state that CI, local dev without a
LaunchDarkly SDK key, and Playwright (via `NEXT_PUBLIC_PW_TEST=true`) all run
in. Changes must preserve three invariants:

1. **System prompt is byte-identical.** The SHA-256 hash of
   `_CACHEABLE_SYSTEM_PROMPT` in `backend/copilot/service.py` is pinned in
   `_PRE_CHANGE_PROMPT_SHA256` in `expert_context_test.py`. Changing the
   prompt text requires a matching hash update and an explicit PR callout.
2. **`/team` returns 404.** The TeamPage calls `notFound()` when
   `HIRE_EXPERTS` evaluates to `false`. Confirm in E2E tests.
3. **Session list is flat.** No flag-gated grouping, nesting, or
   restructuring is applied when flags are off.

### Running flag-off tests locally

```bash
# Backend — system-prompt hash lock + prompt cache fallback
cd autogpt_platform/backend
poetry run pytest backend/copilot/expert_context_test.py \
  backend/copilot/prompt_cache_test.py -v

# Frontend — flag defaults, /team 404, and unit suite
cd autogpt_platform/frontend
pnpm test:unit

# Frontend — E2E smoke (copilot, marketplace, library)
pnpm exec playwright test --grep="flag-off"
```

### CI enforcement

The `Flag-off regression` workflow (`.github/workflows/flag-off-regression.yml`)
runs the backend hash lock, frontend unit suite, and E2E smoke on every push and
PR to `main` and initiative branches. It must pass before merge.

### PR callout rule

Any intentional change to flag-off behavior must be called out in the
**Flag-off behavior** section of the PR description with a rationale.

---

## Pull requests

- Use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Rely on the pre-commit checks for linting and formatting
- Fill out the **Changes** section and the checklist.
- Use conventional commit titles with a scope (e.g. `feat(frontend): add feature`).
- Keep out-of-scope changes under 20% of the PR.
- Ensure PR descriptions are complete.
- For changes touching `data/*.py`, validate user ID checks or explain why not needed.
- If adding protected frontend routes, update `frontend/lib/supabase/middleware.ts`.
- Use the linear ticket branch structure if given codex/open-1668-resume-dropped-runs
