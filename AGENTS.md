# AutoGPT Platform Contribution Guide

This guide provides context for Codex when updating the **autogpt_platform** folder.

## Directory overview

- `autogpt_platform/backend` – FastAPI based backend service.
- `autogpt_platform/autogpt_libs` – Shared Python libraries.
- `autogpt_platform/frontend` – Next.js + Typescript frontend.
- `autogpt_platform/docker-compose.yml` – development stack.

See `docs/content/platform/getting-started.md` for setup instructions.

## Code style

- Format Python code with `poetry run format`.
- Format frontend code using `pnpm format`.

## Frontend-specific guidelines

**When working on files in `autogpt_platform/frontend/`, always read and follow the conventions in `autogpt_platform/frontend/CONTRIBUTING.md`.**

Key frontend conventions:
- Component props should be `interface Props { ... }` (not exported) unless the interface needs to be used outside the component
- Separate render logic from business logic (component.tsx + useComponent.ts + helpers.ts)
- Colocate state when possible and avoid create large components, use sub-components ( local `/components` folder next to the parent component ) when sensible
- Avoid large hooks, abstract logic into `helpers.ts` files when sensible
- Use function declarations for components and handlers, arrow functions only for small inline callbacks
- No barrel files or `index.ts` re-exports

See `autogpt_platform/frontend/CONTRIBUTING.md` for complete frontend architecture, patterns, and conventions.

## Testing

- Backend: `poetry run test` (runs pytest with a docker based postgres + prisma).
- Frontend: `pnpm test` or `pnpm test-ui` for Playwright tests. See `docs/content/platform/contributing/tests.md` for tips.

Always run the relevant linters and tests before committing.
Use conventional commit messages for all commits (e.g. `feat(backend): add API`).
  Types:
    - feat
    - fix
    - refactor
    - ci
    - dx (developer experience)
  Scopes:
    - platform
      - platform/library
      - platform/marketplace
      - backend
        - backend/executor
      - frontend
        - frontend/library
        - frontend/marketplace
      - blocks

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
