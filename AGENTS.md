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
