---
name: openapi-regen
description: Regenerate the OpenAPI spec and frontend API client. Starts the backend REST server, fetches the spec, and regenerates the typed frontend hooks. TRIGGER when API routes change, new endpoints are added, or frontend API types are stale.
user-invokable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# OpenAPI Spec Regeneration

## Steps

1. **Start backend** (background): `cd autogpt_platform/backend && poetry run rest &`
2. **Regenerate client**: `cd autogpt_platform/frontend && pnpm generate:api:force`
3. **Stop backend**: kill the background process
4. **Verify**: `pnpm types && pnpm format`

## Rules

- Always use `pnpm generate:api:force` (not `pnpm generate:api`)
- Don't manually edit files in `src/app/api/__generated__/`
- Generated hooks follow: `use{Method}{Version}{OperationName}`
