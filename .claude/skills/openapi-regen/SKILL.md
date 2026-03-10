---
name: openapi-regen
description: Regenerate the OpenAPI spec and frontend API client. Starts the backend REST server, fetches the spec, and regenerates the typed frontend hooks. TRIGGER when API routes change, new endpoints are added, or frontend API types are stale.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# OpenAPI Spec Regeneration

## Steps

1. **Run end-to-end** in a single shell block (so `REST_PID` persists):
   ```bash
   cd autogpt_platform/backend && poetry run rest &
   REST_PID=$!
   WAIT=0; until curl -sf http://localhost:8006/health > /dev/null 2>&1; do sleep 1; WAIT=$((WAIT+1)); [ $WAIT -ge 60 ] && echo "Timed out" && kill $REST_PID && exit 1; done
   cd ../frontend && pnpm generate:api:force
   kill $REST_PID
   pnpm types && pnpm lint && pnpm format
   ```

## Rules

- Always use `pnpm generate:api:force` (not `pnpm generate:api`)
- Don't manually edit files in `src/app/api/__generated__/`
- Generated hooks follow: `use{Method}{Version}{OperationName}`
