# Proxy + Supabase auth — follow-up plan

This file captures the cleanup items left over after the high-impact PR
(`fix(frontend): stream proxy bodies and skip middleware auth on /api/proxy`).
The high-impact PR landed:

- excluding `/api/proxy/*` from the middleware matcher,
- replacing the per-content-type proxy with a single stream-through path,
- propagating backend status codes and a filtered set of response headers,
- removing the JSON parse → restringify and FormData buffering round-trip,
- (as a side-effect) removing the empty-body POST that became literal `"null"`.

What is left below is grouped roughly by impact. The order inside each group is
the order I would tackle them in.

---

## Medium impact

### 1. Per-request memoisation of `getServerSupabase` / `getServerAuthToken`

**Files:** `src/lib/supabase/server/getServerSupabase.ts`,
`src/lib/autogpt-server-api/helpers.ts`.

After the matcher fix, the only remaining duplication is inside a single proxy
invocation: the route handler calls `getServerAuthToken()`, and on the server-
side `customMutator`/`BackendAPI` paths there can be a second call within the
same request. Wrap both with React's `cache()` so the supabase client and the
session decode happen once per request:

```ts
import { cache } from "react";
export const getServerSupabase = cache(async () => { /* … */ });
```

Risk: `helpers.ts` is dynamically imported by the browser-side `BackendAPI`
client. `cache()` only works on the server, so wrap *only* the symbols defined
in `lib/supabase/server/*` (which is server-only by convention). Verify
`getServerAuthToken` lives in a server-only module before wrapping; if not,
move it.

### 2. Drop the `"no-token-found"` sentinel string

**Files:** `src/lib/autogpt-server-api/helpers.ts`,
`src/app/api/proxy/[...path]/route.ts`,
`src/app/api/transcribe/route.ts`,
`src/app/api/workspace/files/upload/route.ts`,
`src/app/api/mutators/custom-mutator.ts`,
`src/lib/autogpt-server-api/client.ts`,
`src/lib/autogpt-server-api/__tests__/helpers.test.ts`.

`getServerAuthToken()` returns the literal `"no-token-found"` instead of
`null`, and every caller does `if (token && token !== "no-token-found")`.
Change the return type to `Promise<string | null>` and let callers do a plain
truthiness check. Update the helpers test that asserts the sentinel
specifically. The defensive check can stay in `createRequestHeaders` for one
release as a no-op safety net, then be removed.

### 3. Consolidate the two browser-side Supabase clients

**Files:** `src/lib/supabase/hooks/helpers.ts::ensureSupabaseClient`,
`src/lib/autogpt-server-api/client.ts::getSupabaseClient`.

There are two `createBrowserClient(...)` calls in the browser bundle: a
singleton in the supabase hooks layer, and a per-instance one inside
`BackendAPI`. Both pass `persistSession: false`, so they're functionally
equivalent, but each one builds its own internal state and listeners.
`BackendAPI.getSupabaseClient()` should reuse `ensureSupabaseClient()`.

### 4. Stop running session validation on both `focus` and `visibilitychange`

**File:** `src/lib/supabase/hooks/useSupabaseStore.ts` (look at
`handleFocus` / `handleVisibilityChange`).

Returning to a tab fires `visibilitychange` and `focus` near-simultaneously.
The 2-second `lastValidation` debounce catches it most of the time, but a slow
network can let both fire. Pick one — the cleanest choice is
`document.visibilitychange` with a `document.visibilityState === "visible"`
guard, and remove the `focus` handler.

### 5. Replace the `useEffect` in `useSupabase` with mount-once + path setter

**File:** `src/lib/supabase/hooks/useSupabase.ts`.

```ts
useEffect(() => {
  void initialize({ api, router, path: fullPath });
}, [api, initialize, fullPath, router]);
```

Re-runs `initialize` on every navigation. The store has guards so it doesn't
*do* much each time, but it still re-attaches/cleans listeners and triggers
extra work. Refactor to:

- Run `initialize` once via `useMountEffect`.
- Expose a separate `setCurrentRequestContext({ api, router, path })` action
  and call that on path change, which updates `currentPath`/`routerRef`/
  `apiRef` without re-running init.

Conforms to the global rule against direct `useEffect` in components.

### 6. Narrow `revalidatePath` calls in `actions.ts`

**File:** `src/lib/supabase/actions.ts` (`serverLogout`, `refreshSession`).

`revalidatePath("/", "layout")` invalidates the entire app's RSC cache. After
logout the client-side already does `queryClient.clear()` and the storage
listener does `router.refresh()`. The layout-level revalidation is redundant
and forces a fresh server roundtrip per route segment. Drop it from
`refreshSession`; in `serverLogout`, replace with a more targeted
`revalidatePath("/")` (current page only) — or remove if `router.refresh()`
is already covering us.

---

## Low impact / code-quality

### 7. `createResponse` is dead code

**File:** `src/app/api/proxy/[...path]/route.ts`.

The high-impact PR already inlined the only successful exit. The 204 branch in
the old `createResponse` was unreachable (callers always passed 200). Confirm
nothing else imports the helper, then delete.

> Already removed in the high-impact PR; left here so the next pass can grep
> for any straggler imports.

### 8. `getServerUser` mismatched contract + dead comments

**File:** `src/lib/supabase/server/getServerUser.ts`.

Returns `{ user: null, error }` (no `role`) in the early-return branch,
`{ user, role, error: null }` in the success branch. Has commented-out code
left over from a prior debug pass. Doesn't use the `error` from
`auth.getUser()`. `actions.ts::getCurrentUser()` already covers the same
ground — delete this file or unify the two.

### 9. `getServerSupabase` no-op try/catch + dynamic require

**File:** `src/lib/supabase/server/getServerSupabase.ts`.

```ts
try { return createServerClient(...) } catch (error) { throw error }
```

Pure no-op. Drop the try/catch. The `require("next/headers")` workaround can
also be replaced with a static import if the file is moved into a clearly
server-only path (it's already under `lib/supabase/server/`, so it should be
safe).

### 10. Edge runtime for the proxy route

**File:** `src/app/api/proxy/[...path]/route.ts`.

After the streaming refactor, the proxy has no Node-only deps (it pulls
`@supabase/ssr` and `next/headers`, both Edge-compatible). Adding
`export const runtime = "edge";` would cut cold-start to single-digit ms
per region. Validation steps:

1. Ensure `@sentry/nextjs` server-action instrumentation is not applied to
   route handlers (it normally only wraps server actions; if it does wrap
   route handlers, the proxy can't use Edge).
2. Check that the workspace download buffering path's
   `arrayBuffer()` works on Edge — should be fine, it's standard Web API.
3. Run `pnpm test:unit` and a manual smoke test against a deployed preview.

### 11. Dynamic imports inside `BackendAPI` methods

**File:** `src/lib/autogpt-server-api/client.ts` (~6 `await import("./helpers")`
sites).

The `await import("./helpers")` pattern exists because `helpers.ts` has
top-level imports of server-only modules (e.g. `getServerSupabase`). Cleanest
fix: split `helpers.ts` into `helpers.client.ts` and `helpers.server.ts`, or
add `import "server-only"` markers and use static imports. After the split,
the dynamic-import dance disappears.

### 12. Split upload routes from regular proxy for `maxDuration`

**File:** `src/app/api/proxy/[...path]/route.ts`.

`maxDuration = 300` exists for 256 MB uploads. Most calls finish in <1s.
Split uploads to `/api/proxy-upload/[...path]` so the regular proxy can run
with `maxDuration = 30`, which (on Vercel) means tighter scaling and cheaper
worst-case bills.

### 13. `BackendAPI` deprecation tracker

**File:** `src/lib/autogpt-server-api/client.ts` and call sites.

CONTRIBUTING.md marks `BackendAPI` as deprecated, but `useBackendAPI` is still
wired into `useSupabase` for WebSocket / logout flows. Open a ticket to
migrate the WebSocket transport off `BackendAPI` so the legacy HTTP machinery
(`_makeClientRequest`, `_makeServerRequest`, `_makeClient/ServerFileUpload`,
etc.) can be deleted entirely.

### 14. Add proxy route handler unit tests

**File:** `src/app/api/proxy/[...path]/__tests__/route.test.ts` (new).

Today only `route.helpers` (workspace download path detection) is tested. Add
handler-level tests for:

- 200 / 201 / 202 / 204 propagated correctly.
- Backend `Content-Disposition` / `Cache-Control` / `Location` reach the
  client.
- Hop-by-hop headers (`Transfer-Encoding`, `Connection`) are filtered out.
- Request body is forwarded byte-for-byte (a JSON POST with a `Date` field
  arrives at the backend with the original ISO string, not re-stringified).
- 5xx from backend surfaces as 5xx from proxy (not flattened).
- Workspace download path still buffers and short-circuits with the dedicated
  retry helper.

This suite is what makes future refactors safe.

---

## Out of scope here, worth a separate pass

- The `chat/sessions/[sessionId]/stream/route.ts` SSE proxy still does
  `await request.json()` and rebuilds the body — same parse/restringify smell
  as the old generic proxy. Worth the same streaming treatment, but the SSE
  contract makes it a bigger change.
- The `workspace/files/upload/route.ts` route does `await request.formData()`
  before forwarding. Same fix as the proxy refactor — pipe `req.body` through
  with the multipart Content-Type header preserved.
- Backend-side: the cache-protection middleware (`backend/api/middleware/security.py`)
  sets `Cache-Control: no-store`. After the high-impact PR that header now
  reaches the browser, but worth a quick audit to confirm the allow-list is
  still aligned with what the proxy passes through.

---

## Suggested PR splits

1. **Per-request auth memoisation + sentinel cleanup** (items 1, 2). Small,
   self-contained, near-zero risk.
2. **Browser supabase consolidation + `useSupabase` refactor** (items 3, 4, 5).
   Touches the auth state machine, deserves its own review pass.
3. **`revalidatePath` narrowing + `getServerUser` cleanup** (items 6, 8, 9).
   Pure cleanup PR.
4. **Edge runtime + module split** (items 10, 11). Performance / DX.
5. **Upload route split + maxDuration** (item 12).
6. **Proxy route unit tests** (item 14). Land *before* item 4 above so the
   refactor is caught by the new suite.

Optional: items 13 and the SSE/upload streamings can ride alongside whichever
PR is touching the same area.
