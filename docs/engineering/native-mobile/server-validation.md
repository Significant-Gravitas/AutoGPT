# Native mobile server validation

This records executable server boundaries for SECRT-2881. It is not simulator or provider certification. The experiment's scope and delivery gates remain in [the implementation plan](implementation-plan.md).

## Native email sessions

The test harness starts the production Better Auth handler over loopback HTTP, with the real PostgreSQL adapter, plugins, signup policies, password verification, and platform-user provisioning. It does not replace the auth database or JWT signer. The backend request uses the production FastAPI app, JWT verification against the harness JWKS endpoint, and the same PostgreSQL session rows.

Email sign-in returns a signed `set-auth-token` header. A native client stores that session credential in Keychain, partitioned by the selected server and account. It sends the credential only to that server's auth service. `GET /api/auth/token` exchanges it for the short-lived JWT used at the backend. The raw unsigned token in the sign-in JSON is not accepted as a bearer credential.

New JWTs contain the auth session's `sid`. The backend checks that the session still exists, belongs to the JWT subject, has not expired, and has an account without an active ban. Deleted sessions fail with 401; an unavailable session lookup fails with 503 and `Retry-After`. WebSocket authentication applies the same check when opening a connection. Existing streams are not forcibly terminated by this change. Tokens without `sid` retain their existing signature and expiry policy for compatibility during rollout.

The native auth journeys cover email sign-in, obtaining another access token, sign-out invalidating a previously issued JWT, session expiry, an account ban, mismatched session ownership, lookup failure and recovery, and rejection of an unsigned session bearer. Production session validation is bound to the FastAPI application's state, independently of the test dependency override registry. The sign-out journey also runs after that registry is cleared, reproducing the teardown used by existing route tests. Email verification and social-provider policy still use the production auth configuration; Google browser return, account/server switching in the native client, and real provider testing are subsequent gates.

## Running the server journeys

Install the locked backend and frontend dependencies, generate Prisma, and migrate a dedicated PostgreSQL database whose name contains `native_mobile`. Do not point these tests at an application database. Set `DATABASE_URL` and `DIRECT_URL` privately to that database with `schema=platform`; set `REDIS_CLUSTER_HOST` and `REDIS_CLUSTER_PORT` to a test Redis cluster. Use Node 24 on `PATH`.

From `autogpt_platform/backend`:

```sh
poetry run pytest test/native_mobile/auth_journey.py -q
poetry run pytest backend/copilot/pending_retries_db_test.py backend/api/features/chat/pending_retries_test.py -q
```

The auth fixture starts its own auth process on an available loopback port and stops it afterward. It uses a stable, public test-only auth secret because the auth signing keys are encrypted in PostgreSQL and reused across test runs. Its generated accounts are deleted after each journey. The runtime harness is `frontend/scripts/native-auth-server.ts`; it uses the locked `tsx` dependency. No Google, Apple, APNs, Stripe, or live model credentials are needed for these cases.

`platform-native-mobile-ci.yml` runs these journeys explicitly on `ubuntu-latest`, using PostgreSQL with pgvector and a three-node Redis cluster. It installs the locked frontend and backend dependencies concurrently, caches dependency downloads/environments, and avoids a Next.js production build or full application seeding. The native auth file is named `auth_journey.py` so backend-only jobs do not collect tests requiring Node and the frontend installation. This dedicated lane invokes it explicitly. The job records cache hits and publishes JUnit results.

The [first GitHub run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38154751772/job/114515310236), at `8ddfe5627f108e080286b916fed95e7ec86cb833`, passed 20 journeys in a full job duration of **4m55s** (11:16:15–11:21:10 America/Chicago on 2026-10-11). Both dependency caches were absent. The [subsequent warm run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38156630654/job/114522397959) restored both caches and passed 21 journeys, including the override-cleanup regression, in **3m28s** (11:50:58–11:54:26 America/Chicago on the same date). These are measured historical server-lane results below the ten-minute target; they do not establish timings for later dependency changes. iOS simulator build/runtime measurements are separate and remain open.

The independent foundation landing criteria and client-facing contract are recorded in the [backend handoff](backend-handoff.md).

## Pending-message evidence and remaining limits

The retry regressions exercise concurrent HTTP retries against real Redis, a full queue, internal producer overflow, receipt expiry while a message remains queued, and expiration of a restored receipt with its queue. Full buffers reject new admission instead of evicting accepted messages. Restored receipt lifetime follows the remaining queue lifetime.

PostgreSQL cases exercise duplicate pending IDs, an already-persisted ID with stale in-memory history, a genuine primary-key collision from a competing writer followed by rollback/requeue recovery, and HTTP admission through persistence and a retry after the Redis receipt is removed. Database lookups and ownership checks run unchanged; the pending HTTP fixture supplies test authentication.

Redis remains an expiring buffer. Cancellation after receipt expiry or Redis loss is not durably fenced by the current implementation. Rollback can also fail to requeue if the buffer fills before recovery. These are open requirements before enabling automatic native outbox replay. A durable discard ledger alone does not close Redis loss before cancel or database failure while recording the discard. The native cancellation contract must account for delayed requests crossing a cancellation and a subsequent turn; these cases remain part of the implementation and review gates.
