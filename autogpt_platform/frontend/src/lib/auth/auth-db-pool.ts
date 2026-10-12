import { Pool } from "pg";

// Within the single-container appliance's frontend role limit (20, set in
// single-container/bootstrap.sh) with room to spare: Postgres counts a
// connection until its backend exits.
export const AUTH_DB_POOL_MAX = 10;

// One pool per process, shared by Better Auth and its database hooks, cached
// on globalThis. A production build evaluates this module more than once:
// route handlers and the pages' server actions are bundled separately, each
// with its own copy. Without the cache every copy opened its own pool, which
// could take the appliance's frontend role past its connection limit, so
// concurrent sign-ups failed with "too many connections". `next dev` also
// re-evaluates the module on every hot reload, where a fresh pool leaked its
// idle connections.
const globalForAuthDb = globalThis as { __authDbPool?: Pool };

export const authDbPool =
  globalForAuthDb.__authDbPool ??
  new Pool({
    // Fallback matches the docker-compose db service so `make run-frontend`
    // works against `make start-core` without a frontend/.env, mirroring the
    // localhost fallbacks in services/environment. Production must set
    // DATABASE_URL explicitly.
    connectionString:
      process.env.DATABASE_URL ||
      "postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:5432/postgres", // pragma: allowlist secret
    // Better Auth shares the platform Postgres; its tables live in the same
    // schema as the Prisma-managed ones (created by the backend migrations).
    options: `-c search_path=${process.env.AUTH_DB_SCHEMA || "platform"}`,
    max: AUTH_DB_POOL_MAX,
    // A request waiting this long for a connection fails instead of hanging,
    // and a token it could not mint becomes a retryable 503.
    connectionTimeoutMillis: 10_000,
  });

globalForAuthDb.__authDbPool = authDbPool;
