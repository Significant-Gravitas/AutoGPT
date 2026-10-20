/**
 * Seeds the shared preview-environment login accounts into the Better Auth
 * tables of a (branch) database.
 *
 * This lives in the platform repo — next to the schema — because the seed
 * must name real tables (`UserAuthIdentity` / `UserAuthAccount`) and hash
 * passwords exactly the way the frontend verifies them (bcrypt cost 10, see
 * src/lib/auth/auth.ts). A seed maintained anywhere else silently breaks the
 * moment either of those changes; the preview pipeline should call this
 * script instead of carrying its own SQL.
 *
 * Usage (the preview CD pipeline runs this after `prisma migrate deploy`):
 *   DIRECT_URL=postgresql://... PREVIEW_ACCOUNTS_PASSWORD=... \
 *     npx --yes tsx scripts/seed-preview-accounts.ts
 * (tsx is not a package dependency; npx fetches it, matching the other
 * scripts in this folder. Prefer DIRECT_URL; a DATABASE_URL fallback must
 * support an explicit transaction on one checked-out client.)
 *
 * Behavior:
 *   - Idempotent: existing identities are kept (matched by email; role and
 *     emailVerified are converged to the roster), and an existing credential
 *     account is never overwritten — rotating PREVIEW_ACCOUNTS_PASSWORD only
 *     affects freshly seeded databases.
 *   - Exit codes are the cross-repo contract with the preview CD pipeline —
 *     see the named constants in seed-preview-accounts.orchestration.ts.
 */
import { hash } from "bcryptjs";
import { Pool } from "pg";
import { AUTH_PASSWORD_BCRYPT_COST } from "../src/lib/auth/password-policy";
import {
  assertSafeSchemaName,
  closePool,
} from "./seed-preview-accounts.helpers";
import {
  reportPreviewSeedFailure,
  resolvePreviewConnectionString,
  seedPreviewAccounts,
  validatePreviewPassword,
} from "./seed-preview-accounts.orchestration";

async function main() {
  const connectionString = resolvePreviewConnectionString(
    process.env.DIRECT_URL,
    process.env.DATABASE_URL,
  );
  const password = validatePreviewPassword(
    process.env.PREVIEW_ACCOUNTS_PASSWORD,
  );
  const schema = assertSafeSchemaName(process.env.AUTH_DB_SCHEMA || "platform");
  const identityTable = `"${schema}"."UserAuthIdentity"`;
  const accountTable = `"${schema}"."UserAuthAccount"`;

  // Prisma-style URLs carry ?schema=/&pgbouncer= params that node-postgres
  // does not understand but tolerates; strip nothing, qualify tables instead.
  // Bounded timeouts keep the exit-code contract honest: a database that
  // accepts TCP but never answers must become exit 1, not a hung CI step.
  const pool = new Pool({
    connectionString,
    max: 1,
    connectionTimeoutMillis: 15_000,
    query_timeout: 30_000,
  });

  try {
    process.exitCode = await seedPreviewAccounts(
      pool,
      {
        schema,
        identityTable,
        accountTable,
        password,
      },
      {
        hashPassword: (value) => hash(value, AUTH_PASSWORD_BCRYPT_COST),
      },
    );
  } finally {
    await closePool(pool);
  }
}

main().catch((error: unknown) => {
  process.exitCode = reportPreviewSeedFailure(error);
});
