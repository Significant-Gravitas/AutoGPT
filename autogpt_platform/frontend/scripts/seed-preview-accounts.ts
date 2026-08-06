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
 * scripts in this folder. Prefer DIRECT_URL: the explicit transaction wants
 * a direct/session connection; a DATABASE_URL fallback must not be a
 * transaction-pooled endpoint.)
 *
 * Behavior:
 *   - Idempotent: existing identities are kept (matched by email; role and
 *     emailVerified are converged to the roster), and an existing credential
 *     account is never overwritten — rotating PREVIEW_ACCOUNTS_PASSWORD only
 *     affects freshly seeded databases.
 *   - Exit codes are the cross-repo contract with the preview CD pipeline —
 *     see the named constants below.
 */
import { hash } from "bcryptjs";
import { Pool } from "pg";
import {
  assertSafeSchemaName,
  closePool,
  seedRoster,
} from "./seed-preview-accounts.helpers";

// The exit-code contract with the preview CD pipeline. Keep all three in
// lockstep with the seed step in AutoGPT_cloud_infrastructure's workflow.
const SEEDED_EXIT_CODE = 0; // seeded (or already seeded)
const FAILURE_EXIT_CODE = 1; // real failure — the pipeline must fail loudly
const NO_BETTER_AUTH_EXIT_CODE = 3; // pre-migration DB — caller falls back

const MIN_PASSWORD_LENGTH = 12; // the platform's own password floor

async function main() {
  const connectionString = process.env.DIRECT_URL || process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error("DIRECT_URL or DATABASE_URL must be set");
  }
  const password = process.env.PREVIEW_ACCOUNTS_PASSWORD;
  if (!password) {
    throw new Error("PREVIEW_ACCOUNTS_PASSWORD must be set");
  }
  if (password.length < MIN_PASSWORD_LENGTH) {
    // The same secret backs preview-admin (role=admin) on every preview.
    throw new Error(
      `PREVIEW_ACCOUNTS_PASSWORD must be at least ${MIN_PASSWORD_LENGTH} characters`,
    );
  }
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
    // Probe both tables the seeder writes, with the same quoting as the
    // qualified names used in the SQL. All Better Auth tables ship in one
    // migration, so both-absent means a pre-migration database (exit 3),
    // while a mixed result means a half-applied migration — that is a broken
    // database, not a legacy one, and must fail loudly rather than fall back
    // to GoTrue seeding.
    const tableExists = async (qualified: string) => {
      const { rows } = await pool.query<{ reg: string | null }>(
        "SELECT to_regclass($1)::text AS reg",
        [qualified],
      );
      return Boolean(rows[0]?.reg);
    };
    const hasIdentity = await tableExists(identityTable);
    const hasAccount = await tableExists(accountTable);
    if (!hasIdentity && !hasAccount) {
      console.error(
        `Better Auth tables not present in schema "${schema}" — ` +
          "pre-migration database, nothing to seed here.",
      );
      process.exitCode = NO_BETTER_AUTH_EXIT_CODE;
      return;
    }
    if (!hasIdentity || !hasAccount) {
      throw new Error(
        `Schema "${schema}" has only one of UserAuthIdentity/UserAuthAccount ` +
          "— half-applied migration; refusing to seed or fall back.",
      );
    }

    // One hash for all five accounts, cost 10 to match auth.ts verification.
    const passwordHash = await hash(password, 10);

    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      const { createdIdentities, createdAccounts } = await seedRoster(client, {
        identityTable,
        accountTable,
        passwordHash,
      });
      await client.query("COMMIT");
      console.log(
        `Seeded preview accounts: ${createdIdentities} identities and ` +
          `${createdAccounts} credential accounts created.`,
      );
      process.exitCode = SEEDED_EXIT_CODE;
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  } finally {
    await closePool(pool);
  }
}

main().catch((error: unknown) => {
  // Scoped message only — the full pg error object echoes SQL text and DB
  // detail into shared CI logs.
  const code =
    error instanceof Error && "code" in error
      ? ` (code ${(error as { code?: string }).code})`
      : "";
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Preview account seeding failed: ${message}${code}`);
  process.exitCode = FAILURE_EXIT_CODE;
});
