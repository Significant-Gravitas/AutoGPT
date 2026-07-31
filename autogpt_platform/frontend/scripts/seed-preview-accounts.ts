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
 *     pnpm exec tsx scripts/seed-preview-accounts.ts
 *
 * Behavior:
 *   - Idempotent: existing identities are kept (matched by email), and an
 *     existing credential account is never overwritten — rotating
 *     PREVIEW_ACCOUNTS_PASSWORD only affects freshly seeded databases.
 *   - Exit 0: seeded (or already seeded). Exit 3: the Better Auth tables do
 *     not exist — a pre-migration database; the caller decides what to do
 *     (the preview pipeline falls back to its legacy GoTrue seeding arm).
 *     Exit 1: real failure; the pipeline must fail rather than ship a
 *     preview nobody can log into.
 */
import { hash } from "bcryptjs";
import { Pool } from "pg";
import {
  PREVIEW_ACCOUNTS,
  assertSafeSchemaName,
  deterministicUserId,
} from "./seed-preview-accounts.helpers";

const NO_BETTER_AUTH_EXIT_CODE = 3;

async function main() {
  const connectionString = process.env.DIRECT_URL || process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error("DIRECT_URL or DATABASE_URL must be set");
  }
  const password = process.env.PREVIEW_ACCOUNTS_PASSWORD;
  if (!password) {
    throw new Error("PREVIEW_ACCOUNTS_PASSWORD must be set");
  }
  const schema = assertSafeSchemaName(process.env.AUTH_DB_SCHEMA || "platform");
  const identityTable = `"${schema}"."UserAuthIdentity"`;
  const accountTable = `"${schema}"."UserAuthAccount"`;

  // Prisma-style URLs carry ?schema=/&pgbouncer= params that node-postgres
  // does not understand but tolerates; strip nothing, qualify tables instead.
  const pool = new Pool({ connectionString, max: 1 });

  try {
    const { rows } = await pool.query<{ reg: string | null }>(
      "SELECT to_regclass($1)::text AS reg",
      [`${schema}."UserAuthIdentity"`],
    );
    if (!rows[0]?.reg) {
      console.error(
        `Better Auth tables not present in schema "${schema}" — ` +
          "pre-migration database, nothing to seed here.",
      );
      process.exitCode = NO_BETTER_AUTH_EXIT_CODE;
      return;
    }

    // One hash for all five accounts, cost 10 to match auth.ts verification.
    const passwordHash = await hash(password, 10);

    const client = await pool.connect();
    try {
      await client.query("BEGIN");
      let createdIdentities = 0;
      let createdAccounts = 0;

      for (const account of PREVIEW_ACCOUNTS) {
        const existing = await client.query<{ id: string }>(
          `SELECT id FROM ${identityTable} WHERE email = $1`,
          [account.email],
        );
        let userId = existing.rows[0]?.id;
        if (!userId) {
          userId = deterministicUserId(account.email);
          await client.query(
            `INSERT INTO ${identityTable}
               (id, name, email, "emailVerified", role, "createdAt", "updatedAt")
             VALUES ($1, $2, $3, true, $4, now(), now())
             ON CONFLICT (id) DO NOTHING`,
            [userId, account.name, account.email, account.role],
          );
          createdIdentities++;
        }

        const credential = await client.query(
          `SELECT 1 FROM ${accountTable}
           WHERE "userId" = $1 AND "providerId" = 'credential'`,
          [userId],
        );
        if (credential.rowCount === 0) {
          await client.query(
            `INSERT INTO ${accountTable}
               (id, "accountId", "providerId", "userId", password, "createdAt", "updatedAt")
             VALUES (gen_random_uuid()::text, $1, 'credential', $1, $2, now(), now())`,
            [userId, passwordHash],
          );
          createdAccounts++;
        }
      }

      await client.query("COMMIT");
      console.log(
        `Seeded preview accounts: ${createdIdentities} identities and ` +
          `${createdAccounts} credential accounts created ` +
          `(${PREVIEW_ACCOUNTS.length - createdIdentities} identities already present).`,
      );
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error("Preview account seeding failed:", error);
  process.exitCode = 1;
});
