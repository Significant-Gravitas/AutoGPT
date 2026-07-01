/**
 * One-time migration of Supabase GoTrue users into the Better Auth tables.
 *
 * Copies:
 *   - auth.users        -> platform."user"
 *   - email/password    -> platform.account (providerId = 'credential';
 *                          bcrypt hashes carry over because Better Auth is
 *                          configured to verify with bcrypt)
 *   - auth.identities   -> platform.account (google / github / discord)
 *
 * When to run:
 *   Once, after the Better Auth schema migration has been deployed
 *   (backend Prisma migration 20260610120000_add_better_auth_tables),
 *   before — or while — old Supabase sessions are being bridged. Keep
 *   SUPABASE_JWT_SECRET set in the frontend environment for the duration of
 *   the bridge window so pre-migration sessions keep working.
 *
 * Usage:
 *   DATABASE_URL=postgresql://... npx tsx scripts/migrate-supabase-auth.ts
 *
 * Idempotent: every insert is guarded (ON CONFLICT DO NOTHING / NOT EXISTS),
 * so the script is safe to re-run. Each batch of users runs in its own
 * transaction; the script exits non-zero on the first failed batch.
 */
import { Pool } from "pg";

const BATCH_SIZE = 1000;

async function tableExists(pool: Pool, qualifiedName: string) {
  const { rows } = await pool.query<{ reg: string | null }>(
    "SELECT to_regclass($1)::text AS reg",
    [qualifiedName],
  );
  return rows[0]?.reg != null;
}

async function columnExists(
  pool: Pool,
  schema: string,
  table: string,
  column: string,
) {
  const { rows } = await pool.query(
    `SELECT 1 FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2 AND column_name = $3`,
    [schema, table, column],
  );
  return rows.length > 0;
}

async function main() {
  if (!process.env.DATABASE_URL) {
    console.error("DATABASE_URL is not set");
    process.exit(1);
  }

  const pool = new Pool({ connectionString: process.env.DATABASE_URL });

  try {
    if (!(await tableExists(pool, "auth.users"))) {
      console.log("auth.users does not exist — nothing to migrate");
      return;
    }

    const hasIdentities = await tableExists(pool, "auth.identities");
    // GoTrue added identities.provider_id (the provider's user id) in newer
    // versions; older schemas only carry it inside identity_data->>'sub'.
    const providerAccountIdExpr =
      hasIdentities &&
      (await columnExists(pool, "auth", "identities", "provider_id"))
        ? "COALESCE(i.provider_id::text, i.identity_data->>'sub', i.user_id::text)"
        : "COALESCE(i.identity_data->>'sub', i.user_id::text)";

    const { rows: totals } = await pool.query<{ n: string }>(
      "SELECT count(*)::text AS n FROM auth.users",
    );
    console.log(`auth.users rows: ${totals[0].n}`);

    let lastId = "00000000-0000-0000-0000-000000000000";
    let processed = 0;
    let usersInserted = 0;
    let credentialAccounts = 0;
    let oauthAccounts = 0;

    for (;;) {
      const { rows: batch } = await pool.query<{ id: string }>(
        "SELECT id::text FROM auth.users WHERE id > $1::uuid ORDER BY id LIMIT $2",
        [lastId, BATCH_SIZE],
      );
      if (batch.length === 0) break;

      const ids = batch.map((r) => r.id);
      lastId = ids[ids.length - 1];

      const client = await pool.connect();
      try {
        await client.query("BEGIN");

        // 1) auth.users -> platform."user". Skip deleted users and rows
        //    without an email. The NOT EXISTS email guard skips users whose
        //    email is already taken by a different Better Auth user instead
        //    of failing the whole batch on the unique(email) index.
        const userRes = await client.query(
          `INSERT INTO platform."user"
             (id, name, email, "emailVerified", role, banned, "createdAt", "updatedAt")
           SELECT
             u.id::text,
             COALESCE(
               u.raw_user_meta_data->>'name',
               u.raw_user_meta_data->>'full_name',
               split_part(u.email, '@', 1)
             ),
             u.email,
             (u.email_confirmed_at IS NOT NULL),
             CASE
               WHEN COALESCE(u.is_super_admin, false) OR u.role = 'admin'
               THEN 'admin' ELSE 'user'
             END,
             (u.banned_until IS NOT NULL AND u.banned_until > now()),
             COALESCE(u.created_at, now()),
             COALESCE(u.updated_at, now())
           FROM auth.users u
           WHERE u.id = ANY($1::uuid[])
             AND u.email IS NOT NULL
             AND u.deleted_at IS NULL
             AND NOT EXISTS (
               SELECT 1 FROM platform."user" pu
               WHERE pu.email = u.email AND pu.id <> u.id::text
             )
           ON CONFLICT (id) DO NOTHING`,
          [ids],
        );

        // 2) Email/password credentials -> platform.account.
        const credRes = await client.query(
          `INSERT INTO platform.account
             (id, "accountId", "providerId", "userId", password, "createdAt", "updatedAt")
           SELECT
             gen_random_uuid()::text,
             u.id::text,
             'credential',
             u.id::text,
             u.encrypted_password,
             COALESCE(u.created_at, now()),
             COALESCE(u.updated_at, now())
           FROM auth.users u
           WHERE u.id = ANY($1::uuid[])
             AND u.encrypted_password IS NOT NULL
             AND length(u.encrypted_password) > 0
             AND EXISTS (
               SELECT 1 FROM platform."user" pu WHERE pu.id = u.id::text
             )
             AND NOT EXISTS (
               SELECT 1 FROM platform.account a
               WHERE a."userId" = u.id::text AND a."providerId" = 'credential'
             )`,
          [ids],
        );

        // 3) OAuth identities -> platform.account. provider 'email' is the
        //    GoTrue-internal credential identity and is skipped (handled
        //    above); only providers Better Auth is configured for migrate.
        let oauthRes = { rowCount: 0 as number | null };
        if (hasIdentities) {
          oauthRes = await client.query(
            `INSERT INTO platform.account
               (id, "accountId", "providerId", "userId", "createdAt", "updatedAt")
             SELECT
               gen_random_uuid()::text,
               ${providerAccountIdExpr},
               i.provider,
               i.user_id::text,
               COALESCE(i.created_at, now()),
               COALESCE(i.updated_at, now())
             FROM auth.identities i
             WHERE i.user_id = ANY($1::uuid[])
               AND i.provider IN ('google', 'github', 'discord')
               AND EXISTS (
                 SELECT 1 FROM platform."user" pu WHERE pu.id = i.user_id::text
               )
               AND NOT EXISTS (
                 SELECT 1 FROM platform.account a
                 WHERE a."userId" = i.user_id::text AND a."providerId" = i.provider
               )`,
            [ids],
          );
        }

        await client.query("COMMIT");

        processed += ids.length;
        usersInserted += userRes.rowCount ?? 0;
        credentialAccounts += credRes.rowCount ?? 0;
        oauthAccounts += oauthRes.rowCount ?? 0;
        console.log(
          `processed ${processed} users ` +
            `(+${userRes.rowCount ?? 0} users, ` +
            `+${credRes.rowCount ?? 0} credential accounts, ` +
            `+${oauthRes.rowCount ?? 0} oauth accounts)`,
        );
      } catch (error) {
        await client.query("ROLLBACK").catch(() => {});
        throw error;
      } finally {
        client.release();
      }
    }

    console.log(
      `done: ${processed} auth.users processed, ` +
        `${usersInserted} users migrated, ` +
        `${credentialAccounts} credential accounts, ` +
        `${oauthAccounts} oauth accounts`,
    );
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error("migration failed:", error);
  process.exit(1);
});
