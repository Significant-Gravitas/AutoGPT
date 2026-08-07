import { createHash } from "node:crypto";

/**
 * The login accounts every preview environment is seeded with. The
 * "@previews.agpt.co" subdomain dodges the frontend's "@agpt.co" SSO-only
 * login block, and `role` is what the backend's admin gate reads (Better
 * Auth mints it into the JWT role claim).
 */
export const PREVIEW_ACCOUNTS = [
  {
    email: "preview-admin@previews.agpt.co",
    name: "preview-admin",
    role: "admin",
    subscriptionTier: "NO_TIER",
    onboardingComplete: true,
  },
  {
    email: "preview-existing@previews.agpt.co",
    name: "preview-existing",
    role: "user",
    subscriptionTier: "NO_TIER",
    onboardingComplete: true,
  },
  {
    email: "preview-clean@previews.agpt.co",
    name: "preview-clean",
    role: "user",
    subscriptionTier: "NO_TIER",
    onboardingComplete: false,
  },
  {
    email: "preview-pro@previews.agpt.co",
    name: "preview-pro",
    role: "user",
    subscriptionTier: "PRO",
    onboardingComplete: true,
  },
  {
    email: "preview-enterprise@previews.agpt.co",
    name: "preview-enterprise",
    role: "user",
    subscriptionTier: "ENTERPRISE",
    onboardingComplete: true,
  },
] as const;

/**
 * Deterministic user id for FRESH inserts: uuid-shaped truncation of
 * SHA-256(email). Legacy branch databases may carry identities whose IDs were
 * derived with Postgres md5(email)::uuid. Re-seeding remains safe because
 * identities are matched by email before any ID is derived, so the derivation
 * only has to be stable, not backward-identical.
 */
export function deterministicUserID(email: string): string {
  // SHA-256 provides a stable mapping from public roster emails to IDs; this
  // is not a security digest, so secrecy and password-hashing properties are
  // irrelevant here.
  const hex = createHash("sha256").update(email).digest("hex").slice(0, 32);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}

/**
 * Schema names come from the AUTH_DB_SCHEMA env var and are interpolated as
 * SQL identifiers, so refuse anything that isn't a plain lowercase
 * identifier rather than attempting to escape it.
 */
export function assertSafeSchemaName(schema: string): string {
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) {
    throw new Error(`Unsafe schema name: ${JSON.stringify(schema)}`);
  }
  return schema;
}

interface PoolCloser {
  end(): Promise<void>;
}

/** Closes the pool without allowing cleanup to replace the seed outcome. */
export async function closePool(
  pool: PoolCloser,
  reportError: (message: string) => void = console.error,
) {
  try {
    await pool.end();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reportError(`Preview account seeder cleanup failed: ${message}`);
  }
}

interface QueryResultLike {
  rows: { id?: string }[];
  rowCount: number | null;
}

/** The subset of a pg client the seeder needs — injectable for tests. */
export interface QueryExecutor {
  query(text: string, params?: unknown[]): Promise<QueryResultLike>;
}

/**
 * Seeds the roster into the given (already-open, ideally transactional)
 * client. Extracted from the CLI entry so the orchestration — idempotency,
 * email-match attachment, id-collision refusal, role convergence — is unit
 * testable without a database.
 */
export async function seedRoster(
  client: QueryExecutor,
  opts: {
    identityTable: string;
    accountTable: string;
    userTable: string;
    profileTable: string;
    onboardingTable: string;
    subscriptionTierType: string;
    onboardingStepType: string;
    passwordHash: string;
  },
) {
  const {
    identityTable,
    accountTable,
    userTable,
    profileTable,
    onboardingTable,
    subscriptionTierType,
    onboardingStepType,
    passwordHash,
  } = opts;
  let createdIdentities = 0;
  let createdAccounts = 0;
  let createdUsers = 0;
  let updatedUsers = 0;
  let createdProfiles = 0;
  let changedOnboarding = 0;

  for (const account of PREVIEW_ACCOUNTS) {
    // Bare ON CONFLICT DO NOTHING arbitrates on ANY constraint, so both
    // "id already taken" and "email already registered" (including a
    // concurrent seeder racing this one) no-op instead of aborting the
    // transaction.
    const insertedIdentity = await client.query(
      `INSERT INTO ${identityTable}
         (id, name, email, "emailVerified", role, "createdAt", "updatedAt")
       VALUES ($1, $2, $3, true, $4, now(), now())
       ON CONFLICT DO NOTHING`,
      [
        deterministicUserID(account.email),
        account.name,
        account.email,
        account.role,
      ],
    );
    createdIdentities += insertedIdentity.rowCount ?? 0;

    // Resolve the id by email AFTER the insert: this is the identity the
    // credential must attach to whether the row pre-existed, was just
    // created, or was created by a concurrent run. If it's still absent,
    // the deterministic id belongs to some other user and the insert
    // no-opped — attaching a credential to that id would hand the preview
    // password to an unrelated account, so fail loudly instead.
    const identity = await client.query(
      `SELECT id FROM ${identityTable} WHERE email = $1`,
      [account.email],
    );
    const userID = identity.rows[0]?.id;
    if (!userID) {
      throw new Error(
        `Identity for ${account.email} neither existed nor could be ` +
          "created (its deterministic id is taken by a different user)",
      );
    }

    // The roster is the source of truth: deliberately converge pre-existing
    // rows even if a long-lived preview has drifted. An identity that
    // predates this seeder (older seed generations, or a DB cloned from a
    // template) may carry the wrong role or an unverified email, and
    // preview-admin's role='admin' is the property the roster exists to
    // guarantee. Passwords are deliberately NOT converged (see the entry
    // script's docstring).
    if (insertedIdentity.rowCount !== 1) {
      await client.query(
        `UPDATE ${identityTable}
         SET role = $2, "emailVerified" = true, "updatedAt" = now()
         WHERE id = $1
           AND (role IS DISTINCT FROM $2 OR "emailVerified" IS DISTINCT FROM true)`,
        [userID, account.role],
      );
    }

    // The guarded insert keeps sequential retries idempotent. Preview CD runs
    // one seeder per branch DB; cross-process safety would require a partial
    // credential-only unique index because OAuth allows multiple accounts for
    // the same user/provider. Existing credentials are never overwritten.
    const insertedCredential = await client.query(
      `INSERT INTO ${accountTable}
         (id, "accountId", "providerId", "userId", password, "createdAt", "updatedAt")
       SELECT gen_random_uuid()::text, $1, 'credential', $1, $2, now(), now()
       WHERE NOT EXISTS (
         SELECT 1 FROM ${accountTable}
         WHERE "userId" = $1 AND "providerId" = 'credential'
       )`,
      [userID, passwordHash],
    );
    createdAccounts += insertedCredential.rowCount ?? 0;

    const insertedUser = await client.query(
      `INSERT INTO ${userTable}
         (id, email, "emailVerified", name, "subscriptionTier", "createdAt", "updatedAt")
       VALUES ($1, $2, true, $3, $4::${subscriptionTierType}, now(), now())
       ON CONFLICT DO NOTHING`,
      [userID, account.email, account.name, account.subscriptionTier],
    );
    createdUsers += insertedUser.rowCount ?? 0;

    const productUser = await client.query(
      `SELECT id
       FROM ${userTable}
       WHERE email = $1`,
      [account.email],
    );
    const productUserID = productUser.rows[0]?.id;
    if (!productUserID) {
      throw new Error(
        `Platform user for ${account.email} neither existed nor could be ` +
          "created (its auth id is taken by a different user)",
      );
    }
    if (productUserID !== userID) {
      throw new Error(
        `Platform user for ${account.email} has id ${productUserID}, but ` +
          `the auth identity has id ${userID}`,
      );
    }

    if (insertedUser.rowCount !== 1) {
      const updatedUser = await client.query(
        `UPDATE ${userTable}
         SET name = $2,
             "emailVerified" = true,
             "subscriptionTier" = CASE
               WHEN "stripeCustomerId" IS NULL THEN $3::${subscriptionTierType}
               ELSE "subscriptionTier"
             END,
             "updatedAt" = now()
         WHERE id = $1
           AND (
             name IS DISTINCT FROM $2
             OR "emailVerified" IS DISTINCT FROM true
             OR (
               "stripeCustomerId" IS NULL
               AND "subscriptionTier" IS DISTINCT FROM $3::${subscriptionTierType}
             )
           )`,
        [userID, account.name, account.subscriptionTier],
      );
      updatedUsers += updatedUser.rowCount ?? 0;
    }

    const insertedProfile = await client.query(
      `INSERT INTO ${profileTable}
         (id, "userId", name, username, description, links, "avatarUrl", "createdAt", "updatedAt")
       VALUES (gen_random_uuid()::text, $1, $2, $3, 'I''m new here', ARRAY[]::text[], '', now(), now())
       ON CONFLICT DO NOTHING`,
      [userID, account.name, account.name],
    );
    createdProfiles += insertedProfile.rowCount ?? 0;

    const profile = await client.query(
      `SELECT id FROM ${profileTable} WHERE "userId" = $1`,
      [userID],
    );
    if (!profile.rows[0]?.id) {
      throw new Error(
        `Profile for ${account.email} neither existed nor could be created ` +
          `(the username ${account.name} is taken by a different user)`,
      );
    }

    const onboarding = account.onboardingComplete
      ? await client.query(
          `INSERT INTO ${onboardingTable} AS current
             (id, "userId", "completedSteps", "createdAt", "updatedAt")
           VALUES (
             gen_random_uuid()::text,
             $1,
             ARRAY['VISIT_COPILOT'::${onboardingStepType}],
             now(),
             now()
           )
           ON CONFLICT ("userId") DO UPDATE
           SET "completedSteps" = array_append(
                 current."completedSteps",
                 'VISIT_COPILOT'::${onboardingStepType}
               ),
               "updatedAt" = now()
           WHERE NOT (
             'VISIT_COPILOT'::${onboardingStepType}
             = ANY(current."completedSteps")
           )`,
          [userID],
        )
      : await client.query(
          `INSERT INTO ${onboardingTable} AS current
             (id, "userId", "completedSteps", "createdAt", "updatedAt")
           VALUES (
             gen_random_uuid()::text,
             $1,
             ARRAY[]::${onboardingStepType}[],
             now(),
             now()
           )
           ON CONFLICT ("userId") DO UPDATE
           SET "completedSteps" = array_remove(
                 current."completedSteps",
                 'VISIT_COPILOT'::${onboardingStepType}
               ),
               "updatedAt" = now()
           WHERE (
             'VISIT_COPILOT'::${onboardingStepType}
             = ANY(current."completedSteps")
           )`,
          [userID],
        );
    changedOnboarding += onboarding.rowCount ?? 0;
  }

  return {
    createdIdentities,
    createdAccounts,
    createdUsers,
    updatedUsers,
    createdProfiles,
    changedOnboarding,
  };
}
