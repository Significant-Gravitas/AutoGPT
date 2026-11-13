import * as Sentry from "@sentry/nextjs";
import type { Pool } from "pg";

export type AuthIdentity = {
  id: string;
  email: string;
  name?: string | null;
};

export type ProvisionOutcome = "created" | "exists" | "failed";

// From the `user.create.after` hook, so a session never exists before its `User` row. `updatedAt` has
// no DB default. Never throws: a throw after commit strands an identity that rejects every retry.
export async function provisionPlatformUser(
  pool: Pick<Pool, "query">,
  user: AuthIdentity,
): Promise<ProvisionOutcome> {
  try {
    const result = await pool.query(
      'INSERT INTO "User" (id, email, name, "updatedAt") VALUES ($1, $2, $3, NOW()) ON CONFLICT (id) DO NOTHING',
      [user.id, user.email, user.name ?? null],
    );
    return (result.rowCount ?? 0) > 0 ? "created" : "exists";
  } catch (error) {
    // The likely cause is `User_email_key`: a platform row already carries
    // this email under a different id (an auth identity removed through the
    // admin plugin and re-created, say). `ON CONFLICT (id)` cannot absorb
    // that, and neither can the backend's own provisioning, so it has to be
    // visible rather than retried quietly.
    //
    // Report only the SQLSTATE and constraint, never the raw `pg` error: its
    // `detail` field spells out the conflicting value ("Key (email)=(…)
    // already exists"), which would put the user's email into server logs
    // and, via Sentry's extra-error-data capture, into the event body.
    const { code, constraint } = describePgError(error);
    const sanitized = new Error(
      `Failed to provision platform User (pg ${code}, ${constraint})`,
    );
    console.error(sanitized.message, { userId: user.id });
    Sentry.captureException(sanitized, {
      tags: { auth_hook: "user.create.after", pg_code: code },
      extra: { userId: user.id, constraint },
    });
    return "failed";
  }
}

function describePgError(error: unknown): {
  code: string;
  constraint: string;
} {
  const fields =
    typeof error === "object" && error !== null
      ? (error as { code?: unknown; constraint?: unknown })
      : {};
  return {
    code: typeof fields.code === "string" ? fields.code : "unknown",
    constraint:
      typeof fields.constraint === "string" ? fields.constraint : "unknown",
  };
}

// Whether the platform `User` row exists. Throws if it can't be read: the
// callers want opposite answers then (see hasAccountBeenUsed).
export async function platformUserExists(
  pool: Pick<Pool, "query">,
  userId: string,
) {
  try {
    const result = await pool.query('SELECT 1 FROM "User" WHERE id = $1', [
      userId,
    ]);
    return (result.rowCount ?? 0) > 0;
  } catch (error) {
    const { code } = describePgError(error);
    console.error("Failed to look up the platform User", { userId, code });
    throw new Error(`Failed to look up the platform User (pg ${code})`);
  }
}
