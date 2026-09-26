import * as Sentry from "@sentry/nextjs";
import type { Pool } from "pg";

export type AuthIdentity = {
  id: string;
  email: string;
  name?: string | null;
};

export type ProvisionOutcome = "created" | "exists" | "failed";

// Create the platform `User` row for a freshly created auth identity, from
// the Better Auth `user.create.after` hook, so that a session can never exist
// before the row it hangs off. Every first-page-load write with a foreign key
// to `User` (onboarding state, push subscriptions, experiment assignments)
// depends on that ordering.
//
// Every other column has a database default; `updatedAt` is Prisma-managed and
// has none, so it is set explicitly, exactly like the email mirror does.
//
// Best-effort by design. Better Auth queues `create.after` hooks and awaits
// them after the transaction commits, so a throw here would surface as a
// failed sign-up *after* the identity is durable, leaving an identity with no
// session that rejects every retry with "user already exists". The sign-in
// flows still call `POST /auth/user` and roll the session back if it fails,
// and every authenticated backend request self-heals a missing row, so a
// failure here degrades to those paths rather than creating a new one.
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
