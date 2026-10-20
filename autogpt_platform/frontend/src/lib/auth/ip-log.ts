import type { Pool } from "pg";

// Read by the admin users page (backend/data/auth_ip_events.py) and deleted
// after `auth_ip_event_retention_days`. Sessions can't stand in for it: a ban
// deletes them.

export type AuthIpEventKind = "SIGN_UP" | "SIGN_IN";

interface Args {
  userId: string;
  kind: AuthIpEventKind;
  ipAddress: string | null | undefined;
  userAgent: string | null | undefined;
}

// Never throws: a sign-in must not fail because its IP wasn't written down.
export async function recordAuthIp(
  pool: Pick<Pool, "query">,
  { userId, kind, ipAddress, userAgent }: Args,
) {
  if (!ipAddress) return;
  try {
    await pool.query(
      `INSERT INTO "UserAuthIpEvent" ("userId", kind, "ipAddress", "userAgent")
       VALUES ($1, $2::"AuthIpEventKind", $3, $4)`,
      [userId, kind, ipAddress, userAgent || null],
    );
  } catch (error) {
    console.error("Failed to record a sign-in IP", {
      userId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
