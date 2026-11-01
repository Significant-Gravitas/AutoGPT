import type { Pool } from "pg";

// Read by the admin users page (backend/data/auth_ip_events.py) and deleted
// after `auth_ip_event_retention_days`. Sessions can't stand in for it: a ban
// deletes them.

export type AuthIPEventKind = "SIGN_UP" | "SIGN_IN";

interface Args {
  userId: string;
  kind: AuthIPEventKind;
  ipAddress: string | null | undefined;
  userAgent: string | null | undefined;
}

// Never throws: a sign-in must not fail because its IP wasn't written down.
export async function recordAuthIP(
  pool: Pick<Pool, "query">,
  { userId, kind, ipAddress, userAgent }: Args,
) {
  if (!ipAddress) return;
  try {
    await pool.query("SELECT record_auth_ip_event($1, $2, $3, $4)", [
      userId,
      kind,
      ipAddress,
      userAgent ?? "",
    ]);
  } catch (error) {
    console.error("Failed to record a sign-in IP", {
      userId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
