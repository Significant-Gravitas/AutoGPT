import * as Sentry from "@sentry/nextjs";
import { APIError } from "better-auth/api";
import type { Pool } from "pg";

// Admins write these rows from /admin/users (backend/data/access_blocks.py),
// storing an IP exactly as Better Auth's getIp() returns it.

export const SIGN_UP_BLOCKED_CODE = "SIGN_UP_BLOCKED";
export const SIGN_IN_BLOCKED_CODE = "SIGN_IN_BLOCKED";

// One generic sentence, so the message doesn't reveal which rule refused.
const SIGN_UP_BLOCKED_MESSAGE =
  "Sign-up isn't available from this network or email address. If you think this is a mistake, please contact support.";
const SIGN_IN_BLOCKED_MESSAGE =
  "Sign-in isn't available from this network. If you think this is a mistake, please contact support.";

type Queryable = Pick<Pool, "query">;

interface SignUpArgs {
  email: string;
  ipAddress: string | null;
}

export async function assertSignUpAllowed(
  pool: Queryable,
  { email, ipAddress }: SignUpArgs,
) {
  const blocked = await isBlocked(pool, {
    ipAddress,
    emailDomains: getDomainAndParents(email),
  });
  if (blocked) {
    throw new APIError("FORBIDDEN", {
      code: SIGN_UP_BLOCKED_CODE,
      message: SIGN_UP_BLOCKED_MESSAGE,
    });
  }
}

export async function assertSignInAllowed(
  pool: Queryable,
  ipAddress: string | null,
) {
  if (!ipAddress) return;
  if (await isBlocked(pool, { ipAddress, emailDomains: [] })) {
    throw new APIError("FORBIDDEN", {
      code: SIGN_IN_BLOCKED_CODE,
      message: SIGN_IN_BLOCKED_MESSAGE,
    });
  }
}

// A domain block covers its subdomains: blocking example.com refuses
// a@mail.example.com too.
export function getDomainAndParents(email: string) {
  const labels = (email.split("@").pop() ?? "").toLowerCase().split(".");
  const domains: string[] = [];
  for (let i = 0; i < labels.length - 1; i++) {
    domains.push(labels.slice(i).join("."));
  }
  return domains;
}

interface MatchArgs {
  ipAddress: string | null;
  emailDomains: string[];
}

// Fails open: if the table can't be read (a deploy running ahead of its
// migration, say) sign-ups and sign-ins carry on, and Sentry hears of it.
async function isBlocked(
  pool: Queryable,
  { ipAddress, emailDomains }: MatchArgs,
) {
  if (!ipAddress && emailDomains.length === 0) return false;
  try {
    // The rule lives in the database (migration
    // 20261021120000_add_auth_ip_events_and_access_blocks), tested by the backend.
    const result = await pool.query(
      "SELECT is_auth_access_blocked($1, $2::text[]) AS blocked",
      [ipAddress ?? "", emailDomains],
    );
    return result.rows[0]?.blocked === true;
  } catch (error) {
    const code =
      typeof error === "object" && error !== null && "code" in error
        ? String((error as { code: unknown }).code)
        : "unknown";
    const sanitized = new Error(`Failed to read access blocks (pg ${code})`);
    console.error(sanitized.message);
    Sentry.captureException(sanitized, { tags: { pg_code: code } });
    return false;
  }
}
