/**
 * Environment-driven signup gating.
 *
 * Replaces the old DB-level allowlist (a Postgres trigger on `auth.users`,
 * cloud-infra-only). Going forward, signup gating is pure env config so each
 * environment controls it independently with no schema/data:
 *
 *   AUTH_ALLOW_NEW_ACCOUNTS  "false" blocks ALL new signups. Default: open.
 *   AUTH_SIGNUP_ALLOWLIST    comma-separated allowed emails and `@domains`.
 *                            Empty/unset = no allowlist (open). When set,
 *                            only matching emails may create an account.
 *
 * Prod leaves both unset → open signup. Dev/preview sets AUTH_SIGNUP_ALLOWLIST
 * so randoms can't sign up on a public preview URL.
 *
 * Enforced in auth.ts via `databaseHooks.user.create.before`, which fires for
 * BOTH email/password signup and a first OAuth sign-in (both create a user).
 * Existing users are unaffected — the hook only runs on user *creation*, and
 * the one-time data migration inserts via raw SQL, bypassing it.
 *
 * Migrating an existing `allowed_users` table to the env var:
 *   psql -tAc "select string_agg(email, ',') from allowed_users" \
 *     → paste into AUTH_SIGNUP_ALLOWLIST.
 */

export interface SignupGateConfig {
  allowNewAccounts: boolean;
  allowlist: string[];
}

const DISABLED_MESSAGE =
  "New account registration is not allowed at this time.";
const NOT_ALLOWLISTED_MESSAGE =
  "This email address is not allowed to register. Please contact support for assistance.";

export function parseAllowlist(raw: string | undefined | null): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);
}

export function readSignupGateConfig(
  env: Record<string, string | undefined> = process.env,
): SignupGateConfig {
  return {
    // Only an explicit "false" disables; anything else (incl. unset) is open.
    allowNewAccounts: env.AUTH_ALLOW_NEW_ACCOUNTS !== "false",
    allowlist: parseAllowlist(env.AUTH_SIGNUP_ALLOWLIST),
  };
}

export interface SignupDecision {
  allowed: boolean;
  /** Human message; phrased so the frontend `isWaitlistError()` catches it. */
  reason?: string;
}

export function isSignupAllowed(
  email: string,
  config: SignupGateConfig,
): SignupDecision {
  if (!config.allowNewAccounts) {
    return { allowed: false, reason: DISABLED_MESSAGE };
  }

  // No allowlist configured → open (the prod default).
  if (config.allowlist.length === 0) {
    return { allowed: true };
  }

  const normalized = email.trim().toLowerCase();
  const atIndex = normalized.lastIndexOf("@");
  const domain = atIndex >= 0 ? normalized.slice(atIndex) : "";

  const matches = config.allowlist.some((entry) =>
    entry.startsWith("@") ? entry === domain : entry === normalized,
  );

  return matches
    ? { allowed: true }
    : { allowed: false, reason: NOT_ALLOWLISTED_MESSAGE };
}
