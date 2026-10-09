import { sanitizeAuthNext } from "@/lib/auth-redirect";

/**
 * Email verification (AUTH_REQUIRE_EMAIL_VERIFICATION=true).
 *
 * Better Auth then creates no session at email/password sign-up and answers a
 * sign-in by an unverified account with 403 EMAIL_NOT_VERIFIED; both email a
 * link to `/api/auth/verify-email`, which marks the address verified, signs
 * the user in and redirects to the callbackURL built here. `/auth/callback`
 * then provisions the platform user and fires the sign-up conversion, exactly
 * as it does after Google sign-in, so neither happens for an address nobody
 * has proven they own.
 */

export const EMAIL_NOT_VERIFIED_CODE = "EMAIL_NOT_VERIFIED";

// Shown on /login when the callback has no session to work with: an expired or
// already-used link, or a second click once the address is verified.
export const EMAIL_VERIFICATION_NOTICE_PARAM = "email_verification";
export type EmailVerificationNotice = "expired" | "verified";

// A sign-up's marketing refusal rides in the link itself, so it survives the
// link being opened on another device or long after sign-up. The callback
// honours it only for the account that link creates.
const MARKETING_OPT_OUT_PARAM = "marketing_opt_out";

interface CallbackURLArgs {
  next?: string | null;
  marketingOptOut?: boolean;
}

export function getEmailVerificationCallbackURL({
  next,
  marketingOptOut = false,
}: CallbackURLArgs = {}) {
  const params = new URLSearchParams({ method: "email" });
  const safeNext = sanitizeAuthNext(next);
  if (safeNext) params.set("next", safeNext);
  if (marketingOptOut) params.set(MARKETING_OPT_OUT_PARAM, "1");
  return `/auth/callback?${params.toString()}`;
}

export function hasMarketingOptOutParam(searchParams: URLSearchParams) {
  return searchParams.get(MARKETING_OPT_OUT_PARAM) === "1";
}

// The user.create.after hook skips these. Provisioning the platform User (and
// the personal org behind it) waits for the verification link, so an address
// nobody has proven they own never gets an account.
export function isAwaitingEmailVerification(
  user: { emailVerified?: boolean | null },
  requireEmailVerification: boolean,
) {
  return requireEmailVerification && !user.emailVerified;
}

export function getEmailVerificationNotice(
  value: string | null,
): EmailVerificationNotice | null {
  return value === "expired" || value === "verified" ? value : null;
}
