import type { EmailVerificationNotice } from "@/lib/auth/email-verification";

export const ACCOUNT_BANNED_NOTICE_COPY =
  "This account has been banned. If you think this is a mistake, please contact support.";

export const EMAIL_VERIFICATION_NOTICE_COPY: Record<
  EmailVerificationNotice,
  string
> = {
  expired:
    "That verification link has expired or was already used. Log in and we'll email you a new one.",
  verified: "Your email is verified. Log in to continue.",
};

// Better Auth's admin plugin refuses a banned account with this code. A
// deleted account is not banned: its owner signs in to keep it.
const BANNED_USER_ERROR_CODE = "BANNED_USER";

// A social sign-in refused at the provider callback comes back with its error
// code. The app's own copy, never the error_description from the query.
export function getSignInErrorNotice(error: string | null) {
  return error === BANNED_USER_ERROR_CODE ? ACCOUNT_BANNED_NOTICE_COPY : null;
}
