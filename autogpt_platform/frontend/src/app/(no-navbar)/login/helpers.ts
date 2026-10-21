import type { AccountRefusal } from "@/lib/auth/account-banned";
import {
  BANNED_ACCOUNT_ERROR_CODE,
  BANNED_ACCOUNT_MESSAGE,
  CLOSED_ACCOUNT_ERROR_CODE,
  CLOSED_ACCOUNT_MESSAGE,
} from "@/lib/auth/closed-account";
import type { EmailVerificationNotice } from "@/lib/auth/email-verification";

// Shown after a refused account was signed out (lib/auth/account-banned.ts).
export const ACCOUNT_NOTICE_COPY: Record<AccountRefusal, string> = {
  banned: BANNED_ACCOUNT_MESSAGE,
  closed: CLOSED_ACCOUNT_MESSAGE,
};

// A social sign-in Better Auth refused, by the error code in the query.
const SIGN_IN_ERROR_COPY = new Map([
  [CLOSED_ACCOUNT_ERROR_CODE, CLOSED_ACCOUNT_MESSAGE],
  [BANNED_ACCOUNT_ERROR_CODE, BANNED_ACCOUNT_MESSAGE],
]);

export const EMAIL_VERIFICATION_NOTICE_COPY: Record<
  EmailVerificationNotice,
  string
> = {
  expired:
    "That verification link has expired or was already used. Log in and we'll email you a new one.",
  verified: "Your email is verified. Log in to continue.",
};

// The app's own copy, never the error_description from the query.
export function getSignInErrorNotice(error: string | null) {
  return error ? (SIGN_IN_ERROR_COPY.get(error) ?? null) : null;
}
