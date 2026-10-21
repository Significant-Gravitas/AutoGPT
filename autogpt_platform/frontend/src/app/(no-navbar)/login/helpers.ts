import {
  CLOSED_ACCOUNT_ERROR_CODE,
  CLOSED_ACCOUNT_MESSAGE,
} from "@/lib/auth/closed-account";
import type { AccountRefusal } from "@/lib/auth/account-banned";
import type { EmailVerificationNotice } from "@/lib/auth/email-verification";

const ACCOUNT_BANNED_NOTICE_COPY =
  "This account has been banned. If you think this is a mistake, please contact support.";

// Shown after a refused account was signed out (lib/auth/account-banned.ts).
export const ACCOUNT_NOTICE_COPY: Record<AccountRefusal, string> = {
  banned: ACCOUNT_BANNED_NOTICE_COPY,
  closed: CLOSED_ACCOUNT_MESSAGE,
};

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
  return error === CLOSED_ACCOUNT_ERROR_CODE ? CLOSED_ACCOUNT_MESSAGE : null;
}
