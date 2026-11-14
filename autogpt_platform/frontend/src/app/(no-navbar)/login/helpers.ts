import { withAuthTracking } from "@/lib/auth-redirect";
import { ACCOUNT_CLOSED_PATH } from "@/lib/auth/closed-account";
import type { EmailVerificationNotice } from "@/lib/auth/email-verification";
import { ACCOUNT_NOT_LINKED_CODE } from "@/lib/auth/microsoft-login";

export const ACCOUNT_BANNED_NOTICE_COPY =
  "This account has been banned. If you think this is a mistake, please contact support.";

// Same for an account using the address and for an address the provider
// didn't verify (provider-verified-email.ts), so it says neither.
export const ACCOUNT_NOT_LINKED_NOTICE_COPY =
  "We can't sign you in with that account. If you already have an AutoGPT account, log in the way you usually do. If not, sign up with your email address.";

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
// code. The app's own copy, never the error_description from the query; a
// Microsoft refusal gets Microsoft's (getSocialSignInError).
export function getSignInErrorNotice(
  error: string | null,
  provider: string | null,
) {
  if (error === BANNED_USER_ERROR_CODE) return ACCOUNT_BANNED_NOTICE_COPY;
  if (error === ACCOUNT_NOT_LINKED_CODE && provider !== "microsoft") {
    return ACCOUNT_NOT_LINKED_NOTICE_COPY;
  }
  return null;
}

interface PostLoginPathArgs {
  nextUrl: string | null;
  resultNext: string | undefined;
  rawNext: string | null;
}

// Where a successful sign-in goes. A closed account always goes to the page
// where it can keep the account: a ?next= page would only send it back there
// (an optional-auth page, such as the Marketplace, never would). Otherwise a
// ?next= deep link wins, then the server's choice, which keeps the tracking
// params of a default-landing ?next=.
export function getPostLoginPath({
  nextUrl,
  resultNext,
  rawNext,
}: PostLoginPathArgs) {
  if (resultNext === ACCOUNT_CLOSED_PATH) return ACCOUNT_CLOSED_PATH;
  return nextUrl || withAuthTracking(resultNext || "/", rawNext);
}
