// "Continue with Microsoft". Signing in only proves who someone is; mail,
// files and the rest are a separate, optional step afterwards.

import { isWaitlistError } from "@/app/api/auth/utils";
import { UNDELIVERABLE_EMAIL_CODE } from "./undeliverable-email";

// Microsoft's fixed tenant for personal (consumer) accounts.
const PERSONAL_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad";

// Mail domains Microsoft runs itself, where a personal account's address is
// that account's own mailbox. The same Microsoft domains as the backend's
// PUBLIC_EMAIL_DOMAINS (backend/data/access_blocks.py).
const MICROSOFT_MAIL_DOMAINS = new Set([
  "hotmail.be",
  "hotmail.ca",
  "hotmail.co.jp",
  "hotmail.co.uk",
  "hotmail.com",
  "hotmail.com.au",
  "hotmail.de",
  "hotmail.es",
  "hotmail.fr",
  "hotmail.it",
  "hotmail.nl",
  "live.be",
  "live.ca",
  "live.co.uk",
  "live.com",
  "live.com.au",
  "live.de",
  "live.fr",
  "live.ie",
  "live.it",
  "live.jp",
  "live.nl",
  "msn.com",
  "outlook.be",
  "outlook.co.uk",
  "outlook.com",
  "outlook.com.au",
  "outlook.de",
  "outlook.es",
  "outlook.fr",
  "outlook.ie",
  "outlook.in",
  "outlook.it",
  "outlook.jp",
  "windowslive.com",
]);

// The `error` a refused sign-up returns to /login with (see
// microsoft-sign-up-policy.ts).
export const UNCONFIRMED_EMAIL_CODE = "unconfirmed_email";
export const TEAM_EMAIL_SIGN_UP_CODE = "team_email_requires_google";

interface MicrosoftProfile {
  sub: string;
  oid?: string;
  tid?: string;
  email?: string;
  xms_edov?: unknown;
  // Who authenticated the person. Absent, or naming the issuing tenant, for a
  // member; a guest's own provider (live.com, another tenant) otherwise.
  idp?: string;
}

/** Microsoft vouches for a personal account's own Outlook.com address, and
 *  for a work or school address when the organization has verified that
 *  domain (`xms_edov`). A personal account can carry any other address, and
 *  keeps it after its owner loses the mailbox; any other work address is
 *  whatever an organization's admin typed in. A guest signs in through an
 *  organization that isn't its own, which then vouches for addresses it never
 *  verified, so a guest's address is never trusted. */
export function isMicrosoftEmailVerified(profile: MicrosoftProfile): boolean {
  if (profile.tid === PERSONAL_TENANT_ID) {
    return isMicrosoftMailbox(profile.email);
  }
  if (isGuest(profile)) return false;
  const verified = profile.xms_edov;
  return (
    verified === true ||
    verified === "true" ||
    verified === 1 ||
    verified === "1"
  );
}

function isGuest({ idp, tid }: MicrosoftProfile) {
  if (!idp) return false;
  return (
    idp !== `https://sts.windows.net/${tid}/` &&
    idp !== `https://login.microsoftonline.com/${tid}/v2.0`
  );
}

function isMicrosoftMailbox(email: string | undefined) {
  const domain = email?.slice(email.lastIndexOf("@") + 1).toLowerCase();
  return !!domain && MICROSOFT_MAIL_DOMAINS.has(domain);
}

export function microsoftProfileToUser(profile: MicrosoftProfile) {
  // No `id`: Better Auth keys the account on the `oid` claim, never the
  // address, since a personal and a work account can share one. `oid` also
  // stays the same when the deployment moves to a different Microsoft app.
  return {
    // An unconfirmed address joins no existing account (Better Auth answers
    // `account_not_linked`) and creates none (microsoft-sign-up-policy.ts).
    emailVerified: isMicrosoftEmailVerified(profile),
    email: profile.email,
  };
}

export function getMicrosoftLoginProvider() {
  const clientId = process.env.AUTH_MICROSOFT_CLIENT_ID;
  const clientSecret = process.env.AUTH_MICROSOFT_CLIENT_SECRET;
  if (!clientId || !clientSecret) return null;
  return {
    clientId,
    clientSecret,
    tenantId: "common",
    // Personal and work accounts can share an address, so always let the
    // person pick which one they mean.
    prompt: "select_account" as const,
    disableDefaultScope: true,
    scope: ["openid", "profile", "email", "User.Read"],
    // Only the redirect flow, which microsoft-sign-up-policy.ts guards, signs
    // in. An ID token handed straight to /sign-in/social would skip it.
    disableIdTokenSignIn: true,
    mapProfileToUser: microsoftProfileToUser,
  };
}

const SIGN_IN_ERRORS: Record<string, string> = {
  [TEAM_EMAIL_SIGN_UP_CODE]:
    "An AutoGPT team address signs up with Google. Use Continue with Google instead.",
  [UNCONFIRMED_EMAIL_CODE]:
    "Microsoft hasn't confirmed the email address on that account, so it can't be used to sign up here. Sign up with your email address instead. You can still connect Microsoft 365 afterwards.",
  email_not_found:
    "Microsoft didn't share an email address for that account, so it can't sign in here. Use another sign-in method.",
  account_not_linked:
    "An AutoGPT account already uses that email. Log in the way you usually do.",
  access_denied: "Microsoft sign-in was cancelled.",
  // access-blocks.ts, which is server-only.
  SIGN_IN_BLOCKED:
    "This sign-in isn't allowed. Contact support if you think this is a mistake.",
};

// Better Auth sends a sign-up one of our checks refused back with the
// check's code (access-blocks.ts, disposable-email.ts, undeliverable-email.ts;
// the first two are server-only, so their codes are written out here). The
// message that comes with it is never shown, since anyone can put anything
// in the URL.
const REFUSED_SIGN_UP_CODES = new Set([
  "SIGN_UP_BLOCKED",
  "DISPOSABLE_EMAIL",
  UNDELIVERABLE_EMAIL_CODE,
]);
const REFUSED_SIGN_UP =
  "This sign-up isn't allowed. Use another sign-in method, or contact support if you think this is a mistake.";

/** A sign-up the signup gate refused (SIGN_UP_NOT_ALLOWED, signup-gate.ts). */
export function isWaitlistSignInError(code: string | null): boolean {
  if (!code) return false;
  return (
    code === "FORBIDDEN" || isWaitlistError(null, code.replaceAll("_", " "))
  );
}

export function getSocialSignInError(code: string | null): string | null {
  if (!code) return null;
  if (Object.hasOwn(SIGN_IN_ERRORS, code)) return SIGN_IN_ERRORS[code];
  if (REFUSED_SIGN_UP_CODES.has(code)) return REFUSED_SIGN_UP;
  return "That sign-in didn't finish. Try again, or use another sign-in method.";
}
