// "Continue with Microsoft". Signing in only proves who someone is; mail,
// files and the rest are a separate, optional step afterwards.

import { isWaitlistError } from "@/app/api/auth/utils";
import { UNDELIVERABLE_EMAIL_CODE } from "./undeliverable-email";

// Microsoft's fixed tenant for personal (consumer) accounts.
const PERSONAL_TENANT_ID = "9188040d-6c67-4c5b-b112-36a304b66dad";

// Outlook.com's own mail domains, where a personal account's address is that
// account's mailbox. Each one's primary MX was Outlook.com's consumer servers
// (*.olc.protection.outlook.com) on 2026-10-31. A Microsoft-sounding domain
// whose mail goes anywhere else first is left out (outlook.co.uk is someone
// else's; hotmail.nl, hotmail.cz and outlook.cz route to a Microsoft 365
// organization, outlook.cz with Outlook.com only as its backup):
// an address there can be an alias the account keeps after the mailbox is
// gone.
const MICROSOFT_MAIL_DOMAINS = new Set([
  "hotmail.at",
  "hotmail.be",
  "hotmail.ca",
  "hotmail.cl",
  "hotmail.co.id",
  "hotmail.co.il",
  "hotmail.co.in",
  "hotmail.co.jp",
  "hotmail.co.kr",
  "hotmail.co.nz",
  "hotmail.co.th",
  "hotmail.co.uk",
  "hotmail.co.za",
  "hotmail.com",
  "hotmail.com.ar",
  "hotmail.com.au",
  "hotmail.com.br",
  "hotmail.com.hk",
  "hotmail.com.my",
  "hotmail.com.ph",
  "hotmail.com.sg",
  "hotmail.com.tr",
  "hotmail.com.tw",
  "hotmail.com.vn",
  "hotmail.de",
  "hotmail.dk",
  "hotmail.es",
  "hotmail.fi",
  "hotmail.fr",
  "hotmail.gr",
  "hotmail.hk",
  "hotmail.hu",
  "hotmail.ie",
  "hotmail.it",
  "hotmail.jp",
  "hotmail.lt",
  "hotmail.lv",
  "hotmail.my",
  "hotmail.no",
  "hotmail.ph",
  "hotmail.pt",
  "hotmail.rs",
  "hotmail.se",
  "hotmail.sg",
  "hotmail.sk",
  "live.at",
  "live.be",
  "live.ca",
  "live.cl",
  "live.cn",
  "live.co.in",
  "live.co.kr",
  "live.co.uk",
  "live.co.za",
  "live.com",
  "live.com.ar",
  "live.com.au",
  "live.com.mx",
  "live.com.my",
  "live.com.pe",
  "live.com.ph",
  "live.com.pt",
  "live.com.sg",
  "live.de",
  "live.dk",
  "live.fi",
  "live.fr",
  "live.hk",
  "live.ie",
  "live.in",
  "live.it",
  "live.jp",
  "live.nl",
  "live.no",
  "live.ph",
  "live.ru",
  "live.se",
  "msn.com",
  "msn.nl",
  "outlook.at",
  "outlook.be",
  "outlook.cl",
  "outlook.co.id",
  "outlook.co.il",
  "outlook.co.nz",
  "outlook.co.th",
  "outlook.com",
  "outlook.com.ar",
  "outlook.com.au",
  "outlook.com.br",
  "outlook.com.gr",
  "outlook.com.pe",
  "outlook.com.tr",
  "outlook.com.vn",
  "outlook.de",
  "outlook.dk",
  "outlook.es",
  "outlook.fr",
  "outlook.hu",
  "outlook.ie",
  "outlook.in",
  "outlook.it",
  "outlook.jp",
  "outlook.kr",
  "outlook.lv",
  "outlook.my",
  "outlook.ph",
  "outlook.pt",
  "outlook.sa",
  "outlook.sg",
  "outlook.sk",
  "passport.com",
  "windowslive.com",
]);

// The `error` a refused sign-up returns to /login with (see
// microsoft-sign-up-policy.ts). `account_not_linked` is Better Auth's own
// answer when an account already uses the address.
export const ACCOUNT_NOT_LINKED_CODE = "account_not_linked";
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
    // Name and email only: the Graph photo would sit in the user row and in
    // every session cookie.
    disableProfilePhoto: true,
    mapProfileToUser: microsoftProfileToUser,
  };
}

const SIGN_IN_ERRORS: Record<string, string> = {
  [TEAM_EMAIL_SIGN_UP_CODE]:
    "An AutoGPT team address signs up with Google. Use Continue with Google instead.",
  email_not_found:
    "Microsoft didn't share an email address for that account, so it can't sign in here. Use another sign-in method.",
  // Also what an address Microsoft didn't confirm gets, so it doesn't say
  // whether an account uses the address.
  [ACCOUNT_NOT_LINKED_CODE]:
    "We can't sign you in with that Microsoft account. If you already have an AutoGPT account, log in the way you usually do. If not, sign up with your email address. You can connect Microsoft 365 afterwards.",
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
