// "Continue with Microsoft". Signing in only proves who someone is; mail,
// files and the rest are a separate, optional step afterwards.

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

interface MicrosoftProfile {
  sub: string;
  oid?: string;
  tid?: string;
  email?: string;
  xms_edov?: unknown;
}

/** Microsoft vouches for a personal account's own Outlook.com address, and
 *  for a work or school address when the organization has verified that
 *  domain (`xms_edov`). A personal account can carry any other address, and
 *  keeps it after its owner loses the mailbox; any other work address is
 *  whatever an organization's admin typed in. */
export function isMicrosoftEmailVerified(profile: MicrosoftProfile): boolean {
  if (profile.tid === PERSONAL_TENANT_ID) {
    return isMicrosoftMailbox(profile.email);
  }
  const verified = profile.xms_edov;
  return (
    verified === true ||
    verified === "true" ||
    verified === 1 ||
    verified === "1"
  );
}

function isMicrosoftMailbox(email: string | undefined) {
  const domain = email?.slice(email.lastIndexOf("@") + 1).toLowerCase();
  return !!domain && MICROSOFT_MAIL_DOMAINS.has(domain);
}

export function microsoftProfileToUser(profile: MicrosoftProfile) {
  return {
    // The account, not its address, is the identity: a personal and a work
    // account can share one. Tenant plus object ID also survives the
    // deployment moving to a different Microsoft app.
    id: `${profile.tid ?? "unknown"}:${profile.oid ?? profile.sub}`,
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
    mapProfileToUser: microsoftProfileToUser,
  };
}

const SIGN_IN_ERRORS: Record<string, string> = {
  [UNCONFIRMED_EMAIL_CODE]:
    "Microsoft hasn't confirmed the email address on that account, so it can't be used to sign up here. Sign up with your email address instead. You can still connect Microsoft 365 afterwards.",
  email_not_found:
    "Microsoft didn't share an email address for that account, so it can't sign in here. Use another sign-in method.",
  account_not_linked:
    "An AutoGPT account already uses that email. Log in the way you usually do.",
  access_denied: "Microsoft sign-in was cancelled.",
};

export function getSocialSignInError(code: string | null): string | null {
  if (!code) return null;
  return (
    SIGN_IN_ERRORS[code] ??
    "That sign-in didn't finish. Try again, or use another sign-in method."
  );
}
