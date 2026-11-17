import { isMicrosoftMailboxAccount } from "./microsoft-login";

// Gmail's own domains: a Google account with one of these addresses is that
// mailbox.
const GOOGLE_MAIL_DOMAINS = new Set(["gmail.com", "googlemail.com"]);

interface Claim {
  email: string;
  emailVerified: boolean;
  // The provider's raw claims (Better Auth's `source.oauth.profile`).
  profile: Record<string, unknown> | undefined;
}

/**
 * Whether a provider sign-in proves its holder controls `email` as surely as
 * a reset link mailed there would: when the provider runs that mailbox. A
 * Gmail address through Google, or an Outlook.com one through a personal
 * Microsoft account, is the account itself.
 *
 * Any other address is one the provider says it checked: a company domain in
 * Google Workspace or Microsoft Entra, a Google or Microsoft account opened
 * with someone's other address, a GitHub or Discord login. Such checks have
 * failed before (Google's Workspace verification was bypassed in 2024; in
 * 2026 a sign-in provider vouched for addresses it never verified and opened
 * thousands of Dropbox accounts), so those sign-ins join an existing account
 * only through a code emailed to the address (sign-in-code-plugin.ts).
 */
export function providerRunsMailbox(providerId: string, claim: Claim) {
  switch (providerId) {
    case "google":
      return claim.emailVerified && GOOGLE_MAIL_DOMAINS.has(domainOf(claim));
    case "microsoft":
      return (
        claim.profile !== undefined && isMicrosoftMailboxAccount(claim.profile)
      );
    default:
      return false;
  }
}

function domainOf({ email }: Claim) {
  return email.slice(email.lastIndexOf("@") + 1).toLowerCase();
}
