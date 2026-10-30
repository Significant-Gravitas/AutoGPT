/**
 * Client-safe half of the team-email policy: whether an address is an
 * AutoGPT team address. Lives apart from `team-email-policy` so the
 * login/signup hooks don't pull `better-auth/api` into the browser bundle.
 *
 * Only the exact domain counts: `@previews.agpt.co` QA accounts keep
 * password sign-up, and look-alike domains (`agpt.com`, `agpt.co.uk`,
 * `agpt.company`) are not team addresses.
 */

const TEAM_EMAIL_DOMAIN = "agpt.co";

export function isTeamEmail(email: string) {
  const at = email.lastIndexOf("@");
  if (at === -1) return false;
  return email.slice(at + 1).toLowerCase() === TEAM_EMAIL_DOMAIN;
}
