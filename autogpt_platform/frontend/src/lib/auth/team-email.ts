/**
 * Client-safe team-email check, shared by the login/signup pages and the
 * server-side sign-up policy (`team-email-policy.ts`). Kept free of server
 * imports so the browser bundle doesn't pull in `better-auth/api`.
 *
 * Only the exact domain counts: `@agpt.com`, `@agpt.co.uk` and
 * `@previews.agpt.co` are not team addresses; case doesn't matter.
 */
export const TEAM_EMAIL_DOMAIN = "agpt.co";

export function isTeamEmail(email: string) {
  const at = email.lastIndexOf("@");
  if (at === -1) return false;
  return email.slice(at + 1).toLowerCase() === TEAM_EMAIL_DOMAIN;
}
