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
  const normalized = email.trim().toLowerCase();
  const at = normalized.lastIndexOf("@");
  if (at === -1) return false;
  return normalized.slice(at + 1) === TEAM_EMAIL_DOMAIN;
}
