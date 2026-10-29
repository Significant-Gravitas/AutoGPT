import { APIError } from "better-auth/api";

/**
 * AutoGPT team addresses must sign up through Google, which only vouches for
 * an address its owner controls. With email verification off, a password
 * sign-up as `made-up@agpt.co` would otherwise get a session straight away and
 * pass any `@agpt.co` allowlist entry (AUTH_SIGNUP_ALLOWLIST on dev/previews).
 *
 * The sign-up page refuses these addresses in the browser; this is the server
 * half, so a direct POST to /api/auth/sign-up/email is refused too. Only the
 * exact domain counts: `@previews.agpt.co` QA accounts keep password sign-up.
 */

const TEAM_EMAIL_DOMAIN = "agpt.co";
const PASSWORD_SIGN_UP_PATH = "/sign-up/email";

export const TEAM_EMAIL_REQUIRES_GOOGLE_CODE = "TEAM_EMAIL_REQUIRES_GOOGLE";
export const TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE =
  "Please use Google sign-in to create an account with an AutoGPT email.";

export function isTeamEmail(email: string) {
  const at = email.lastIndexOf("@");
  if (at === -1) return false;
  return email.slice(at + 1).toLowerCase() === TEAM_EMAIL_DOMAIN;
}

// Called from the user.create.before hook. `ctx` is the endpoint context that
// is creating the user: null for internal callers, `/callback/:id` for OAuth.
export function assertTeamEmailUsesGoogle(
  email: string,
  ctx: { path?: string } | null | undefined,
) {
  if (ctx?.path !== PASSWORD_SIGN_UP_PATH || !isTeamEmail(email)) return;

  throw new APIError("FORBIDDEN", {
    code: TEAM_EMAIL_REQUIRES_GOOGLE_CODE,
    message: TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE,
  });
}
