import { APIError } from "better-auth/api";
import { isTeamEmail } from "./team-email";

export { isTeamEmail };

/**
 * AutoGPT team addresses must sign up through Google, which only vouches for
 * an address its owner controls. With email verification off, a password
 * sign-up as `made-up@agpt.co` would otherwise get a session straight away and
 * pass any `@agpt.co` allowlist entry (AUTH_SIGNUP_ALLOWLIST on dev/previews).
 *
 * The sign-up page refuses these addresses in the browser; this is the server
 * half, so a direct POST to /api/auth/sign-up/email is refused too. Only the
 * exact domain counts: `@previews.agpt.co` QA accounts keep password sign-up.
 *
 * An unverified account's /change-email writes the new address on the spot, so
 * it is held to the same rule. A verified account's change is written on
 * /verify-email, from the link mailed to the new address, and stays allowed.
 */

const PASSWORD_SIGN_UP_PATH = "/sign-up/email";
export const CHANGE_EMAIL_PATH = "/change-email";

export const TEAM_EMAIL_REQUIRES_GOOGLE_CODE = "TEAM_EMAIL_REQUIRES_GOOGLE";
export const TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE =
  "Please use Google sign-in to create an account with an AutoGPT email.";
const TEAM_EMAIL_CHANGE_REQUIRES_GOOGLE_MESSAGE =
  "Please sign in with Google to use an AutoGPT email.";

const MESSAGES_BY_PATH = new Map([
  [PASSWORD_SIGN_UP_PATH, TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE],
  [CHANGE_EMAIL_PATH, TEAM_EMAIL_CHANGE_REQUIRES_GOOGLE_MESSAGE],
]);

// Called from the user.create.before and user.update.before hooks. `ctx` is
// the endpoint context writing the user: null for internal callers,
// `/callback/:id` for OAuth.
export function assertTeamEmailUsesGoogle(
  email: string,
  ctx: { path?: string } | null | undefined,
) {
  const message = ctx?.path ? MESSAGES_BY_PATH.get(ctx.path) : undefined;
  if (!message || !isTeamEmail(email)) return;

  throw new APIError("FORBIDDEN", {
    code: TEAM_EMAIL_REQUIRES_GOOGLE_CODE,
    message,
  });
}
