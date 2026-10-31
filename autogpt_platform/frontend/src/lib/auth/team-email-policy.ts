import { APIError } from "better-auth/api";
import { isTeamEmail } from "./team-email";

export { isTeamEmail };

/**
 * An AutoGPT team address gets an account only through Google, which only
 * vouches for an address its owner controls, or from an admin. So an
 * `@agpt.co` allowlist entry (AUTH_SIGNUP_ALLOWLIST on dev/previews) admits
 * team members only.
 *
 * The sign-up page refuses these addresses in the browser; this is the server
 * half, so a direct POST to /api/auth/sign-up/email is refused too. Only the
 * exact domain counts: `@previews.agpt.co` QA accounts keep password sign-up.
 */

const ADMIN_CREATE_USER_PATH = "/admin/create-user";
const OAUTH_CALLBACK_PATH = "/callback/:id";

export const TEAM_EMAIL_REQUIRES_GOOGLE_CODE = "TEAM_EMAIL_REQUIRES_GOOGLE";
export const TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE =
  "Please use Google sign-in to create an account with an AutoGPT email.";

// Called from the user.create.before hook. Its `path` is the route as declared,
// so an OAuth callback is `/callback/:id` with the provider in `params.id`.
export function assertTeamEmailUsesGoogle(
  email: string,
  ctx:
    | { path?: string; params?: Record<string, string | undefined> }
    | null
    | undefined,
) {
  if (!isTeamEmail(email) || ctx?.path === ADMIN_CREATE_USER_PATH) return;
  if (ctx?.path === OAUTH_CALLBACK_PATH && ctx.params?.id === "google") return;

  throw new APIError("FORBIDDEN", {
    code: TEAM_EMAIL_REQUIRES_GOOGLE_CODE,
    message: TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE,
  });
}
