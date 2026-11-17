import { APIError } from "better-auth/api";
import { TEAM_EMAIL_SIGN_UP_CODE } from "./microsoft-login";
import { signInProviderOf } from "./sign-in-provider";
import { isTeamEmail } from "./team-email";

/**
 * "Continue with Microsoft" creates an account straight away only for an
 * address Microsoft confirmed (isMicrosoftEmailVerified, its `emailVerified`).
 * For any other address, whoever holds the Microsoft account might not hold
 * the mailbox, so the sign-up first emails the address a code, typed into the
 * page that started it (sign-in-code-plugin.ts). A link to click would not do:
 * the mailbox's owner could open it for someone else's sign-up, and that
 * person's Microsoft login would then sign in to the account.
 *
 * An AutoGPT team address gets an account only through Google, as with every
 * other way to sign up; a team member's existing account still signs in with
 * Microsoft.
 */

interface SignUpContext {
  path?: string;
  params?: { id?: string };
  body?: unknown;
}

// Called from the user.create.before hook. `ctx` is the endpoint context that
// is creating the user (sign-in-provider.ts). Better Auth sends a refused
// OAuth sign-up to its error URL with the thrown `code` as `error`, which
// /login explains, and the code's endpoint answers with it. An APIError
// without a code ends the callback on a raw 403 instead.
export function assertMicrosoftSignUpAllowed(
  user: { email: string },
  ctx: SignUpContext | null | undefined,
) {
  if (signInProviderOf(ctx) !== "microsoft") return;
  if (isTeamEmail(user.email)) {
    throw new APIError("FORBIDDEN", {
      code: TEAM_EMAIL_SIGN_UP_CODE,
      message: "An AutoGPT team address signs up with Google.",
    });
  }
}
