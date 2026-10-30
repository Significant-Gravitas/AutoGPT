import { APIError } from "better-auth/api";
import {
  TEAM_EMAIL_SIGN_UP_CODE,
  UNCONFIRMED_EMAIL_CODE,
} from "./microsoft-login";
import { isTeamEmail } from "./team-email";

/**
 * "Continue with Microsoft" creates an account only for an address Microsoft
 * confirmed (isMicrosoftEmailVerified). For any other address, whoever holds
 * the Microsoft account might not hold the mailbox, and emailing a link to
 * confirm it would not help: the mailbox's owner could open it, or claim the
 * account later with a reset link, and the other person's Microsoft login
 * would still sign in to it. So nothing is created, and /login tells the
 * person to sign up with their email address instead.
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
// is creating the user: `/callback/:id` names the provider in its params, and
// `/sign-in/social` (ID tokens, switched off for Microsoft) in its body.
// Better Auth sends a refused OAuth sign-up to its error URL with the thrown
// message as the `error` code, which /login explains.
export function assertMicrosoftSignUpAllowed(
  user: { email: string; emailVerified?: boolean | null },
  ctx: SignUpContext | null | undefined,
) {
  if (!isMicrosoftSignUp(ctx)) return;
  if (!user.emailVerified) {
    throw new APIError("FORBIDDEN", { message: UNCONFIRMED_EMAIL_CODE });
  }
  if (isTeamEmail(user.email)) {
    throw new APIError("FORBIDDEN", { message: TEAM_EMAIL_SIGN_UP_CODE });
  }
}

function isMicrosoftSignUp(ctx: SignUpContext | null | undefined) {
  const body = ctx?.body;
  const provider =
    typeof body === "object" && body !== null && "provider" in body
      ? body.provider
      : undefined;
  return ctx?.params?.id === "microsoft" || provider === "microsoft";
}
