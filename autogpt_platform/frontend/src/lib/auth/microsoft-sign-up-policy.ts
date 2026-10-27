import { APIError } from "better-auth/api";
import { UNCONFIRMED_EMAIL_CODE } from "./microsoft-login";

/**
 * "Continue with Microsoft" creates an account only for an address Microsoft
 * confirmed (isMicrosoftEmailVerified). For any other address, whoever holds
 * the Microsoft account might not hold the mailbox, and emailing a link to
 * confirm it would not help: the mailbox's owner could open it, or claim the
 * account later with a reset link, and the other person's Microsoft login
 * would still sign in to it. So nothing is created, and /login tells the
 * person to sign up with their email address instead.
 */

interface SignUpContext {
  path?: string;
  params?: { id?: string };
  body?: unknown;
}

// Called from the user.create.before hook. `ctx` is the endpoint context that
// is creating the user: `/callback/:id` names the provider in its params, and
// `/sign-in/social` (ID tokens, switched off for Microsoft) in its body.
export function assertMicrosoftSignUpConfirmed(
  user: { emailVerified?: boolean | null },
  ctx: SignUpContext | null | undefined,
) {
  if (!isMicrosoftSignUp(ctx) || user.emailVerified) return;
  // Better Auth sends a refused OAuth sign-up to its error URL with this
  // message as the `error` code.
  throw new APIError("FORBIDDEN", { message: UNCONFIRMED_EMAIL_CODE });
}

function isMicrosoftSignUp(ctx: SignUpContext | null | undefined) {
  const body = ctx?.body;
  const provider =
    typeof body === "object" && body !== null && "provider" in body
      ? body.provider
      : undefined;
  return ctx?.params?.id === "microsoft" || provider === "microsoft";
}
