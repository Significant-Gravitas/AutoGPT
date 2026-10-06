import { randomBytes } from "node:crypto";
import {
  type AuthEmailContext,
  createVerification,
} from "./auth-email-cooldown";
import { sendAuthEmail } from "./email";

// Better Auth's own default for a reset link.
const RESET_LINK_EXPIRES_IN_SECONDS = 60 * 60;

/**
 * Whether the account has been used, i.e. its password was set by someone who
 * then got in. A sign-up while verification was off got a session, and the
 * platform `User` row with it; with verification on, the row waits for the
 * link. Sign-out deletes the session but never the row, so the row is the
 * lasting evidence, and a session still counts in case provisioning failed.
 */
export async function hasAccountBeenUsed(
  context: AuthEmailContext,
  userId: string,
  hasPlatformUser: (userId: string) => Promise<boolean>,
) {
  if (await hasPlatformUser(userId)) return true;
  const sessions = await context.adapter.findMany({
    model: "session",
    where: [{ field: "userId", value: userId }],
    limit: 1,
  });
  return sessions.length > 0;
}

// The same token and link Better Auth's request-password-reset issues, so its
// /reset-password endpoint redeems it, and opening it verifies the address
// (onPasswordReset).
export async function sendSetPasswordLink(
  context: AuthEmailContext,
  user: { id: string; email: string },
  redirectTo: string,
) {
  const token = randomToken(18);
  await createVerification(context, {
    identifier: `reset-password:${token}`,
    value: user.id,
    expiresInSeconds: RESET_LINK_EXPIRES_IN_SECONDS,
  });
  await sendAuthEmail({
    to: user.email,
    type: "set_password",
    url: `${context.baseURL}/reset-password/${token}?callbackURL=${encodeURIComponent(redirectTo)}`,
  });
}

export function randomToken(bytes: number) {
  return randomBytes(bytes).toString("base64url");
}
