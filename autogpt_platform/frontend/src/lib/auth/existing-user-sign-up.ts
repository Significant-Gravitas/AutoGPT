import { randomBytes } from "node:crypto";
import {
  type AuthEmailContext,
  claimEmailSlot,
  createVerification,
} from "./auth-email-cooldown";
import { sendAuthEmail } from "./email";

// Better Auth's own default for a reset link.
const RESET_LINK_EXPIRES_IN_SECONDS = 60 * 60;

interface Args {
  user: { id: string; email: string; emailVerified?: boolean | null };
  getAuthContext: () => Promise<AuthEmailContext>;
  resetRedirectTo: string;
}

/**
 * With email verification required, Better Auth answers a sign-up for an
 * address that already has an account exactly like a new one (no session, no
 * error), so as not to reveal that the account exists, and the sign-up page
 * then says to check the inbox. A verified account gets nothing here.
 *
 * An unverified one gets a password reset link, not a verification link, at
 * most once per address per window (auth-email-cooldown.ts). If the account
 * has never held a session, its password was set by a sign-up whose author
 * need not own the address: a verification link would sign the owner in to an
 * account someone else still has the password to. So that password is first
 * replaced with a random one. An account that has held a session (one from
 * before the flag) keeps its password: its owner set it, and a stranger
 * signing up with their address must not lock them out. Either way the reset
 * link lets the owner set a password, and opening it proves they hold the
 * address, so it verifies it too (onPasswordReset).
 *
 * Better Auth runs this alongside the response (see background-tasks.ts), so
 * none of it shows in the response's timing. Never throws: a failure must not
 * turn this response into an error that a brand-new address would not get.
 */
export async function emailRepeatSignUp({
  user,
  getAuthContext,
  resetRedirectTo,
}: Args) {
  if (user.emailVerified) return;
  try {
    const context = await getAuthContext();
    if (!(await claimEmailSlot(context, "repeat-sign-up", user.email))) return;
    if (!(await hasHeldSession(context, user.id))) {
      await context.adapter.updateMany({
        model: "account",
        where: [
          { field: "userId", value: user.id },
          { field: "providerId", value: "credential" },
        ],
        update: { password: await context.password.hash(randomToken(32)) },
      });
    }
    await sendPasswordReset(context, user, resetRedirectTo);
  } catch (error) {
    console.error("Failed to handle a repeat sign-up", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

// The same test the orphan sweep uses: with the flag on, a password sign-up
// gets no session until its address is verified.
async function hasHeldSession(context: AuthEmailContext, userId: string) {
  const sessions = await context.adapter.findMany({
    model: "session",
    where: [{ field: "userId", value: userId }],
    limit: 1,
  });
  return sessions.length > 0;
}

// The same token and link Better Auth's request-password-reset issues, so its
// /reset-password endpoint redeems it.
async function sendPasswordReset(
  context: AuthEmailContext,
  user: Args["user"],
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
    type: "reset_password",
    url: `${context.baseURL}/reset-password/${token}?callbackURL=${encodeURIComponent(redirectTo)}`,
  });
}

function randomToken(bytes: number) {
  return randomBytes(bytes).toString("base64url");
}
