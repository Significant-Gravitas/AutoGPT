import { type AuthEmailContext, claimEmailSlot } from "./auth-email-cooldown";
import {
  hasAccountBeenUsed,
  randomToken,
  sendSetPasswordLink,
} from "./set-password-link";

interface Args {
  user: { id: string; email: string; emailVerified?: boolean | null };
  getAuthContext: () => Promise<AuthEmailContext>;
  hasPlatformUser: (userId: string) => Promise<boolean>;
  resetRedirectTo: string;
}

/**
 * With email verification required, Better Auth answers a sign-up for an
 * address that already has an account exactly like a new one (no session, no
 * error), so as not to reveal that the account exists, and the sign-up page
 * then says to check the inbox. A verified account gets nothing here.
 *
 * An unverified one gets a set-password link, not a verification link, at
 * most once per address per window (auth-email-cooldown.ts). A verification
 * link would sign the owner in to an account whose password whoever signed up
 * first still holds. If the account has never been used, that password is
 * replaced with a random one first. A used one (from before the flag) keeps
 * it: its owner may have set it, and a stranger signing up with their address
 * must not lock them out. Either way, setting a password through the link
 * replaces the old one and verifies the address.
 *
 * The cooldown is claimed before any of the work, so repeats can't make us
 * hash or mail more than once a window. Better Auth runs this alongside the
 * response (see background-tasks.ts), so none of it shows in the response's
 * timing. Never throws: a failure must not turn this response into an error
 * that a brand-new address would not get.
 */
export async function emailRepeatSignUp({
  user,
  getAuthContext,
  hasPlatformUser,
  resetRedirectTo,
}: Args) {
  if (user.emailVerified) return;
  try {
    const context = await getAuthContext();
    if (!(await claimEmailSlot(context, "repeat-sign-up", user.email))) return;
    // Unknown counts as never used: the scramble costs a real owner nothing
    // the set-password link below doesn't give back.
    if (!(await hasAccountBeenUsed(context, user.id, hasPlatformUser, false))) {
      await context.adapter.updateMany({
        model: "account",
        where: [
          { field: "userId", value: user.id },
          { field: "providerId", value: "credential" },
        ],
        update: { password: await context.password.hash(randomToken(32)) },
      });
    }
    await sendSetPasswordLink(context, user, resetRedirectTo);
  } catch (error) {
    console.error("Failed to handle a repeat sign-up", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
