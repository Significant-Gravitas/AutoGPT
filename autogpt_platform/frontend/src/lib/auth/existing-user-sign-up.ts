import { randomBytes } from "node:crypto";
import { sendAuthEmail } from "./email";

// One email per address per window, however often it is signed up again.
export const REPEAT_SIGN_UP_EMAIL_COOLDOWN_SECONDS = 10 * 60;
// Better Auth's own default for a reset link.
const RESET_LINK_EXPIRES_IN_SECONDS = 60 * 60;

interface Where {
  field: string;
  value: string | Date;
  operator?: "eq" | "gt";
}

export interface RepeatSignUpContext {
  baseURL: string;
  adapter: {
    findMany: (args: {
      model: string;
      where: Where[];
      limit: number;
    }) => Promise<unknown[]>;
    create: (args: {
      model: string;
      data: Record<string, unknown>;
    }) => Promise<unknown>;
    updateMany: (args: {
      model: string;
      where: Where[];
      update: Record<string, unknown>;
    }) => Promise<unknown>;
  };
  password: { hash: (password: string) => Promise<string> };
}

interface Args {
  user: { id: string; email: string; emailVerified?: boolean | null };
  getAuthContext: () => Promise<RepeatSignUpContext>;
  resetRedirectTo: string;
}

/**
 * With email verification required, Better Auth answers a sign-up for an
 * address that already has an account exactly like a new one (no session, no
 * error), so as not to reveal that the account exists, and the sign-up page
 * then says to check the inbox. A verified account gets nothing here.
 *
 * An unverified one gets a password reset link, not a verification link. Its
 * password was set by whoever signed up first, who need not own the address:
 * a verification link would sign the owner in to an account someone else
 * still has the password to. So the old password is replaced with a random
 * one at once, and the reset link lets the owner set their own; opening it
 * proves they hold the address, so it verifies it too (onPasswordReset).
 *
 * Better Auth runs this alongside the response (see background-tasks.ts), so
 * neither the work nor the cooldown shows in its timing. By then the sign-up's
 * database transaction has committed, and Better Auth's internal adapter (and
 * any auth.api call) would still go through it, so this works on the base
 * adapter directly. Never throws: a failure must not turn this response into
 * an error that a brand-new address would not get.
 */
export async function emailRepeatSignUp({
  user,
  getAuthContext,
  resetRedirectTo,
}: Args) {
  if (user.emailVerified) return;
  try {
    const context = await getAuthContext();
    await context.adapter.updateMany({
      model: "account",
      where: [
        { field: "userId", value: user.id },
        { field: "providerId", value: "credential" },
      ],
      update: { password: await context.password.hash(randomToken(32)) },
    });
    if (!(await claimEmailSlot(context, user.email))) return;
    await sendPasswordReset(context, user, resetRedirectTo);
  } catch (error) {
    console.error("Failed to handle a repeat sign-up", {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

// Kept in Better Auth's verification table, which every server shares and
// which Better Auth clears of expired rows itself. Two sign-ups in the same
// instant can both pass, which costs one extra email, not an unbounded number.
async function claimEmailSlot(context: RepeatSignUpContext, email: string) {
  const identifier = `repeat-sign-up:${email.toLowerCase()}`;
  const live = await context.adapter.findMany({
    model: "verification",
    where: [
      { field: "identifier", value: identifier },
      { field: "expiresAt", value: new Date(), operator: "gt" },
    ],
    limit: 1,
  });
  if (live.length) return false;
  await createVerification(context, {
    identifier,
    value: "sent",
    expiresInSeconds: REPEAT_SIGN_UP_EMAIL_COOLDOWN_SECONDS,
  });
  return true;
}

// The same token and link Better Auth's request-password-reset issues, so its
// /reset-password endpoint redeems it.
async function sendPasswordReset(
  context: RepeatSignUpContext,
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

function createVerification(
  context: RepeatSignUpContext,
  args: { identifier: string; value: string; expiresInSeconds: number },
) {
  const now = new Date();
  return context.adapter.create({
    model: "verification",
    data: {
      identifier: args.identifier,
      value: args.value,
      expiresAt: new Date(now.getTime() + args.expiresInSeconds * 1000),
      createdAt: now,
      updatedAt: now,
    },
  });
}

function randomToken(bytes: number) {
  return randomBytes(bytes).toString("base64url");
}
