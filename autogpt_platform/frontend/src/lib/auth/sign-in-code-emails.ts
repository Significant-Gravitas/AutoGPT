import { createHash } from "node:crypto";
import { isAPIError } from "better-auth/api";
import { isAuthEmailNotSent } from "./auth-email-not-sent";
import {
  type AuthEmailContext,
  claimWindowSlot,
  releaseWindowSlot,
} from "./auth-email-cooldown";
import { sendAuthEmail } from "./email";
import {
  isSignInProvider,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import type { PendingSignIn } from "./sign-in-code-store";
import { UNDELIVERABLE_EMAIL_CODE } from "./undeliverable-email";

// Documented in docs/platform/rate-limits.md: update it with any change here.
const CODE_EMAILS_PER_ADDRESS = { max: 5, windowSeconds: 60 * 60 };
// The provider's callback waits on the code's email before it redirects, so
// a mailer that stalls fails the sign-in instead of hanging it.
const SEND_TIMEOUT_MS = 10_000;

type CodeEmailClaim =
  | { slot: string }
  | { refusal: typeof SIGN_IN_CODE_LIMIT | typeof SIGN_IN_CODE_UNSENT };

/**
 * One of the five codes an hour that may reach an address, from sign-ins and
 * resends together. Unlike the other auth emails, a count that can't be read
 * refuses the code: the limit also caps how many codes anyone can guess at.
 */
export async function claimCodeEmail(
  context: AuthEmailContext,
  email: string,
): Promise<CodeEmailClaim> {
  const key = createHash("sha256").update(email.toLowerCase()).digest("hex");
  try {
    const slot = await claimWindowSlot(
      context,
      `sign-in-code-email:${key}`,
      CODE_EMAILS_PER_ADDRESS,
    );
    return slot ? { slot } : { refusal: SIGN_IN_CODE_LIMIT };
  } catch (error) {
    logFailure("Failed to check the sign-in code email limit", error);
    return { refusal: SIGN_IN_CODE_UNSENT };
  }
}

/**
 * Sends the code, cancelling the request if the backend takes too long. A
 * send that failed gives its hourly slot back; one cancelled mid-request keeps
 * it, since its email may still go out.
 */
export async function sendCode(
  context: AuthEmailContext,
  pending: PendingSignIn,
  code: string,
  slot: string,
) {
  const cancel = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  // The whole send, the service token it mints first included, gets the
  // deadline; the request to the backend is also cancelled when it passes.
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      cancel.abort();
      reject(new Error(`No answer in ${SEND_TIMEOUT_MS} ms`));
    }, SEND_TIMEOUT_MS);
  });
  const message = {
    to: pending.email,
    type: "sign_in_code",
    code,
    provider: pending.providerId,
  } as const;
  try {
    await Promise.race([
      sendAuthEmail(message, { signal: cancel.signal }),
      deadline,
    ]);
    return true;
  } catch (error) {
    logFailure("Failed to send a sign-in code", error);
    // Only a code that surely didn't go out gives its slot back: a send cut
    // off, or lost after it left, may still have reached the address.
    if (surelyNotSent(error)) await releaseCodeEmail(context, slot);
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Gives back an hourly slot whose code never went out. */
export async function releaseCodeEmail(
  context: AuthEmailContext,
  slot: string,
) {
  await releaseWindowSlot(context, slot).catch((error: unknown) =>
    logFailure("Failed to give back a sign-in code's slot", error),
  );
}

// The users each request created, by its endpoint context: Better Auth hands
// the same one to every database hook a request runs.
const signUps = new WeakMap<object, Set<string>>();

/** Notes a user this request created (Better Auth's `user.create.after`). */
export function noteSignUp(ctx: object | null, userID: string) {
  if (!ctx) return;
  signUps.set(ctx, (signUps.get(ctx) ?? new Set<string>()).add(userID));
}

/**
 * Whether a new account row joins an account, rather than being the one its
 * user signed up with, which the same request writes. Counting the account's
 * rows instead would miss an account that had none, such as one copied from
 * Supabase with neither a password nor a provider.
 */
export function joinsAnAccount(
  account: { userId: string },
  ctx: object | null,
) {
  return !(ctx && signUps.get(ctx)?.has(account.userId));
}

/**
 * Emails the account's owner that a sign-in provider joined their account
 * (`joinsAnAccount`). Called after the account row is written.
 */
export async function alertNewSignInMethod(
  context: AuthEmailContext,
  account: { userId: string; providerId: string },
) {
  if (!isSignInProvider(account.providerId)) return;
  const [user] = (await context.adapter.findMany({
    model: "user",
    where: [{ field: "id", value: account.userId }],
    limit: 1,
  })) as Array<{ email?: string }>;
  if (!user?.email) return;
  await sendAuthEmail({
    to: user.email,
    type: "sign_in_method_added",
    provider: account.providerId,
  });
}

function logFailure(message: string, error: unknown) {
  console.error(message, {
    error: error instanceof Error ? error.message : String(error),
  });
}

// The backend answered without sending, or the address takes no mail.
function surelyNotSent(error: unknown) {
  return (
    isAuthEmailNotSent(error) ||
    (isAPIError(error) && error.body?.code === UNDELIVERABLE_EMAIL_CODE)
  );
}
