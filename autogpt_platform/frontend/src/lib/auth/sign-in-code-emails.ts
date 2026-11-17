import { createHash } from "node:crypto";
import {
  type AuthEmailContext,
  claimWindowSlot,
  releaseWindowSlot,
} from "./auth-email-cooldown";
import { sendAuthEmail, type SignInEmailProvider } from "./email";
import { SIGN_IN_CODE_LIMIT, SIGN_IN_CODE_UNSENT } from "./sign-in-code";
import type { PendingSignIn } from "./sign-in-code-store";

// Documented in docs/platform/rate-limits.md: update it with any change here.
const CODE_EMAILS_PER_ADDRESS = { max: 5, windowSeconds: 60 * 60 };
// The provider's callback waits on the code's email before it redirects, so
// a mailer that stalls fails the sign-in instead of hanging it.
const SEND_TIMEOUT_MS = 10_000;

// The providers a sign-in method notice can name.
const NOTICE_PROVIDERS: ReadonlySet<string> = new Set([
  "google",
  "microsoft",
  "github",
  "discord",
  "apple",
]);

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
  const key = createHash("sha256").update(email).digest("hex");
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

/** Sends the code, giving its hourly slot back when it doesn't go out. */
export async function sendCode(
  context: AuthEmailContext,
  pending: PendingSignIn,
  code: string,
  slot: string,
) {
  try {
    await withTimeout(
      sendAuthEmail({
        to: pending.email,
        type: "sign_in_code",
        code,
        provider: pending.providerId,
      }),
      SEND_TIMEOUT_MS,
    );
    return true;
  } catch (error) {
    logFailure("Failed to send a sign-in code", error);
    await releaseWindowSlot(context, slot).catch((releaseError: unknown) =>
      logFailure("Failed to give back a sign-in code's slot", releaseError),
    );
    return false;
  }
}

/**
 * Emails the account's owner when a sign-in provider joins an account it
 * didn't create. Called after the account row is written; a provider's first
 * row on a new account is its sign-up, which needs no notice.
 */
export async function alertNewSignInMethod(
  context: AuthEmailContext,
  account: { userId: string; providerId: string },
) {
  if (!isNoticeProvider(account.providerId)) return;
  const accounts = await context.adapter.findMany({
    model: "account",
    where: [{ field: "userId", value: account.userId }],
    limit: 2,
  });
  if (accounts.length < 2) return;
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

async function withTimeout<T>(promise: Promise<T>, ms: number) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`No answer in ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

function logFailure(message: string, error: unknown) {
  console.error(message, {
    error: error instanceof Error ? error.message : String(error),
  });
}

function isNoticeProvider(
  providerId: string,
): providerId is SignInEmailProvider {
  return NOTICE_PROVIDERS.has(providerId);
}
