import type { GenericEndpointContext } from "@better-auth/core";
import type { BetterAuthPlugin } from "better-auth";
import { createAuthEndpoint } from "better-auth/api";
import * as z from "zod";
import { limitAttemptsPerIP } from "./attempt-limits";
import {
  type PendingSignInCode,
  SIGN_IN_CODE_PATHS,
  SIGN_IN_CODE_RESEND_SECONDS,
  SIGN_IN_CODE_TTL_SECONDS,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import { codeMatches, hashCode, newCode } from "./sign-in-code-crypto";
import {
  expiredError,
  forgetPendingCookie,
  setPendingCookie,
} from "./sign-in-code-cookie";
import {
  codeEmailRefused,
  lockedOutError,
  outOfTriesError,
  tooSoon,
  uncheckedError,
  wrongCode,
} from "./sign-in-code-errors";
import { isLockedOut, recordWrongCode } from "./sign-in-code-lockout";
import {
  claimCodeEmail,
  releaseCodeEmail,
  sendCode,
} from "./sign-in-code-emails";
import {
  claimTry,
  consumePending,
  findPending,
  MAX_CODE_ATTEMPTS,
  type Pending,
  replacePending,
  restorePending,
  secondsUntilResend,
} from "./sign-in-code-store";
import { finishSignIn } from "./sign-in-code-finish";

// Documented in docs/platform/rate-limits.md: update it with any change here.
const CODE_CHECKS_PER_IP = { max: 10, windowSeconds: 60 };

/**
 * The endpoints /login uses while a provider sign-in waits on the code
 * emailed to its address (require-sign-in-code.ts starts one).
 */
export function signInCode() {
  return {
    id: "sign-in-code",
    endpoints: {
      signInCodeStatus: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.status,
        { method: "GET" },
        status,
      ),
      signInCodeVerify: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.verify,
        { method: "POST", body: z.object({ code: z.string().max(64) }) },
        (ctx) => verify(ctx, ctx.body.code),
      ),
      signInCodeResend: createAuthEndpoint(
        SIGN_IN_CODE_PATHS.resend,
        { method: "POST" },
        resend,
      ),
    },
  } satisfies BetterAuthPlugin;
}

// Where the code went, for the page that asks for it.
async function status(ctx: GenericEndpointContext) {
  const pending = await findPending(ctx);
  if (!pending) return ctx.json({ pending: null });
  const answer: PendingSignInCode = {
    provider: pending.record.providerId,
    email: pending.record.email,
    resendInSeconds: secondsUntilResend(pending.record),
  };
  return ctx.json({ pending: answer });
}

async function verify(ctx: GenericEndpointContext, typed: string) {
  await limitAttemptsPerIP({
    name: "sign-in-code",
    limit: CODE_CHECKS_PER_IP,
    headers: ctx.request?.headers ?? ctx.headers,
    context: ctx.context,
  });
  const pending = await findPending(ctx);
  if (!pending) throw expiredError(ctx);
  const attempt = await claimTry(ctx.context, pending);
  const code = typed.replace(/\s+/g, "");
  const { id, record } = pending;
  if (
    attempt !== null &&
    codeMatches(ctx.context.secret, id, code, record.codeHash)
  ) {
    // One use: of two requests with the right code, one finishes.
    if (!(await consumePending(ctx.context, pending))) {
      throw await afterLostVersion(ctx);
    }
    forgetPendingCookie(ctx);
    return finishSignIn(ctx, record);
  }
  if (attempt !== null) await countWrongCode(ctx, pending);
  if (attempt !== null && attempt < MAX_CODE_ATTEMPTS - 1) throw wrongCode();
  // The code's last try: the sign-in ends, unless a new code replaced this one.
  if (await consumePending(ctx.context, pending)) throw outOfTriesError(ctx);
  throw await afterLostVersion(ctx);
}

// Past the address's limit the sign-in ends, and no code goes to the address
// for a while. It ends too when the wrong code can't be counted, so an outage
// can't let guesses past the limit. Either way, if a new code replaced this
// one meanwhile, the sign-in goes on with the new code.
async function countWrongCode(ctx: GenericEndpointContext, pending: Pending) {
  const counted = await recordWrongCode(ctx.context, pending.record.email);
  if (counted === "counted") return;
  if (!(await consumePending(ctx.context, pending))) {
    throw await afterLostVersion(ctx);
  }
  throw counted === "locked" ? lockedOutError(ctx) : uncheckedError(ctx);
}

// A newer code replaced the one this check read, or another request used it
// up: the sign-in goes on with the newer code, if there is one.
async function afterLostVersion(ctx: GenericEndpointContext) {
  return (await findPending(ctx)) ? wrongCode() : expiredError(ctx);
}

async function resend(ctx: GenericEndpointContext) {
  const pending = await findPending(ctx);
  if (!pending) throw expiredError(ctx);
  const { id, record } = pending;
  const wait = secondsUntilResend(record);
  if (wait > 0) throw tooSoon(wait);
  if (await isLockedOut(ctx.context, record.email)) throw lockedOutError(ctx);
  const claim = await claimCodeEmail(ctx.context, record.email);
  if ("refusal" in claim) throw codeEmailRefused(claim.refusal);
  const code = newCode();
  const sentAt = Date.now();
  const codeHash = hashCode(ctx.context.secret, id, code);
  const renewed = new Date(sentAt + SIGN_IN_CODE_TTL_SECONDS * 1000);
  const replaced = { ...record, codeHash, sentAt };
  const stored = await replacePending(
    ctx.context,
    pending,
    replaced,
    renewed,
  ).catch(async (error: unknown) => {
    console.error("Failed to store a new sign-in code", {
      error: error instanceof Error ? error.message : String(error),
    });
    // Nothing was replaced or sent, so the slot goes back.
    await releaseCodeEmail(ctx.context, claim.slot);
    throw codeEmailRefused(SIGN_IN_CODE_UNSENT);
  });
  if (!stored) {
    await releaseCodeEmail(ctx.context, claim.slot);
    // Another resend got there first and its code is on its way, or the
    // sign-in ran out meanwhile.
    throw (await findPending(ctx))
      ? tooSoon(SIGN_IN_CODE_RESEND_SECONDS)
      : expiredError(ctx);
  }
  if (!(await sendCode(ctx.context, record, code, claim.slot))) {
    // The old code still works, as long as it did, and the email can be asked
    // for again.
    await restorePending(ctx.context, pending, stored);
    throw codeEmailRefused(SIGN_IN_CODE_UNSENT);
  }
  await setPendingCookie(ctx, id);
  return ctx.json({ resendInSeconds: SIGN_IN_CODE_RESEND_SECONDS });
}
