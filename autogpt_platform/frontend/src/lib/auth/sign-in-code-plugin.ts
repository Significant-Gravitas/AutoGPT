import type { GenericEndpointContext } from "@better-auth/core";
import type { BetterAuthPlugin } from "better-auth";
import { APIError, createAuthEndpoint } from "better-auth/api";
import { setSessionCookie } from "better-auth/cookies";
import { handleOAuthUserInfo } from "better-auth/oauth2";
import * as z from "zod";
import { limitAttemptsPerIP } from "./attempt-limits";
import {
  type PendingSignInCode,
  SIGN_IN_CODE_INVALID,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_PATHS,
  SIGN_IN_CODE_RESEND_SECONDS,
  SIGN_IN_CODE_TOO_SOON,
  SIGN_IN_CODE_TTL_SECONDS,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import { codeMatches, hashCode, newCode } from "./sign-in-code-crypto";
import { claimCodeEmail, sendCode } from "./sign-in-code-emails";
import {
  claimTry,
  deletePending,
  expiredError,
  findPending,
  forgetPendingCookie,
  MAX_CODE_ATTEMPTS,
  type PendingRecord,
  savePending,
  secondsUntilResend,
  setPendingCookie,
} from "./sign-in-code-store";
import { findProvider, markConfirmedByCode } from "./sign-in-provider";

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
  const isLastTry = attempt === MAX_CODE_ATTEMPTS - 1;
  const code = typed.replace(/\s+/g, "");
  const { id, rowID, record } = pending;
  if (
    attempt === null ||
    !codeMatches(ctx.context.secret, id, code, record.codeHash)
  ) {
    if (attempt === null || isLastTry) {
      await deletePending(ctx.context, rowID);
      throw expiredError(ctx);
    }
    throw new APIError("BAD_REQUEST", {
      code: SIGN_IN_CODE_INVALID,
      message: "That code isn't right.",
    });
  }
  // One use: of two requests with the right code, one finishes.
  if (!(await deletePending(ctx.context, rowID))) throw expiredError(ctx);
  forgetPendingCookie(ctx);
  return finishSignIn(ctx, record);
}

async function resend(ctx: GenericEndpointContext) {
  const pending = await findPending(ctx);
  if (!pending) throw expiredError(ctx);
  const { id, rowID, record, expiresAt } = pending;
  const wait = secondsUntilResend(record);
  if (wait > 0) {
    throw new APIError(
      "TOO_MANY_REQUESTS",
      {
        code: SIGN_IN_CODE_TOO_SOON,
        message: `Wait ${wait} seconds before asking for a new code.`,
      },
      { "X-Retry-After": String(wait) },
    );
  }
  const claim = await claimCodeEmail(ctx.context, record.email);
  if ("refusal" in claim) {
    const limited = claim.refusal === SIGN_IN_CODE_LIMIT;
    throw new APIError(limited ? "TOO_MANY_REQUESTS" : "SERVICE_UNAVAILABLE", {
      code: claim.refusal,
      message: limited
        ? "Too many codes for this address."
        : "The code couldn't be sent.",
    });
  }
  const code = newCode();
  const sentAt = Date.now();
  const codeHash = hashCode(ctx.context.secret, id, code);
  const renewed = new Date(sentAt + SIGN_IN_CODE_TTL_SECONDS * 1000);
  await savePending(
    ctx.context,
    rowID,
    { ...record, codeHash, sentAt },
    renewed,
  );
  if (!(await sendCode(ctx.context, record, code, claim.slot))) {
    // The old code still works, as long as it did, and the email can be asked
    // for again.
    await savePending(ctx.context, rowID, record, expiresAt);
    throw new APIError("SERVICE_UNAVAILABLE", {
      code: SIGN_IN_CODE_UNSENT,
      message: "The code couldn't be sent.",
    });
  }
  await setPendingCookie(ctx, id);
  return ctx.json({ resendInSeconds: SIGN_IN_CODE_RESEND_SECONDS });
}

// Better Auth's own sign-in for the provider account the callback saw, with
// the address now proven. It links that account to the address's account, or
// creates one, through every database hook, then starts the session. The
// account row gets no provider tokens: the callback's were never kept, and
// nothing reads a sign-in account's tokens (`/get-access-token` is off). The
// provider's next sign-in stores them.
async function finishSignIn(
  ctx: GenericEndpointContext,
  record: PendingRecord,
) {
  markConfirmedByCode(ctx, record.providerId);
  const provider = await findProvider(ctx, record.providerId);
  if (!provider) throw expiredError(ctx);
  const result = await handleOAuthUserInfo(ctx, {
    userInfo: {
      id: record.accountId,
      email: record.email,
      emailVerified: true,
      name: record.name,
      image: record.image,
    },
    account: { providerId: record.providerId, accountId: record.accountId },
    disableSignUp: provider.options?.disableSignUp === true,
    requireExactAccountBinding: true,
    source: { method: "oauth", oauth: { providerId: record.providerId } },
  });
  if (!result.data) {
    // Better Auth's sign-in errors, as its callback sends them to /login.
    throw new APIError("FORBIDDEN", {
      code: result.error.split(" ").join("_"),
      message: "",
    });
  }
  await setSessionCookie(ctx, result.data);
  return ctx.json({
    provider: record.providerId,
    newUser: result.isRegister === true,
  });
}
