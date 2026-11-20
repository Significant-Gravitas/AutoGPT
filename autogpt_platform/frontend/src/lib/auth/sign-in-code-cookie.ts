import type { GenericEndpointContext } from "@better-auth/core";
import { APIError } from "better-auth/api";
import { expireCookie } from "better-auth/cookies";
import { SIGN_IN_CODE_EXPIRED, SIGN_IN_CODE_TTL_SECONDS } from "./sign-in-code";

// The signed cookie that ties a waiting sign-in to the browser that started
// it: it holds the sign-in's id (sign-in-code-store.ts).
const COOKIE_NAME = "sign_in_code";

export function pendingCookieName(ctx: GenericEndpointContext) {
  return pendingCookie(ctx).name;
}

export async function setPendingCookie(
  ctx: GenericEndpointContext,
  id: string,
) {
  const cookie = pendingCookie(ctx);
  await ctx.setSignedCookie(
    cookie.name,
    id,
    ctx.context.secret,
    cookie.attributes,
  );
}

export function forgetPendingCookie(ctx: GenericEndpointContext) {
  expireCookie(ctx, pendingCookie(ctx));
}

/** The refusal when nothing waits on a code here; it drops the cookie too. */
export function expiredError(ctx: GenericEndpointContext) {
  forgetPendingCookie(ctx);
  return new APIError("BAD_REQUEST", {
    code: SIGN_IN_CODE_EXPIRED,
    message: "This sign-in has expired. Start again.",
  });
}

function pendingCookie(ctx: GenericEndpointContext) {
  return ctx.context.createAuthCookie(COOKIE_NAME, {
    maxAge: SIGN_IN_CODE_TTL_SECONDS,
  });
}
