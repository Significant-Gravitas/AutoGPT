import type { GenericEndpointContext } from "@better-auth/core";
import { APIError } from "better-auth/api";
import {
  SIGN_IN_CODE_INVALID,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_LOCKED,
  SIGN_IN_CODE_OUT_OF_TRIES,
  SIGN_IN_CODE_TOO_SOON,
  SIGN_IN_CODE_UNCHECKED,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";
import { forgetPendingCookie } from "./sign-in-code-cookie";

// The code endpoints' refusals (sign-in-code-plugin.ts), by the code /login
// reads from them. Those that end the sign-in also drop its cookie; the
// expired one is in sign-in-code-cookie.ts.

export function wrongCode() {
  return new APIError("BAD_REQUEST", {
    code: SIGN_IN_CODE_INVALID,
    message: "That code isn't right.",
  });
}

export function tooSoon(wait: number) {
  return new APIError(
    "TOO_MANY_REQUESTS",
    {
      code: SIGN_IN_CODE_TOO_SOON,
      message: `Wait ${wait} seconds before asking for a new code.`,
    },
    { "X-Retry-After": String(wait) },
  );
}

export function codeEmailRefused(
  refusal: typeof SIGN_IN_CODE_LIMIT | typeof SIGN_IN_CODE_UNSENT,
) {
  const limited = refusal === SIGN_IN_CODE_LIMIT;
  return new APIError(limited ? "TOO_MANY_REQUESTS" : "SERVICE_UNAVAILABLE", {
    code: refusal,
    message: limited
      ? "Too many codes for this address."
      : "The code couldn't be sent.",
  });
}

/** The address has had too many wrong codes for now; the sign-in ends. */
export function lockedOutError(ctx: GenericEndpointContext) {
  forgetPendingCookie(ctx);
  return new APIError("TOO_MANY_REQUESTS", {
    code: SIGN_IN_CODE_LOCKED,
    message: "Too many wrong codes for this address.",
  });
}

/** The code's last try was wrong; the sign-in ends. */
export function outOfTriesError(ctx: GenericEndpointContext) {
  forgetPendingCookie(ctx);
  return new APIError("BAD_REQUEST", {
    code: SIGN_IN_CODE_OUT_OF_TRIES,
    message: "Too many wrong codes. Start again.",
  });
}

/** A wrong code that couldn't be counted; the sign-in ends. */
export function uncheckedError(ctx: GenericEndpointContext) {
  forgetPendingCookie(ctx);
  return new APIError("SERVICE_UNAVAILABLE", {
    code: SIGN_IN_CODE_UNCHECKED,
    message: "The code couldn't be checked. Start again.",
  });
}
