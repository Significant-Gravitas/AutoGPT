import { APIError } from "better-auth/api";
import {
  SIGN_IN_CODE_INVALID,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_TOO_SOON,
  SIGN_IN_CODE_UNSENT,
} from "./sign-in-code";

// The code endpoints' refusals (sign-in-code-plugin.ts), by the code /login
// reads from them. The expired one also drops the cookie: sign-in-code-cookie.ts.

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
