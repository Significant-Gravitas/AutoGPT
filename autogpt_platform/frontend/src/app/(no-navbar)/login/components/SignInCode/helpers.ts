import {
  getSocialSignInError,
  isWaitlistSignInError,
} from "@/lib/auth/microsoft-login";
import {
  isSignInProvider,
  type PendingSignInCode,
  SIGN_IN_CODE_EXPIRED,
  SIGN_IN_CODE_INVALID,
  SIGN_IN_CODE_LIMIT,
  SIGN_IN_CODE_LOCKED,
  SIGN_IN_CODE_OUT_OF_TRIES,
  SIGN_IN_CODE_PATHS,
  SIGN_IN_CODE_TOO_SOON,
  SIGN_IN_CODE_UNCHECKED,
  SIGN_IN_CODE_UNSENT,
} from "@/lib/auth/sign-in-code";
import type { LoginProvider } from "@/types/auth";

// Focused when the screen replaces the form: on a phone that opens the number
// keyboard, which offers the emailed code.
export const SIGN_IN_CODE_INPUT_ID = "sign-in-code";

export const SIGN_IN_CODE_COPY = {
  invalid: "That code isn't right. Check the email and try again.",
  tooSoon: "Wait a moment before asking for another code.",
  limit:
    "We've already sent several codes to this address. Try again in an hour.",
  unsent: "We couldn't send a new code. Try again in a moment.",
  rateLimited: "Too many attempts. Wait a moment and try again.",
  failed: "That didn't work. Try again in a moment.",
} as const;

// The answers after which the sign-in is over, each with its own screen.
const END_CODES = [
  SIGN_IN_CODE_EXPIRED,
  SIGN_IN_CODE_OUT_OF_TRIES,
  SIGN_IN_CODE_LOCKED,
  SIGN_IN_CODE_UNCHECKED,
] as const;
export type SignInCodeEnd = (typeof END_CODES)[number];

export type SignInCodeOutcome =
  | { kind: "signed-in"; provider: LoginProvider }
  | { kind: "invalid" }
  | { kind: "ended"; end: SignInCodeEnd }
  | { kind: "waitlisted" }
  | { kind: "refused"; message: string };

export type ResendOutcome =
  | { kind: "sent"; resendInSeconds: number }
  | { kind: "ended"; end: SignInCodeEnd }
  | { kind: "failed"; message: string };

interface ErrorBody {
  code?: unknown;
}

const BASE = "/api/auth";

export async function fetchPendingSignInCode() {
  const response = await fetch(`${BASE}${SIGN_IN_CODE_PATHS.status}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Could not load the sign-in");
  const { pending } = (await response.json()) as {
    pending: PendingSignInCode | null;
  };
  return pending;
}

export async function submitSignInCode(
  code: string,
): Promise<SignInCodeOutcome> {
  const response = await post(SIGN_IN_CODE_PATHS.verify, { code });
  const body = (await response.json().catch(() => ({}))) as ErrorBody & {
    provider?: unknown;
  };
  if (response.ok) {
    return isSignInProvider(body.provider)
      ? { kind: "signed-in", provider: body.provider }
      : { kind: "refused", message: SIGN_IN_CODE_COPY.failed };
  }
  const errorCode = typeof body.code === "string" ? body.code : null;
  if (errorCode === SIGN_IN_CODE_INVALID) return { kind: "invalid" };
  if (isEnd(errorCode)) return { kind: "ended", end: errorCode };
  if (response.status === 429) {
    return { kind: "refused", message: SIGN_IN_CODE_COPY.rateLimited };
  }
  if (isWaitlistSignInError(errorCode)) return { kind: "waitlisted" };
  // A sign-up or sign-in check refused the account the code finished:
  // the same codes a provider sign-in brings back to /login.
  return {
    kind: "refused",
    message: getSocialSignInError(errorCode) ?? SIGN_IN_CODE_COPY.failed,
  };
}

export async function resendSignInCode(): Promise<ResendOutcome> {
  const response = await post(SIGN_IN_CODE_PATHS.resend, {});
  const body = (await response.json().catch(() => ({}))) as ErrorBody & {
    resendInSeconds?: unknown;
  };
  if (response.ok) {
    return {
      kind: "sent",
      resendInSeconds: Number(body.resendInSeconds) || 0,
    };
  }
  if (isEnd(body.code)) return { kind: "ended", end: body.code };
  switch (body.code) {
    case SIGN_IN_CODE_TOO_SOON:
      return { kind: "failed", message: SIGN_IN_CODE_COPY.tooSoon };
    case SIGN_IN_CODE_LIMIT:
      return { kind: "failed", message: SIGN_IN_CODE_COPY.limit };
    case SIGN_IN_CODE_UNSENT:
      return { kind: "failed", message: SIGN_IN_CODE_COPY.unsent };
    default:
      return {
        kind: "failed",
        message:
          response.status === 429
            ? SIGN_IN_CODE_COPY.rateLimited
            : SIGN_IN_CODE_COPY.failed,
      };
  }
}

function isEnd(code: unknown): code is SignInCodeEnd {
  return END_CODES.some((end) => end === code);
}

function post(path: string, body: object) {
  return fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
