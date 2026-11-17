// A provider sign-in that can't vouch for its address ends with a code
// emailed there, typed into /login (require-sign-in-code.ts). What the page
// and the server share; no server imports.

export const SIGN_IN_CODE_LENGTH = 6;
// Must match the email's "The code expires in 10 minutes" (backend
// notifications/sign_in_email.py).
export const SIGN_IN_CODE_TTL_SECONDS = 10 * 60;
// Documented in docs/platform/rate-limits.md: update it with any change here.
export const SIGN_IN_CODE_RESEND_SECONDS = 60;

// The `error` a provider sign-in comes back to /login with when it waits on
// a code.
export const SIGN_IN_CODE_SENT = "sign_in_code_sent";
// The endpoints' refusals, and the callback's when no code could go out.
export const SIGN_IN_CODE_INVALID = "sign_in_code_invalid";
export const SIGN_IN_CODE_EXPIRED = "sign_in_code_expired";
export const SIGN_IN_CODE_TOO_SOON = "sign_in_code_too_soon";
export const SIGN_IN_CODE_LIMIT = "sign_in_code_limit";
export const SIGN_IN_CODE_UNSENT = "sign_in_code_unsent";
export const SIGN_IN_CODE_LOCKED = "sign_in_code_locked";
// The code's last try was wrong.
export const SIGN_IN_CODE_OUT_OF_TRIES = "sign_in_code_out_of_tries";
// A wrong code that couldn't be counted against its address.
export const SIGN_IN_CODE_UNCHECKED = "sign_in_code_unchecked";

// Under Better Auth's /api/auth.
export const SIGN_IN_CODE_PATHS = {
  status: "/sign-in-code/status",
  verify: "/sign-in-code/verify",
  resend: "/sign-in-code/resend",
} as const;

export interface PendingSignInCode {
  provider: string;
  email: string;
  resendInSeconds: number;
}

/** Copy for a sign-in the callback couldn't start a code for. */
export const SIGN_IN_CODE_START_ERRORS: Record<string, string> = {
  [SIGN_IN_CODE_LIMIT]:
    "We've already sent several codes to this address. Try again in an hour, or log in the way you usually do.",
  [SIGN_IN_CODE_UNSENT]:
    "We couldn't email you a code. Try again in a moment, or log in the way you usually do.",
  [SIGN_IN_CODE_LOCKED]:
    "Too many wrong codes were entered for this address, so we've stopped sending it codes for now. Log in the way you usually do.",
};

// Every "Continue with" provider, by Better Auth's id: the one list the
// code, its emails and /login name providers from.
export const SIGN_IN_PROVIDERS = [
  "google",
  "microsoft",
  "github",
  "discord",
  "apple",
] as const;
export type SignInProvider = (typeof SIGN_IN_PROVIDERS)[number];

const PROVIDER_NAMES: Record<SignInProvider, string> = {
  google: "Google",
  microsoft: "Microsoft",
  github: "GitHub",
  discord: "Discord",
  apple: "Apple",
};

export function isSignInProvider(value: unknown): value is SignInProvider {
  return SIGN_IN_PROVIDERS.some((provider) => provider === value);
}

export function signInProviderName(provider: string) {
  return isSignInProvider(provider) ? PROVIDER_NAMES[provider] : "that account";
}
