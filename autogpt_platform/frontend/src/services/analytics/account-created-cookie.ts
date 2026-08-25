export const ACCOUNT_CREATED_COOKIE = "agpt_account_created";
export const ACCOUNT_CREATED_COOKIE_MAX_AGE_SECONDS = 600;

export const SIGNUP_METHODS = ["email", "google"] as const;
export type SignupMethod = (typeof SIGNUP_METHODS)[number];

// The server sets this flag when it provisions a brand-new account (see
// account-created-server.ts); the browser reads it once to report the
// Google Ads sign-up conversion, then clears it so a reload can't report twice.
export function consumeAccountCreatedFlag(): SignupMethod | null {
  if (typeof document === "undefined") return null;

  const prefix = `${ACCOUNT_CREATED_COOKIE}=`;
  const entry = document.cookie
    .split("; ")
    .find((part) => part.startsWith(prefix));
  if (!entry) return null;

  document.cookie = `${ACCOUNT_CREATED_COOKIE}=; Path=/; Max-Age=0`;
  const value = entry.slice(prefix.length);
  return isSignupMethod(value) ? value : null;
}

function isSignupMethod(value: string): value is SignupMethod {
  return SIGNUP_METHODS.includes(value as SignupMethod);
}
