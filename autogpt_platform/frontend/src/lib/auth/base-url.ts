import { SECURE_COOKIE_PREFIX } from "better-auth/cookies";

// Shared by auth.ts and the edge middleware, so no server-only imports here:
// the middleware reads the cookies auth.ts sets, whose names follow this URL.
export function getAuthBaseURL(): string {
  return (
    process.env.BETTER_AUTH_URL ||
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
    "http://localhost:3000"
  );
}

// The prefix of Better Auth's cookie names. auth.ts sets it, and the
// middleware builds the session cookie's and its cache's names from it.
export const AUTH_COOKIE_PREFIX = "better-auth";

// The session cookie's name as Better Auth gives it: Secure cookies carry
// the `__Secure-` prefix. admin-gate-flow.test.ts pins it to the name the real
// server sets.
export function sessionTokenCookieName(isSecure: boolean) {
  return `${isSecure ? SECURE_COOKIE_PREFIX : ""}${AUTH_COOKIE_PREFIX}.session_token`;
}

// Whether Better Auth's cookies are Secure, which also prefixes their names
// with `__Secure-`: only over https, as a browser drops a Secure cookie that
// reaches it over plain http.
export function usesSecureCookies(baseURL: string = getAuthBaseURL()) {
  try {
    return new URL(baseURL).protocol === "https:";
  } catch {
    return false;
  }
}
