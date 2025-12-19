/**
 * Cookie management for authentication tokens.
 *
 * Tokens are stored in httpOnly cookies for security:
 * - access_token: Short-lived JWT for API calls
 * - refresh_token: Long-lived token for session refresh
 *
 * NOTE: This file can only be imported from server-side code
 * because it uses next/headers. For the cookie constants only,
 * import from "./constants" instead.
 */

import "server-only";

import { cookies } from "next/headers";

import { AUTH_COOKIE_NAMES } from "./constants";

export { AUTH_COOKIE_NAMES };

const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
};

/**
 * Set authentication tokens in cookies.
 * Only call this from server-side code.
 */
export async function setAuthCookies(
  accessToken: string,
  refreshToken: string,
  expiresIn: number,
): Promise<void> {
  const cookieStore = await cookies();

  // Access token expires in `expiresIn` seconds
  cookieStore.set(AUTH_COOKIE_NAMES.ACCESS_TOKEN, accessToken, {
    ...COOKIE_OPTIONS,
    maxAge: expiresIn,
  });

  // Refresh token expires in 7 days (or configure via env)
  const refreshTokenMaxAge = 7 * 24 * 60 * 60; // 7 days in seconds
  cookieStore.set(AUTH_COOKIE_NAMES.REFRESH_TOKEN, refreshToken, {
    ...COOKIE_OPTIONS,
    maxAge: refreshTokenMaxAge,
  });
}

/**
 * Get access token from cookies.
 * Only call this from server-side code.
 */
export async function getAccessToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get(AUTH_COOKIE_NAMES.ACCESS_TOKEN)?.value ?? null;
}

/**
 * Get refresh token from cookies.
 * Only call this from server-side code.
 */
export async function getRefreshToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get(AUTH_COOKIE_NAMES.REFRESH_TOKEN)?.value ?? null;
}

/**
 * Clear all authentication cookies.
 * Only call this from server-side code.
 */
export async function clearAuthCookies(): Promise<void> {
  const cookieStore = await cookies();

  cookieStore.delete(AUTH_COOKIE_NAMES.ACCESS_TOKEN);
  cookieStore.delete(AUTH_COOKIE_NAMES.REFRESH_TOKEN);
}

/**
 * Check if authentication cookies exist.
 * Only call this from server-side code.
 */
export async function hasAuthCookies(): Promise<boolean> {
  const cookieStore = await cookies();

  const accessToken = cookieStore.get(AUTH_COOKIE_NAMES.ACCESS_TOKEN);
  const refreshToken = cookieStore.get(AUTH_COOKIE_NAMES.REFRESH_TOKEN);

  return Boolean(accessToken || refreshToken);
}
