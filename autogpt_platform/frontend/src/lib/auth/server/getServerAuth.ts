/**
 * Server-side authentication utilities.
 *
 * Use these utilities in server components and API routes
 * to access authentication state.
 */

import "server-only";

import { getAccessToken, getRefreshToken } from "../cookies";
import * as authApi from "../api";
import type { AuthUser } from "../types";

/**
 * Get the authentication token for server-side API requests.
 * Attempts to refresh if the access token is missing but refresh token exists.
 */
export async function getServerAuthToken(): Promise<string | null> {
  const accessToken = await getAccessToken();

  if (!accessToken) {
    const refreshToken = await getRefreshToken();

    if (!refreshToken) {
      return null;
    }

    // Note: In a real implementation, you'd want to update the cookies here
    // but that requires the response object. For now, just return null
    // and let the client-side handle the refresh.
    return null;
  }

  return accessToken;
}

/**
 * Get the current user from the access token.
 * Does not validate the token - use for display purposes only.
 */
export async function getServerUser(): Promise<AuthUser | null> {
  const accessToken = await getAccessToken();

  if (!accessToken) {
    return null;
  }

  return authApi.decodeJwtPayload(accessToken);
}

/**
 * Check if the user is authenticated.
 */
export async function isAuthenticated(): Promise<boolean> {
  const accessToken = await getAccessToken();
  const refreshToken = await getRefreshToken();

  return Boolean(accessToken || refreshToken);
}

/**
 * Check if the user has admin role.
 */
export async function isAdmin(): Promise<boolean> {
  const user = await getServerUser();

  if (!user) {
    return false;
  }

  return user.role === "admin" || user.role === "service_role";
}
