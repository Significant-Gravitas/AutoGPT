"use server";

/**
 * Server actions for authentication.
 *
 * These actions are called from client components and handle
 * server-side authentication operations with httpOnly cookies.
 */

import { redirect } from "next/navigation";
import * as Sentry from "@sentry/nextjs";

import * as authApi from "./api";
import {
  clearAuthCookies,
  getAccessToken,
  getRefreshToken,
  setAuthCookies,
} from "./cookies";
import type {
  AuthResult,
  AuthUser,
  LoginCredentials,
  RegisterCredentials,
} from "./types";
import { getRedirectPath, PROTECTED_PAGES } from "./helpers";

/**
 * Validate the current session and return user info.
 * Used by middleware and protected pages.
 */
export async function validateSession(
  currentPath?: string,
): Promise<{ user: AuthUser | null; error: string | null; redirect?: string }> {
  return Sentry.withServerActionInstrumentation(
    "validateSession",
    {},
    async () => {
      const accessToken = await getAccessToken();

      if (!accessToken) {
        // Try to refresh using refresh token
        const refreshToken = await getRefreshToken();

        if (!refreshToken) {
          // No tokens - user needs to log in
          if (
            currentPath &&
            PROTECTED_PAGES.some((page) => currentPath.startsWith(page))
          ) {
            return {
              user: null,
              error: "Not authenticated",
              redirect: getRedirectPath(currentPath),
            };
          }
          return { user: null, error: null };
        }

        // Try to refresh the tokens
        const refreshResult = await authApi.refreshTokens(refreshToken);

        if (refreshResult.error || !refreshResult.data) {
          await clearAuthCookies();
          return {
            user: null,
            error: "Session expired",
            redirect: "/login",
          };
        }

        // Set new tokens
        await setAuthCookies(
          refreshResult.data.access_token,
          refreshResult.data.refresh_token,
          refreshResult.data.expires_in,
        );

        // Decode user from new token
        const user = authApi.decodeJwtPayload(refreshResult.data.access_token);
        return { user, error: null };
      }

      // Decode user from existing token
      const user = authApi.decodeJwtPayload(accessToken);

      if (!user) {
        // Token is invalid - try refresh
        const refreshToken = await getRefreshToken();

        if (!refreshToken) {
          await clearAuthCookies();
          return { user: null, error: "Invalid session", redirect: "/login" };
        }

        const refreshResult = await authApi.refreshTokens(refreshToken);

        if (refreshResult.error || !refreshResult.data) {
          await clearAuthCookies();
          return { user: null, error: "Session expired", redirect: "/login" };
        }

        await setAuthCookies(
          refreshResult.data.access_token,
          refreshResult.data.refresh_token,
          refreshResult.data.expires_in,
        );

        return {
          user: authApi.decodeJwtPayload(refreshResult.data.access_token),
          error: null,
        };
      }

      return { user, error: null };
    },
  );
}

/**
 * Get the current user without path validation.
 */
export async function getCurrentUser(): Promise<AuthUser | null> {
  return Sentry.withServerActionInstrumentation(
    "getCurrentUser",
    {},
    async () => {
      const accessToken = await getAccessToken();

      if (!accessToken) {
        return null;
      }

      return authApi.decodeJwtPayload(accessToken);
    },
  );
}

/**
 * Get the WebSocket token (access token) for authenticated connections.
 */
export async function getWebSocketToken(): Promise<string | null> {
  return Sentry.withServerActionInstrumentation(
    "getWebSocketToken",
    {},
    async () => {
      const accessToken = await getAccessToken();

      if (!accessToken) {
        // Try to refresh
        const refreshToken = await getRefreshToken();

        if (!refreshToken) {
          return null;
        }

        const refreshResult = await authApi.refreshTokens(refreshToken);

        if (refreshResult.error || !refreshResult.data) {
          return null;
        }

        await setAuthCookies(
          refreshResult.data.access_token,
          refreshResult.data.refresh_token,
          refreshResult.data.expires_in,
        );

        return refreshResult.data.access_token;
      }

      return accessToken;
    },
  );
}

/**
 * Login with email and password.
 */
export async function serverLogin(
  credentials: LoginCredentials,
): Promise<AuthResult<AuthUser>> {
  return Sentry.withServerActionInstrumentation("serverLogin", {}, async () => {
    const result = await authApi.login(credentials);

    if (result.error || !result.data) {
      return { data: null, error: result.error };
    }

    // Set cookies
    await setAuthCookies(
      result.data.access_token,
      result.data.refresh_token,
      result.data.expires_in,
    );

    // Decode user from token
    const user = authApi.decodeJwtPayload(result.data.access_token);

    return { data: user, error: null };
  });
}

/**
 * Register a new user.
 */
export async function serverRegister(
  credentials: RegisterCredentials,
): Promise<AuthResult<AuthUser>> {
  return Sentry.withServerActionInstrumentation(
    "serverRegister",
    {},
    async () => {
      const result = await authApi.register(credentials);

      if (result.error || !result.data) {
        return { data: null, error: result.error };
      }

      // Set cookies
      await setAuthCookies(
        result.data.access_token,
        result.data.refresh_token,
        result.data.expires_in,
      );

      // Decode user from token
      const user = authApi.decodeJwtPayload(result.data.access_token);

      return { data: user, error: null };
    },
  );
}

/**
 * Logout the current user.
 */
export async function serverLogout(options?: {
  global?: boolean;
  redirectTo?: string;
}): Promise<{ success: boolean; error: string | null }> {
  return Sentry.withServerActionInstrumentation(
    "serverLogout",
    {},
    async () => {
      const refreshToken = await getRefreshToken();

      if (refreshToken) {
        // Revoke the refresh token on the server
        await authApi.logout(refreshToken);
      }

      // Clear cookies
      await clearAuthCookies();

      // Handle redirect if specified
      if (options?.redirectTo) {
        redirect(options.redirectTo);
      }

      return { success: true, error: null };
    },
  );
}

/**
 * Refresh the current session.
 */
export async function refreshSession(): Promise<{
  user: AuthUser | null;
  error: string | null;
}> {
  return Sentry.withServerActionInstrumentation(
    "refreshSession",
    {},
    async () => {
      const refreshToken = await getRefreshToken();

      if (!refreshToken) {
        return { user: null, error: "No refresh token" };
      }

      const result = await authApi.refreshTokens(refreshToken);

      if (result.error || !result.data) {
        await clearAuthCookies();
        return { user: null, error: result.error?.message || "Refresh failed" };
      }

      await setAuthCookies(
        result.data.access_token,
        result.data.refresh_token,
        result.data.expires_in,
      );

      const user = authApi.decodeJwtPayload(result.data.access_token);

      return { user, error: null };
    },
  );
}
