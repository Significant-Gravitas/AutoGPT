"use server";
import { getServerAuthToken } from "@/lib/autogpt-server-api/helpers";
import * as Sentry from "@sentry/nextjs";
import { cookies, headers } from "next/headers";
import { auth } from "./auth";
import { getRedirectPath } from "./helpers";
import { getServerSession } from "./server/getServerSession";
import { mapSessionUser, type User } from "./types";

export interface SessionValidationResult {
  user: User | null;
  isValid: boolean;
  redirectPath?: string;
}

export async function validateSession(
  currentPath: string,
): Promise<SessionValidationResult> {
  return await Sentry.withServerActionInstrumentation(
    "validateSession",
    {},
    async () => {
      try {
        const session = await getServerSession();

        if (!session?.user) {
          const redirectPath = getRedirectPath(currentPath);
          return {
            user: null,
            isValid: false,
            redirectPath: redirectPath || undefined,
          };
        }

        return {
          user: mapSessionUser(session.user),
          isValid: true,
        };
      } catch (error) {
        console.error("Session validation error:", error);
        const redirectPath = getRedirectPath(currentPath);
        return {
          user: null,
          isValid: false,
          redirectPath: redirectPath || undefined,
        };
      }
    },
  );
}

export async function getCurrentUser(): Promise<{
  user: User | null;
  error?: string;
}> {
  return await Sentry.withServerActionInstrumentation(
    "getCurrentUser",
    {},
    async () => {
      try {
        const session = await getServerSession();
        return { user: session?.user ? mapSessionUser(session.user) : null };
      } catch (error) {
        console.error("Get current user error:", error);
        return {
          user: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export async function getWebSocketToken(): Promise<{
  token: string | null;
  error?: string;
}> {
  return await Sentry.withServerActionInstrumentation(
    "getWebSocketToken",
    {},
    async () => {
      try {
        const token = await getServerAuthToken();
        return { token };
      } catch (error) {
        console.error("Get WebSocket token error:", error);
        return {
          token: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export type ServerLogoutOptions = {
  globalLogout?: boolean;
};

export async function serverLogout(_options: ServerLogoutOptions = {}) {
  return await Sentry.withServerActionInstrumentation(
    "serverLogout",
    {},
    async () => {
      try {
        // Better Auth's signOut revokes the current session and clears the
        // cookie; other devices' sessions stay valid (matching the previous
        // default "local" scope).
        await auth.api.signOut({ headers: await headers() });
      } catch (error) {
        console.error("Logout error:", error);
      }

      // Always expire the auth cookies, even when signOut() fails (revoked or
      // already-rotated session token, auth API hiccup). If they survive, the
      // middleware still sees a session and bounces the user from /login
      // straight back into the app with a half-dead session: the paywall
      // re-mounts while every API call fails with a 401.
      await clearAuthCookies();

      // No `revalidatePath`: `/` is a client-only spinner that redirects to
      // `/copilot`, so there is no RSC payload to invalidate. The cross-tab
      // storage listener calls `router.refresh()` for any visible page, and
      // the React Query cache is cleared client-side.
      return { success: true };
    },
  );
}

// Better Auth stores the session in `better-auth.session_token` /
// `better-auth.session_data` cookies (prefixed `__Secure-` over HTTPS).
// Also sweep any leftover `sb-`-prefixed Supabase cookies so a user mid-
// migration is fully signed out.
async function clearAuthCookies() {
  const cookieStore = await cookies();
  for (const cookie of cookieStore.getAll()) {
    if (
      cookie.name.startsWith("better-auth.") ||
      cookie.name.startsWith("__Secure-better-auth.") ||
      cookie.name.startsWith("sb-")
    ) {
      cookieStore.delete(cookie.name);
    }
  }
}

export async function refreshSession() {
  return await Sentry.withServerActionInstrumentation(
    "refreshSession",
    {},
    async () => {
      try {
        // Cookie sessions don't need an explicit token refresh; re-reading
        // the session is enough to confirm (and slide) it.
        const session = await getServerSession();

        if (!session?.user) {
          return { user: null, error: "No active session" };
        }

        return { user: mapSessionUser(session.user) };
      } catch (error) {
        console.error("Refresh session error:", error);
        return {
          user: null,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}
