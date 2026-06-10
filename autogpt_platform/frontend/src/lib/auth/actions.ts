"use server";
import * as Sentry from "@sentry/nextjs";
import { headers } from "next/headers";
import { auth } from "./auth";
import { getRedirectPath } from "./helpers";
import { getServerSession } from "./server/getServerSession";
import { getBackendAuthToken } from "./server/token";
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
        const token = await getBackendAuthToken();
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
        return {
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }

      // No `revalidatePath`: `/` is a client-only spinner that redirects to
      // `/copilot`, so there is no RSC payload to invalidate. The cross-tab
      // storage listener calls `router.refresh()` for any visible page, and
      // the React Query cache is cleared client-side.
      return { success: true };
    },
  );
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
