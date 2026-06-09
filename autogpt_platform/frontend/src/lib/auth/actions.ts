"use server";

import * as Sentry from "@sentry/nextjs";
import { auth, getServerAuthSession, getServerBackendToken, toLegacyAuthUser } from "@/lib/auth/auth";
import { headers } from "next/headers";
import { getRedirectPath } from "./helpers";

export interface SessionValidationResult {
  user: ReturnType<typeof toLegacyAuthUser> | null;
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
        const session = await getServerAuthSession();

        if (!session) {
          return {
            user: null,
            isValid: false,
            redirectPath: getRedirectPath(currentPath) || undefined,
          };
        }

        const user = toLegacyAuthUser(session.user);
        const redirectPath = getRedirectPath(currentPath, user.role);

        if (redirectPath === "/") {
          return {
            user,
            isValid: false,
            redirectPath,
          };
        }

        return {
          user,
          isValid: true,
        };
      } catch (error) {
        console.error("Session validation error:", error);
        return {
          user: null,
          isValid: false,
          redirectPath: getRedirectPath(currentPath) || undefined,
        };
      }
    },
  );
}

export async function getCurrentUser(): Promise<{
  user: ReturnType<typeof toLegacyAuthUser> | null;
  error?: string;
}> {
  return await Sentry.withServerActionInstrumentation(
    "getCurrentUser",
    {},
    async () => {
      try {
        const session = await getServerAuthSession();
        return { user: session ? toLegacyAuthUser(session.user) : null };
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
        return {
          token: await getServerBackendToken(),
        };
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

export async function serverLogout() {
  return await Sentry.withServerActionInstrumentation(
    "serverLogout",
    {},
    async () => {
      try {
        const result = await auth.api.signOut({
          headers: new Headers(await headers()),
        });
        return { success: result.success };
      } catch (error) {
        console.error("Logout error:", error);
        return {
          success: false,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
  );
}

export async function refreshSession() {
  return await Sentry.withServerActionInstrumentation(
    "refreshSession",
    {},
    async () => {
      try {
        const session = await getServerAuthSession();
        return {
          user: session ? toLegacyAuthUser(session.user) : null,
        };
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
