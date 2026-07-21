"use server";
import * as Sentry from "@sentry/nextjs";
import type { User } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { getRedirectPath } from "./helpers";
import { getServerSupabase } from "./server/getServerSupabase";

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
      const supabase = await getServerSupabase();

      if (!supabase) {
        return {
          user: null,
          isValid: false,
          redirectPath: getRedirectPath(currentPath) || undefined,
        };
      }

      try {
        const {
          data: { user },
          error,
        } = await supabase.auth.getUser();

        if (error || !user) {
          const redirectPath = getRedirectPath(currentPath);
          return {
            user: null,
            isValid: false,
            redirectPath: redirectPath || undefined,
          };
        }

        return {
          user,
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
      const supabase = await getServerSupabase();

      if (!supabase) {
        return {
          user: null,
          error: "Supabase client not available",
        };
      }

      try {
        const {
          data: { user },
          error,
        } = await supabase.auth.getUser();

        if (error) {
          return {
            user: null,
            error: error.message,
          };
        }

        return { user };
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
      const supabase = await getServerSupabase();

      if (!supabase) {
        return {
          token: null,
          error: "Supabase client not available",
        };
      }

      try {
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();

        if (error) {
          return {
            token: null,
            error: error.message,
          };
        }

        return { token: session?.access_token || null };
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

export async function serverLogout(options: ServerLogoutOptions = {}) {
  return await Sentry.withServerActionInstrumentation(
    "serverLogout",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        return { success: true };
      }

      try {
        const { error } = await supabase.auth.signOut({
          scope: options.globalLogout ? "global" : "local",
        });

        if (error) {
          console.error("Error logging out:", error);
        }
      } catch (error) {
        console.error("Logout error:", error);
      }

      // Always expire the auth cookies, even when signOut() fails (revoked or
      // already-rotated refresh token, Supabase API hiccup). If they survive,
      // the middleware still sees a session and bounces the user from /login
      // straight back into the app with a half-dead session: the paywall
      // re-mounts while every API call fails with a 401.
      await clearSupabaseAuthCookies();

      // No `revalidatePath`: `/` is a client-only spinner that redirects to
      // `/copilot`, so there is no RSC payload to invalidate. The cross-tab
      // storage listener calls `router.refresh()` for any visible page, and
      // the React Query cache is cleared client-side.
      return { success: true };
    },
  );
}

// Supabase SSR stores the session in `sb-<project-ref>-auth-token` cookies
// (chunked as `.0`, `.1`, ... when large), all `sb-`-prefixed.
async function clearSupabaseAuthCookies() {
  const cookieStore = await cookies();
  for (const cookie of cookieStore.getAll()) {
    if (cookie.name.startsWith("sb-")) {
      cookieStore.delete(cookie.name);
    }
  }
}

export async function refreshSession() {
  return await Sentry.withServerActionInstrumentation(
    "refreshSession",
    {},
    async () => {
      const supabase = await getServerSupabase();

      if (!supabase) {
        return {
          user: null,
          error: "Supabase client not available",
        };
      }

      try {
        const {
          data: { user },
          error,
        } = await supabase.auth.refreshSession();

        if (error) {
          return {
            user: null,
            error: error.message,
          };
        }

        return { user };
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
