import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";
import { type CookieOptions } from "@supabase/ssr";
import { SupabaseClient } from "@supabase/supabase-js";

export const PROTECTED_PAGES = [
  "/auth/authorize",
  "/auth/integrations",
  "/monitor",
  "/build",
  "/onboarding",
  "/profile",
  "/library",
  "/monitoring",
] as const;

export const ADMIN_PAGES = ["/admin"] as const;

export function getCookieSettings(): Partial<CookieOptions> {
  return {
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    httpOnly: true,
  } as const;
}

// Page protection utilities
export function isProtectedPage(pathname: string): boolean {
  return PROTECTED_PAGES.some((page) => pathname.startsWith(page));
}

export function isAdminPage(pathname: string): boolean {
  return ADMIN_PAGES.some((page) => pathname.startsWith(page));
}

export function shouldRedirectOnLogout(pathname: string): boolean {
  return isProtectedPage(pathname) || isAdminPage(pathname);
}

// Cross-tab logout utilities
export function broadcastLogout(): void {
  storage.set(Key.LOGOUT, Date.now().toString());
}

export function isLogoutEvent(event: StorageEvent): boolean {
  return event.key === Key.LOGOUT;
}

// WebSocket disconnect intent utilities
export function setWebSocketDisconnectIntent(): void {
  storage.set(Key.WEBSOCKET_DISCONNECT_INTENT, "true");
}

export function clearWebSocketDisconnectIntent(): void {
  storage.clean(Key.WEBSOCKET_DISCONNECT_INTENT);
}

export function hasWebSocketDisconnectIntent(): boolean {
  return storage.get(Key.WEBSOCKET_DISCONNECT_INTENT) === "true";
}

// Redirect utilities
export function getRedirectPath(
  path: string, // including query strings
  userRole?: string,
): string | null {
  if (shouldRedirectOnLogout(path)) {
    // Preserve the original path as a 'next' parameter so user can return after login
    return `/login?next=${encodeURIComponent(path)}`;
  }

  if (isAdminPage(path) && userRole !== "admin") {
    return "/";
  }

  return null;
}

// Event listener management
export interface EventListeners {
  cleanup: () => void;
}

export function setupSessionEventListeners(
  onVisibilityChange: () => void,
  onFocus: () => void,
  onStorageChange: (e: StorageEvent) => void,
): EventListeners {
  if (environment.isServerSide()) {
    return { cleanup: () => {} };
  }

  document.addEventListener("visibilitychange", onVisibilityChange);
  window.addEventListener("focus", onFocus);
  window.addEventListener("storage", onStorageChange);

  return {
    cleanup: () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("storage", onStorageChange);
    },
  };
}

export interface CodeExchangeResult {
  success: boolean;
  error?: string;
}

export async function exchangePasswordResetCode(
  supabase: SupabaseClient<any, "public", any>,
  code: string,
): Promise<CodeExchangeResult> {
  try {
    const { data, error } = await supabase.auth.exchangeCodeForSession(code);

    if (error) {
      return {
        success: false,
        error: error.message,
      };
    }

    if (!data.session) {
      return {
        success: false,
        error: "Failed to create session",
      };
    }

    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}
