import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";

export const PROTECTED_PAGES = [
  "/auth/authorize",
  "/auth/integrations",
  "/copilot",
  "/home",
  "/monitor",
  "/build",
  "/onboarding",
  "/profile",
  "/library",
  "/settings",
  "/team",
] as const;

export const ADMIN_PAGES = ["/admin"] as const;

// Page protection utilities
export function isProtectedPage(pathname: string | null | undefined): boolean {
  if (!pathname) return false;
  return PROTECTED_PAGES.some((page) => pathname.startsWith(page));
}

export function isAdminPage(pathname: string | null | undefined): boolean {
  if (!pathname) return false;
  return ADMIN_PAGES.some((page) => pathname.startsWith(page));
}

export function shouldRedirectOnLogout(
  pathname: string | null | undefined,
): boolean {
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
  path: string | null | undefined, // including query strings
  userRole?: string,
): string | null {
  if (!path) return null;

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
  onStorageChange: (e: StorageEvent) => void,
): EventListeners {
  if (environment.isServerSide()) {
    return { cleanup: () => {} };
  }

  document.addEventListener("visibilitychange", onVisibilityChange);
  window.addEventListener("storage", onStorageChange);

  return {
    cleanup: () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("storage", onStorageChange);
    },
  };
}
