import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";
import { SessionKey, sessionStorage } from "@/services/storage/session-storage";

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
  onStorageChange: (e: StorageEvent) => void,
  onOnline?: () => void,
): EventListeners {
  if (environment.isServerSide()) {
    return { cleanup: () => {} };
  }

  document.addEventListener("visibilitychange", onVisibilityChange);
  window.addEventListener("storage", onStorageChange);
  if (onOnline) window.addEventListener("online", onOnline);

  return {
    cleanup: () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("storage", onStorageChange);
      if (onOnline) window.removeEventListener("online", onOnline);
    },
  };
}

const DEPLOYMENT_RELOAD_COOLDOWN_MS = 60_000;

/**
 * Reloads the page so it picks up the deployment the server is now running.
 * Returns false without reloading if it already did so within the cooldown,
 * so a server that keeps rejecting this bundle can't trap the tab in a loop.
 */
export function reloadForNewDeployment(): boolean {
  if (environment.isServerSide()) return false;

  const lastReloadAt = Number(
    sessionStorage.get(SessionKey.DEPLOYMENT_RELOAD_AT) ?? 0,
  );
  if (Date.now() - lastReloadAt < DEPLOYMENT_RELOAD_COOLDOWN_MS) return false;

  sessionStorage.set(SessionKey.DEPLOYMENT_RELOAD_AT, Date.now().toString());
  window.location.reload();
  return true;
}
