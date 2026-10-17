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
// Other tabs only hear a storage write that changes the value, so each
// signal is unique even when two land in the same millisecond.
export function uniqueSignal(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// Full reload, so the page shows (and requests act as) what changed in
// another tab rather than what this tab started with.
export function reloadPage(): void {
  window.location.reload();
}

export function broadcastLogout(): void {
  storage.set(Key.LOGOUT, uniqueSignal());
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
