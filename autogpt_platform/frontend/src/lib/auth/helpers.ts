import { environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";

export const PROTECTED_PAGES = [
  "/auth/authorize",
  "/auth/integrations",
  "/copilot",
  "/monitor",
  "/build",
  "/onboarding",
  "/profile",
  "/library",
  "/settings",
] as const;

export const ADMIN_PAGES = ["/admin"] as const;

export function isProtectedPage(pathname: string): boolean {
  return PROTECTED_PAGES.some((page) => pathname.startsWith(page));
}

export function isAdminPage(pathname: string): boolean {
  return ADMIN_PAGES.some((page) => pathname.startsWith(page));
}

export function shouldRedirectOnLogout(pathname: string): boolean {
  return isProtectedPage(pathname) || isAdminPage(pathname);
}

export function broadcastLogout(): void {
  storage.set(Key.LOGOUT, Date.now().toString());
}

export function isLogoutEvent(event: StorageEvent): boolean {
  return event.key === Key.LOGOUT;
}

export function setWebSocketDisconnectIntent(): void {
  storage.set(Key.WEBSOCKET_DISCONNECT_INTENT, "true");
}

export function clearWebSocketDisconnectIntent(): void {
  storage.clean(Key.WEBSOCKET_DISCONNECT_INTENT);
}

export function getRedirectPath(
  path: string,
  userRole?: string,
): string | null {
  if (shouldRedirectOnLogout(path)) {
    return `/login?next=${encodeURIComponent(path)}`;
  }

  if (isAdminPage(path) && userRole !== "admin") {
    return "/";
  }

  return null;
}

export function setupSessionEventListeners(
  onVisibilityChange: () => void,
  onStorageChange: (e: StorageEvent) => void,
) {
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
