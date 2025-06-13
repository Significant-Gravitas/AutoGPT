// Session management constants and utilities

export const PROTECTED_PAGES = [
  "/monitor",
  "/build",
  "/onboarding",
  "/profile",
  "/library",
  "/monitoring",
] as const;

export const ADMIN_PAGES = ["/admin"] as const;

export const STORAGE_KEYS = {
  LOGOUT: "supabase-logout",
} as const;

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
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEYS.LOGOUT, Date.now().toString());
  }
}

export function isLogoutEvent(event: StorageEvent): boolean {
  return event.key === STORAGE_KEYS.LOGOUT;
}

// Redirect utilities
export function getRedirectPath(
  pathname: string,
  userRole?: string,
): string | null {
  if (shouldRedirectOnLogout(pathname)) {
    return "/login";
  }

  if (isAdminPage(pathname) && userRole !== "admin") {
    return "/marketplace";
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
  if (typeof window === "undefined" || typeof document === "undefined") {
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
