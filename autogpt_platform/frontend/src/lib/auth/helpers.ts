/**
 * Helper utilities for authentication.
 */

/**
 * List of protected pages that require authentication.
 */
export const PROTECTED_PAGES = [
  "/monitor",
  "/build",
  "/onboarding",
  "/profile",
  "/library",
  "/monitoring",
] as const;

/**
 * List of admin-only pages.
 */
export const ADMIN_PAGES = ["/admin"] as const;

/**
 * Storage key for cross-tab logout broadcasting.
 */
export const LOGOUT_BROADCAST_KEY = "auth_logout_broadcast";

/**
 * Storage key for cross-tab login broadcasting.
 */
export const LOGIN_BROADCAST_KEY = "auth_login_broadcast";

/**
 * Get the redirect path based on the current path.
 */
export function getRedirectPath(currentPath: string): string {
  // Encode the current path for redirect after login
  const encodedPath = encodeURIComponent(currentPath);
  return `/login?redirect=${encodedPath}`;
}

/**
 * Check if a path requires authentication.
 */
export function isProtectedPath(path: string): boolean {
  return PROTECTED_PAGES.some((protectedPath) =>
    path.startsWith(protectedPath),
  );
}

/**
 * Check if a path requires admin access.
 */
export function isAdminPath(path: string): boolean {
  return ADMIN_PAGES.some((adminPath) => path.startsWith(adminPath));
}

/**
 * Broadcast logout to other tabs.
 * Sets a timestamp in localStorage which triggers a storage event in other tabs.
 * The value is intentionally NOT removed immediately to ensure the event fires.
 */
export function broadcastLogout(): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(LOGOUT_BROADCAST_KEY, Date.now().toString());
  }
}

/**
 * Broadcast login to other tabs.
 * Sets a timestamp in localStorage which triggers a storage event in other tabs.
 */
export function broadcastLogin(): void {
  if (typeof window !== "undefined") {
    localStorage.setItem(LOGIN_BROADCAST_KEY, Date.now().toString());
  }
}

/**
 * Check if a user has admin role.
 */
export function isAdmin(role: string): boolean {
  return role === "admin" || role === "service_role";
}

/**
 * Check if an error is a "not on waitlist" error.
 */
export function isWaitlistError(error: {
  message?: string;
  code?: string;
}): boolean {
  // Check for PostgreSQL error code P0001 or specific message
  return Boolean(
    error.code === "P0001" ||
      error.message?.toLowerCase().includes("waitlist") ||
      error.message?.toLowerCase().includes("allowlist") ||
      error.message?.toLowerCase().includes("not allowed"),
  );
}
