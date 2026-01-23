/**
 * Shared constants for the frontend application
 */

// Admin impersonation
export const IMPERSONATION_HEADER_NAME = "X-Act-As-User-Id";
export const IMPERSONATION_STORAGE_KEY = "admin-impersonate-user-id";

// API key authentication
export const API_KEY_HEADER_NAME = "X-API-Key";

// Layout
export const NAVBAR_HEIGHT_PX = 60;

// Routes
export function getHomepageRoute(isChatEnabled?: boolean | null): string {
  if (isChatEnabled === true) return "/copilot";
  if (isChatEnabled === false) return "/library";
  return "/";
}
