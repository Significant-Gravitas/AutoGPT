/**
 * Shared constants for the frontend application
 */

// Admin impersonation
export const IMPERSONATION_HEADER_NAME = "X-Act-As-User-Id";
// Intentionally identical strings: the cookie and sessionStorage key share the same name
// so both storage mechanisms use a predictable, consistent identifier for the same value.
export const IMPERSONATION_STORAGE_KEY = "admin-impersonate-user-id";
export const IMPERSONATION_COOKIE_NAME = "admin-impersonate-user-id";

// API key authentication
export const API_KEY_HEADER_NAME = "X-API-Key";

// Layout
export const NAVBAR_HEIGHT_PX = 60;
