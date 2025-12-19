/**
 * Native authentication library for AutoGPT Platform.
 *
 * This library provides a complete authentication solution without Supabase,
 * using FastAPI backend endpoints and JWT tokens.
 *
 * @example Client-side usage:
 * ```tsx
 * import { useAuth } from "@/lib/auth";
 *
 * function MyComponent() {
 *   const { user, isLoggedIn, logOut } = useAuth();
 *   // ...
 * }
 * ```
 *
 * @example Server-side usage:
 * ```tsx
 * // Import server actions directly from their files
 * import { getCurrentUser, serverLogout } from "@/lib/auth/actions";
 *
 * async function ServerComponent() {
 *   const user = await getCurrentUser();
 *   // ...
 * }
 * ```
 *
 * @example Middleware usage:
 * ```tsx
 * // Import middleware utilities directly
 * import { handleAuthMiddleware } from "@/lib/auth/middleware";
 * ```
 */

// Types
export type {
  AuthUser,
  AuthTokens,
  AuthSession,
  LoginCredentials,
  RegisterCredentials,
  AuthError,
  AuthResult,
} from "./types";

// Hooks (client-side)
export { useAuth } from "./hooks/useAuth";
export { useAuthStore } from "./hooks/useAuthStore";

// API client (client-safe, no server-only imports)
export {
  login,
  register,
  logout,
  refreshTokens,
  requestPasswordReset,
  confirmPasswordReset,
  getGoogleLoginUrl,
  decodeJwtPayload,
} from "./api";

// Helpers (client-safe)
export {
  PROTECTED_PAGES,
  ADMIN_PAGES,
  LOGOUT_BROADCAST_KEY,
  LOGIN_BROADCAST_KEY,
  getRedirectPath,
  isProtectedPath,
  isAdminPath,
  broadcastLogout,
  broadcastLogin,
  isAdmin,
  isWaitlistError,
} from "./helpers";

// Constants (client-safe)
export { AUTH_COOKIE_NAMES } from "./constants";

// NOTE: Server actions and middleware are NOT exported from this barrel
// because they use next/headers which is server-only.
// Import them directly when needed:
// - Server actions: import { ... } from "@/lib/auth/actions"
// - Middleware: import { ... } from "@/lib/auth/middleware"
