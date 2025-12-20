"use client";

/**
 * Main authentication hook for components.
 *
 * This hook provides access to authentication state and actions.
 * It's designed to be a drop-in replacement for useSupabase().
 */

import { useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";

import { useAuthStore } from "./useAuthStore";
import type { AuthUser } from "../types";

export interface UseAuthReturn {
  /** Current authenticated user, or null if not logged in */
  user: AuthUser | null;
  /** Whether the user is currently logged in */
  isLoggedIn: boolean;
  /** Whether the initial user loading is in progress */
  isUserLoading: boolean;
  /** Log out the current user */
  logOut: (options?: {
    global?: boolean;
    redirectTo?: string;
  }) => Promise<void>;
  /** Validate the current session */
  validateSession: () => Promise<AuthUser | null>;
  /** Refresh the current session */
  refreshSession: () => Promise<AuthUser | null>;
  /** Get auth token for WebSocket connections */
  getToken: () => Promise<string | null>;
}

/**
 * Hook to access authentication state and actions.
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { user, isLoggedIn, logOut } = useAuth();
 *
 *   if (!isLoggedIn) {
 *     return <div>Please log in</div>;
 *   }
 *
 *   return (
 *     <div>
 *       <p>Hello, {user.email}!</p>
 *       <button onClick={() => logOut({ redirectTo: '/' })}>
 *         Log Out
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useAuth(): UseAuthReturn {
  const router = useRouter();
  const pathname = usePathname();

  const {
    user,
    isUserLoading,
    hasLoadedUser,
    setRouterRef,
    setCurrentPathname,
    initializeListeners,
    cleanupListeners,
    validateUserSession,
    refreshUserSession,
    logOutUser,
    getToken,
  } = useAuthStore();

  // Set up router and pathname refs
  useEffect(() => {
    setRouterRef(router);
    setCurrentPathname(pathname);
  }, [router, pathname, setRouterRef, setCurrentPathname]);

  // Initialize event listeners on mount
  useEffect(() => {
    initializeListeners();

    return () => {
      cleanupListeners();
    };
  }, [initializeListeners, cleanupListeners]);

  // Validate session on initial mount if not already loaded
  useEffect(() => {
    if (!hasLoadedUser) {
      validateUserSession();
    }
  }, [hasLoadedUser, validateUserSession]);

  // Memoized logout function
  const logOut = useCallback(
    async (options?: { global?: boolean; redirectTo?: string }) => {
      await logOutUser(options);
    },
    [logOutUser],
  );

  // Memoized validate function
  const validateSession = useCallback(async () => {
    return validateUserSession(true);
  }, [validateUserSession]);

  // Memoized refresh function
  const refreshSession = useCallback(async () => {
    return refreshUserSession();
  }, [refreshUserSession]);

  return {
    user,
    isLoggedIn: Boolean(user),
    isUserLoading,
    logOut,
    validateSession,
    refreshSession,
    getToken,
  };
}

export default useAuth;
