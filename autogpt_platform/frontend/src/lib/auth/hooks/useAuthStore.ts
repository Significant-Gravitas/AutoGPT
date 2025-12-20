"use client";

/**
 * Zustand store for authentication state management.
 *
 * This store manages the client-side authentication state,
 * including user info, loading states, and cross-tab synchronization.
 */

import { create } from "zustand";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";

import type { AuthUser } from "../types";
import {
  broadcastLogout,
  LOGOUT_BROADCAST_KEY,
  LOGIN_BROADCAST_KEY,
} from "../helpers";
import {
  validateSession,
  serverLogout,
  refreshSession,
  getWebSocketToken,
} from "../actions";

// Minimum time between session validations (in ms)
const VALIDATION_COOLDOWN = 30_000; // 30 seconds

interface AuthStore {
  // State
  user: AuthUser | null;
  isUserLoading: boolean;
  isValidating: boolean;
  hasLoadedUser: boolean;
  lastValidation: number;

  // References for cleanup
  listenersCleanup: (() => void) | null;
  routerRef: AppRouterInstance | null;
  currentPathname: string | null;

  // WebSocket disconnect intent
  intendToDisconnect: boolean;

  // Actions
  setUser: (user: AuthUser | null) => void;
  setLoading: (loading: boolean) => void;
  setRouterRef: (router: AppRouterInstance | null) => void;
  setCurrentPathname: (pathname: string | null) => void;
  initializeListeners: () => void;
  cleanupListeners: () => void;
  validateUserSession: (force?: boolean) => Promise<AuthUser | null>;
  refreshUserSession: () => Promise<AuthUser | null>;
  logOutUser: (options?: {
    global?: boolean;
    redirectTo?: string;
  }) => Promise<void>;
  getToken: () => Promise<string | null>;
}

export const useAuthStore = create<AuthStore>((set, get) => ({
  // Initial state
  user: null,
  isUserLoading: true,
  isValidating: false,
  hasLoadedUser: false,
  lastValidation: 0,
  listenersCleanup: null,
  routerRef: null,
  currentPathname: null,
  intendToDisconnect: false,

  // Set user
  setUser: (user) => {
    set({ user, hasLoadedUser: true, isUserLoading: false });
  },

  // Set loading state
  setLoading: (loading) => {
    set({ isUserLoading: loading });
  },

  // Set router reference for navigation
  setRouterRef: (router) => {
    set({ routerRef: router });
  },

  // Set current pathname for redirect logic
  setCurrentPathname: (pathname) => {
    set({ currentPathname: pathname });
  },

  // Initialize event listeners for cross-tab sync and visibility changes
  initializeListeners: () => {
    const state = get();

    // Clean up existing listeners first
    if (state.listenersCleanup) {
      state.listenersCleanup();
    }

    // Handle cross-tab auth events (login/logout)
    const handleStorageChange = (event: StorageEvent) => {
      // Handle logout broadcast
      if (event.key === LOGOUT_BROADCAST_KEY && event.newValue) {
        // Clear the broadcast key in this tab to prevent duplicate handling
        try {
          localStorage.removeItem(LOGOUT_BROADCAST_KEY);
        } catch {
          // Ignore errors when cleaning up
        }

        // Clear user state immediately
        set({
          user: null,
          intendToDisconnect: true,
          hasLoadedUser: false,
        });

        // Clear React Query cache asynchronously
        (async () => {
          try {
            const { getQueryClient } = await import(
              "@/lib/react-query/queryClient"
            );
            const queryClient = getQueryClient();
            queryClient.clear();
          } catch (error) {
            console.error(
              "Failed to clear query cache on cross-tab logout:",
              error,
            );
          }
        })();

        // Always redirect to login on cross-tab logout for security
        // Use window.location as fallback if routerRef not available
        const { routerRef } = get();
        if (routerRef) {
          routerRef.push("/login");
        } else {
          // Fallback to window.location for redirect
          window.location.href = "/login";
        }
      }

      // Handle login broadcast - refresh session to get user data
      if (event.key === LOGIN_BROADCAST_KEY && event.newValue) {
        // Clear the broadcast key in this tab
        try {
          localStorage.removeItem(LOGIN_BROADCAST_KEY);
        } catch {
          // Ignore errors when cleaning up
        }

        // Force refresh session to get the new user data
        get().validateUserSession(true);
      }
    };

    // Refresh session when tab becomes visible
    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        const { user, lastValidation } = get();
        const now = Date.now();

        // Only refresh if user is logged in and cooldown has passed
        if (user && now - lastValidation > VALIDATION_COOLDOWN) {
          get().refreshUserSession();
        }
      }
    };

    // Refresh session when window gains focus
    const handleFocus = () => {
      const { user, lastValidation } = get();
      const now = Date.now();

      if (user && now - lastValidation > VALIDATION_COOLDOWN) {
        get().refreshUserSession();
      }
    };

    // Add listeners
    window.addEventListener("storage", handleStorageChange);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("focus", handleFocus);

    // Store cleanup function
    const cleanup = () => {
      window.removeEventListener("storage", handleStorageChange);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("focus", handleFocus);
    };

    set({ listenersCleanup: cleanup });
  },

  // Clean up event listeners
  cleanupListeners: () => {
    const { listenersCleanup } = get();
    if (listenersCleanup) {
      listenersCleanup();
      set({ listenersCleanup: null });
    }
  },

  // Validate user session (with cooldown)
  validateUserSession: async (force = false) => {
    const state = get();
    const now = Date.now();

    // Skip if already validating or if cooldown hasn't passed (unless forced)
    if (state.isValidating) {
      return state.user;
    }

    if (!force && now - state.lastValidation < VALIDATION_COOLDOWN) {
      return state.user;
    }

    set({ isValidating: true });

    try {
      const result = await validateSession(state.currentPathname ?? undefined);

      set({
        user: result.user,
        lastValidation: now,
        isValidating: false,
        hasLoadedUser: true,
        isUserLoading: false,
      });

      // Handle redirect if needed
      if (result.redirect && state.routerRef) {
        state.routerRef.push(result.redirect);
      }

      return result.user;
    } catch (error) {
      set({
        isValidating: false,
        isUserLoading: false,
        hasLoadedUser: true,
      });
      console.error("Session validation error:", error);
      return null;
    }
  },

  // Refresh user session
  refreshUserSession: async () => {
    try {
      const result = await refreshSession();

      if (result.user) {
        set({
          user: result.user,
          lastValidation: Date.now(),
        });
      }

      return result.user;
    } catch (error) {
      console.error("Session refresh error:", error);
      return null;
    }
  },

  // Log out user
  logOutUser: async (options) => {
    const state = get();

    set({ intendToDisconnect: true });

    try {
      // Broadcast logout to other tabs
      broadcastLogout();

      // Clear React Query cache to prevent stale user data from being visible
      if (typeof window !== "undefined") {
        const { getQueryClient } = await import(
          "@/lib/react-query/queryClient"
        );
        const queryClient = getQueryClient();
        queryClient.clear();
      }

      // Clear local state
      set({
        user: null,
        hasLoadedUser: false,
        lastValidation: 0,
      });

      // Call server logout
      await serverLogout({
        global: options?.global,
        redirectTo: options?.redirectTo,
      });

      // Navigate if redirectTo is specified and we have a router
      if (options?.redirectTo && state.routerRef) {
        state.routerRef.push(options.redirectTo);
      }
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      set({ intendToDisconnect: false });
    }
  },

  // Get auth token for WebSocket connections
  getToken: async () => {
    try {
      return await getWebSocketToken();
    } catch (error) {
      console.error("Error getting WebSocket token:", error);
      return null;
    }
  },
}));
