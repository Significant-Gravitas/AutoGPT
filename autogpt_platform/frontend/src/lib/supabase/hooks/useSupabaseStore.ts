"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { SupabaseClient, User } from "@supabase/supabase-js";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
  isProtectedPage,
  setWebSocketDisconnectIntent,
  setupSessionEventListeners,
} from "../helpers";
import {
  ensureSupabaseClient,
  fetchUser,
  handleStorageEvent as handleStorageEventHelper,
  refreshSession as refreshSessionHelper,
  validateSession as validateSessionHelper,
} from "./helpers";

interface InitializeParams {
  api: BackendAPI;
  router: AppRouterInstance;
  pathname: string;
}

interface LogOutParams {
  api?: BackendAPI;
  options?: ServerLogoutOptions;
  router?: AppRouterInstance;
}

interface ValidateParams {
  force?: boolean;
  pathname?: string;
  router?: AppRouterInstance;
}

interface SupabaseStoreState {
  user: User | null;
  supabase: SupabaseClient | null;
  isUserLoading: boolean;
  isValidating: boolean;
  hasLoadedUser: boolean;
  lastValidation: number;
  initializationPromise: Promise<void> | null;
  listenersCleanup: (() => void) | null;
  routerRef: AppRouterInstance | null;
  apiRef: BackendAPI | null;
  currentPathname: string;
  initialize: (params: InitializeParams) => Promise<void>;
  logOut: (params?: LogOutParams) => Promise<void>;
  validateSession: (params?: ValidateParams) => Promise<boolean>;
  refreshSession: () => ReturnType<typeof refreshSessionHelper>;
  cleanup: () => void;
}

export const useSupabaseStore = create<SupabaseStoreState>((set, get) => {
  async function initialize(params: InitializeParams): Promise<void> {
    set({
      routerRef: params.router,
      apiRef: params.api,
      currentPathname: params.pathname,
    });

    const supabaseClient = ensureSupabaseClient();
    if (supabaseClient !== get().supabase) {
      set({ supabase: supabaseClient });
    }

    let initializationPromise = get().initializationPromise;

    if (!initializationPromise) {
      initializationPromise = (async () => {
        // Always fetch user if we haven't loaded it yet, or if user is null but hasLoadedUser is true
        // This handles the case where hasLoadedUser might be stale after logout/login
        if (!get().hasLoadedUser || !get().user) {
          set({ isUserLoading: true });
          const result = await fetchUser();

          // Always update state with fetch result
          set(result);

          // If fetchUser didn't return a user, validate the session to ensure we have the latest state
          // This handles race conditions after login where cookies might not be immediately available
          if (!result.user && !result.hasLoadedUser) {
            // Cookies might not be ready yet, retry validation
            const validationResult = await validateSessionHelper({
              pathname: params.pathname,
              currentUser: null,
            });

            if (validationResult.user && validationResult.isValid) {
              set({
                user: validationResult.user,
                hasLoadedUser: true,
                isUserLoading: false,
              });
            } else if (!validationResult.isValid) {
              // Session is invalid, mark as loaded so we don't keep retrying
              set({
                hasLoadedUser: true,
                isUserLoading: false,
              });
            } else {
              // Validation succeeded but no user - might be cookies not ready
              // If we're on a protected page, schedule a retry since we should have a user
              const isProtected = isProtectedPage(params.pathname);
              if (isProtected && params.router) {
                // Retry after a short delay to allow cookies to propagate
                // Use router.refresh() to trigger a re-initialization
                setTimeout(() => {
                  const currentState = get();
                  if (
                    !currentState.user &&
                    isProtectedPage(currentState.currentPathname)
                  ) {
                    // Trigger router refresh to cause re-initialization
                    params.router.refresh();
                  }
                }, 500);
              }
              // Don't mark as loaded yet, allow retry on next initialization
              set({
                isUserLoading: false,
              });
            }
          } else if (!result.user && result.hasLoadedUser) {
            // Explicit error or already marked as loaded - don't retry
            set({
              isUserLoading: false,
            });
          }
        } else {
          // Even if we have a user, validate session to catch account switches
          // This ensures that if user logged out and logged in with different account,
          // we detect the change immediately
          const currentUser = get().user;
          if (currentUser) {
            const validationResult = await validateSessionHelper({
              pathname: params.pathname,
              currentUser,
            });

            // Update user if IDs differ (account switch detected)
            if (
              validationResult.user &&
              validationResult.isValid &&
              validationResult.user.id !== currentUser.id
            ) {
              set({
                user: validationResult.user,
                hasLoadedUser: true,
                isUserLoading: false,
              });
            } else {
              set({ isUserLoading: false });
            }
          } else {
            set({ isUserLoading: false });
          }
        }

        const existingCleanup = get().listenersCleanup;
        if (existingCleanup) {
          existingCleanup();
        }

        const cleanup = setupSessionEventListeners(
          handleVisibilityChange,
          handleFocus,
          handleStorageEventInternal,
        );
        set({ listenersCleanup: cleanup.cleanup });
      })();

      set({ initializationPromise });
    }

    try {
      await initializationPromise;
    } finally {
      set({ initializationPromise: null });
    }
  }

  async function logOut(params?: LogOutParams): Promise<void> {
    const api = params?.api ?? get().apiRef;
    const options = params?.options ?? {};

    setWebSocketDisconnectIntent();

    if (api) {
      api.disconnectWebSocket();
    }

    const existingCleanup = get().listenersCleanup;
    if (existingCleanup) {
      existingCleanup();
      set({ listenersCleanup: null });
    }

    broadcastLogout();

    // Clear React Query cache to prevent stale data from old user
    if (typeof window !== "undefined") {
      const { getQueryClient } = await import("@/lib/react-query/queryClient");
      const queryClient = getQueryClient();
      queryClient.clear();
    }

    // Reset all state to ensure fresh initialization on next login
    set({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
      initializationPromise: null, // Force fresh initialization
      lastValidation: 0, // Reset validation timestamp
    });

    await serverLogout(options);
  }

  async function validateSessionInternal(
    params?: ValidateParams,
  ): Promise<boolean> {
    const router = params?.router ?? get().routerRef;
    const pathname = params?.pathname ?? get().currentPathname;

    if (!router || !pathname) return true;
    if (!params?.force && get().isValidating) return true;

    const now = Date.now();
    if (!params?.force && now - get().lastValidation < 2000) return true;

    set({
      isValidating: true,
      lastValidation: now,
    });

    try {
      const result = await validateSessionHelper({
        pathname,
        currentUser: get().user,
      });

      if (!result.isValid) {
        set({
          user: null,
          hasLoadedUser: false,
          isUserLoading: false,
        });

        if (result.redirectPath) {
          router.push(result.redirectPath);
        }

        return false;
      }

      // Always update user if:
      // 1. We got a user and current user is null (login scenario)
      // 2. We got a user and IDs differ (account switch scenario)
      // 3. shouldUpdateUser is true (session validation detected change)
      if (result.user) {
        const currentUser = get().user;
        const shouldUpdate =
          !currentUser ||
          currentUser.id !== result.user.id ||
          result.shouldUpdateUser;

        if (shouldUpdate) {
          // Invalidate profile query when user changes to ensure fresh data
          if (
            typeof window !== "undefined" &&
            currentUser?.id !== result.user.id
          ) {
            const { getQueryClient } = await import(
              "@/lib/react-query/queryClient"
            );
            const { getGetV2GetUserProfileQueryKey } = await import(
              "@/app/api/__generated__/endpoints/store/store"
            );
            const queryClient = getQueryClient();
            queryClient.invalidateQueries({
              queryKey: getGetV2GetUserProfileQueryKey(),
            });
          }

          set({
            user: result.user,
            hasLoadedUser: true,
            isUserLoading: false,
          });
        } else {
          // Even if user didn't change, ensure loading state is cleared
          set({
            hasLoadedUser: true,
            isUserLoading: false,
          });
        }
      }

      return true;
    } finally {
      set({ isValidating: false });
    }
  }

  function handleVisibilityChange(): void {
    if (document.visibilityState !== "visible") return;
    void validateSessionInternal();
  }

  function handleFocus(): void {
    void validateSessionInternal();
  }

  function handleStorageEventInternal(event: StorageEvent): void {
    const result = handleStorageEventHelper({
      event,
      api: get().apiRef,
      router: get().routerRef,
      pathname: get().currentPathname,
    });

    if (!result.shouldLogout) return;

    set({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });

    const router = get().routerRef;
    if (router) {
      router.refresh();
      if (result.redirectPath) {
        router.push(result.redirectPath);
      }
    }
  }

  async function refreshSessionInternal() {
    const result = await refreshSessionHelper();

    if (result.user) {
      set({
        user: result.user,
        hasLoadedUser: true,
        isUserLoading: false,
      });
    } else if (result.error) {
      set({
        user: null,
        hasLoadedUser: false,
        isUserLoading: false,
      });
    }

    return result;
  }

  function cleanup(): void {
    const existingCleanup = get().listenersCleanup;
    if (existingCleanup) {
      existingCleanup();
      set({ listenersCleanup: null });
    }
  }

  return {
    user: null,
    supabase: null,
    isUserLoading: true,
    isValidating: false,
    hasLoadedUser: false,
    lastValidation: 0,
    initializationPromise: null,
    listenersCleanup: null,
    routerRef: null,
    apiRef: null,
    currentPathname: "",
    initialize,
    logOut,
    validateSession: validateSessionInternal,
    refreshSession: refreshSessionInternal,
    cleanup,
  };
});
