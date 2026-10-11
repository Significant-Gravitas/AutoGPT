"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import { resetAnalyticsIdentity } from "@/services/analytics/reset-identity";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
  navigateWithFullLoad,
  reloadForNewDeployment,
  setupSessionEventListeners,
  setWebSocketDisconnectIntent,
} from "../helpers";
import type { User } from "../types";
import {
  fetchUser,
  handleStorageEvent as handleStorageEventHelper,
  refreshSession as refreshSessionHelper,
  validateSession as validateSessionHelper,
  type SessionUnavailableReason,
} from "./helpers";

const RETRY_BASE_DELAY_MS = 1_000;
const RETRY_MAX_DELAY_MS = 30_000;
const MAX_RETRY_ATTEMPTS = 6;

interface InitializeParams {
  api: BackendAPI;
  router: AppRouterInstance;
  path: string;
}

interface LogOutParams {
  api?: BackendAPI;
  options?: ServerLogoutOptions;
  router?: AppRouterInstance;
}

interface ValidateParams {
  force?: boolean;
  path?: string;
  router?: AppRouterInstance;
}

interface AuthStoreState {
  user: User | null;
  isUserLoading: boolean;
  isValidating: boolean;
  hasLoadedUser: boolean;
  lastValidation: number;
  initializationPromise: Promise<void> | null;
  listenersCleanup: (() => void) | null;
  routerRef: AppRouterInstance | null;
  apiRef: BackendAPI | null;
  currentPath: string;
  retryTimer: ReturnType<typeof setTimeout> | null;
  retryAttempt: number;
  initialize: (params: InitializeParams) => Promise<void>;
  setCurrentRequestContext: (params: InitializeParams) => void;
  logOut: (params?: LogOutParams) => Promise<void>;
  validateSession: (params?: ValidateParams) => Promise<boolean>;
  refreshSession: () => ReturnType<typeof refreshSessionHelper>;
  cleanup: () => void;
}

export const useAuthStore = create<AuthStoreState>((set, get) => {
  function setCurrentRequestContext(params: InitializeParams): void {
    const state = get();
    if (
      state.routerRef === params.router &&
      state.apiRef === params.api &&
      state.currentPath === params.path
    ) {
      return;
    }
    set({
      routerRef: params.router,
      apiRef: params.api,
      currentPath: params.path,
    });
  }

  async function initialize(params: InitializeParams): Promise<void> {
    setCurrentRequestContext(params);

    let initializationPromise = get().initializationPromise;

    if (!initializationPromise) {
      initializationPromise = (async () => {
        // Always fetch user if we haven't loaded it yet, or if user is null but hasLoadedUser is true
        // This handles the case where hasLoadedUser might be stale after logout/login
        if (!get().hasLoadedUser || !get().user) {
          set({ isUserLoading: true });
          const result = await fetchUser();
          set(result);

          // If fetchUser didn't return a user, validate the session to ensure we have the latest state
          // This handles race conditions after login where cookies might not be immediately available
          if (!result.user) {
            const validationResult = await validateSessionHelper({
              path: params.path,
              currentUser: null,
            });

            if (validationResult.unavailable) {
              handleUnavailable(validationResult.unavailable);
            } else if (validationResult.user && validationResult.isValid) {
              set({
                user: validationResult.user,
                hasLoadedUser: true,
                isUserLoading: false,
              });
            }
          }
        } else {
          set({ isUserLoading: false });
        }

        const existingCleanup = get().listenersCleanup;
        if (existingCleanup) {
          existingCleanup();
        }

        const cleanup = setupSessionEventListeners(
          handleVisibilityChange,
          handleStorageEventInternal,
          handleOnline,
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
    cancelRetry();

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

    set({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });

    await serverLogout(options);
  }

  async function validateSessionInternal(
    params?: ValidateParams,
  ): Promise<boolean> {
    const router = params?.router ?? get().routerRef;
    const pathname = params?.path ?? get().currentPath;

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
        path: pathname,
        currentUser: get().user,
      });

      if (result.unavailable) {
        handleUnavailable(result.unavailable);
        return true;
      }

      cancelRetry();

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

      if (result.user && result.shouldUpdateUser) {
        set({ user: result.user });
      }

      if (result.user) {
        set({
          hasLoadedUser: true,
          isUserLoading: false,
        });
      }

      return true;
    } finally {
      set({ isValidating: false });
    }
  }

  // Focus and `online` are fresh chances to reach the backend, so each one
  // restores the full retry budget for whatever it runs into.
  function handleVisibilityChange(): void {
    if (document.visibilityState !== "visible") return;
    set({ retryAttempt: 0 });
    void validateSessionInternal();
  }

  function handleOnline(): void {
    cancelRetry();
    void validateSessionInternal({ force: true });
  }

  // A validation that couldn't reach the auth backend says nothing about the
  // session, so the user stays signed in. Retry with backoff; the next focus
  // or `online` event also revalidates.
  function handleUnavailable(reason: SessionUnavailableReason): void {
    if (reason === "deployment-skew" && reloadForNewDeployment()) return;

    const attempt = get().retryAttempt;
    // Once the budget is spent, backoff stops; the next focus or `online`
    // event restores it and tries again.
    if (get().retryTimer || attempt >= MAX_RETRY_ATTEMPTS) return;

    const delay = Math.min(
      RETRY_BASE_DELAY_MS * 2 ** attempt,
      RETRY_MAX_DELAY_MS,
    );
    const retryTimer = setTimeout(() => {
      set({ retryTimer: null });
      void validateSessionInternal({ force: true });
    }, delay);
    set({ retryTimer, retryAttempt: attempt + 1 });
  }

  function cancelRetry(): void {
    const retryTimer = get().retryTimer;
    if (retryTimer) clearTimeout(retryTimer);
    set({ retryTimer: null, retryAttempt: 0 });
  }

  function handleStorageEventInternal(event: StorageEvent): void {
    const result = handleStorageEventHelper({
      event,
      api: get().apiRef,
      router: get().routerRef,
      path: get().currentPath,
    });

    if (!result.shouldLogout) return;

    cancelRetry();
    set({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });

    // Full page load: navigates away from the protected page and drops this
    // tab's cached data for the account that just signed out.
    if (result.redirectPath) {
      navigateWithFullLoad(result.redirectPath);
      return;
    }
    get().routerRef?.refresh();
  }

  async function refreshSessionInternal() {
    const result = await refreshSessionHelper();

    if (result.user) {
      set({
        user: result.user,
        hasLoadedUser: true,
        isUserLoading: false,
      });
    } else if (result.error && !result.isUnavailable) {
      set({
        user: null,
        hasLoadedUser: false,
        isUserLoading: false,
      });
    }

    return result;
  }

  function cleanup(): void {
    cancelRetry();
    const existingCleanup = get().listenersCleanup;
    if (existingCleanup) {
      existingCleanup();
      set({ listenersCleanup: null });
    }
  }

  return {
    user: null,
    isUserLoading: true,
    isValidating: false,
    hasLoadedUser: false,
    lastValidation: 0,
    initializationPromise: null,
    listenersCleanup: null,
    routerRef: null,
    apiRef: null,
    currentPath: "",
    retryTimer: null,
    retryAttempt: 0,
    initialize,
    setCurrentRequestContext,
    logOut,
    validateSession: validateSessionInternal,
    refreshSession: refreshSessionInternal,
    cleanup,
  };
});

useAuthStore.subscribe((state, previousState) => {
  if (previousState.user && previousState.user.id !== state.user?.id) {
    resetAnalyticsIdentity();
  }
});
