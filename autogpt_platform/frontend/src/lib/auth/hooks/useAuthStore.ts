"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import { resetAnalyticsIdentity } from "@/services/analytics/reset-identity";
import {
  clearImpersonationEverywhere,
  ImpersonationState,
  isImpersonationClearedEvent,
  isImpersonationUsableBy,
} from "@/lib/impersonation";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
  setupSessionEventListeners,
  setWebSocketDisconnectIntent,
} from "../helpers";
import type { User } from "../types";
import {
  fetchUser,
  handleStorageEvent as handleStorageEventHelper,
  refreshSession as refreshSessionHelper,
  validateSession as validateSessionHelper,
} from "./helpers";

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
          const { user, ...result } = await fetchUser();
          set(result);
          if (user) adoptUser(user);
          else set({ user: null });

          // If fetchUser didn't return a user, validate the session to ensure we have the latest state
          // This handles race conditions after login where cookies might not be immediately available
          if (!user) {
            const validationResult = await validateSessionHelper({
              path: params.path,
              currentUser: null,
            });

            if (validationResult.user && validationResult.isValid) {
              adoptUser(validationResult.user);
              set({ hasLoadedUser: true, isUserLoading: false });
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

    // Drop any admin impersonation (sessionStorage + cookie, in every tab) so
    // the next user signing in on this browser doesn't send X-Act-As-User-Id.
    clearImpersonationEverywhere();

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

      if (!result.isValid) {
        clearImpersonationEverywhere();
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
        adoptUser(result.user);
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

  function handleVisibilityChange(): void {
    if (document.visibilityState !== "visible") return;
    void validateSessionInternal();
  }

  // An impersonation target belongs to the admin session that chose it. When
  // this tab takes on a different account, or one that may not use it, the
  // target goes. That includes a tab starting from an empty store, which can
  // hold a target from a session that ended while it wasn't listening.
  function adoptUser(user: User): void {
    const previous = get().user;
    const switchedAccount = previous !== null && previous.id !== user.id;
    if (
      (switchedAccount || !isImpersonationUsableBy(user)) &&
      ImpersonationState.get()
    ) {
      clearImpersonationEverywhere();
    }
    set({ user });
  }

  function handleStorageEventInternal(event: StorageEvent): void {
    if (isImpersonationClearedEvent(event)) {
      ImpersonationState.clear();
      return;
    }

    const result = handleStorageEventHelper({
      event,
      api: get().apiRef,
      router: get().routerRef,
      path: get().currentPath,
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
      adoptUser(result.user);
      set({ hasLoadedUser: true, isUserLoading: false });
    } else if (result.error) {
      clearImpersonationEverywhere();
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
    isUserLoading: true,
    isValidating: false,
    hasLoadedUser: false,
    lastValidation: 0,
    initializationPromise: null,
    listenersCleanup: null,
    routerRef: null,
    apiRef: null,
    currentPath: "",
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
