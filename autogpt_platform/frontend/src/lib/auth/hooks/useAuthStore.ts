"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import { resetQueryClientForIdentityChange } from "@/lib/react-query/queryClient";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
  clearWebSocketDisconnectIntent,
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
  hasCompletedInitialAuthHydration: boolean;
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

let authRequestGeneration = 0;
let validationRequestGeneration = 0;
let pendingLogoutPromise: ReturnType<typeof serverLogout> | null = null;

function beginAuthRequest(): number {
  authRequestGeneration += 1;
  return authRequestGeneration;
}

function beginNonValidationAuthRequest(): number {
  validationRequestGeneration += 1;
  return beginAuthRequest();
}

async function waitForPendingLogout(): Promise<boolean> {
  while (pendingLogoutPromise) {
    const pendingLogout = pendingLogoutPromise;
    try {
      await pendingLogout;
    } catch {
      return false;
    }
    if (pendingLogoutPromise === pendingLogout) {
      pendingLogoutPromise = null;
    }
  }
  return true;
}

export const useAuthStore = create<AuthStoreState>((set, get) => {
  function stopLoadingIfCurrent(requestGeneration: number): void {
    if (requestGeneration === authRequestGeneration) {
      set({ isUserLoading: false });
    }
  }

  function clearAuthenticatedState(): void {
    const state = get();
    const identityChangeWillResetQueries =
      state.user !== null && state.hasCompletedInitialAuthHydration;
    set({
      user: null,
      hasLoadedUser: false,
      hasCompletedInitialAuthHydration: true,
      isUserLoading: false,
      isValidating: false,
      initializationPromise: null,
    });
    if (!identityChangeWillResetQueries) {
      void resetQueryClientForIdentityChange(null);
    }
  }

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
        const existingCleanup = get().listenersCleanup;
        if (existingCleanup) {
          existingCleanup();
        }

        const cleanup = setupSessionEventListeners(
          handleVisibilityChange,
          handleStorageEventInternal,
        );
        set({ listenersCleanup: cleanup.cleanup });

        if (get().hasLoadedUser && get().user) {
          set({ isUserLoading: false });
          return;
        }

        const requestGeneration = beginNonValidationAuthRequest();
        set({ isUserLoading: true, isValidating: false });
        if (pendingLogoutPromise && !(await waitForPendingLogout())) {
          stopLoadingIfCurrent(requestGeneration);
          return;
        }
        if (requestGeneration !== authRequestGeneration) return;

        // Always fetch user if we haven't loaded it yet, or if user is null but hasLoadedUser is true
        // This handles the case where hasLoadedUser might be stale after logout/login
        const result = await fetchUser();
        if (requestGeneration !== authRequestGeneration) return;
        if (result.user) clearWebSocketDisconnectIntent();
        set(result);

        // If fetchUser didn't return a user, validate the session to ensure we have the latest state
        // This handles race conditions after login where cookies might not be immediately available
        if (!result.user) {
          const validationResult = await validateSessionHelper({
            path: params.path,
            currentUser: null,
          });
          if (requestGeneration !== authRequestGeneration) return;

          if (validationResult.user && validationResult.isValid) {
            clearWebSocketDisconnectIntent();
            set({
              user: validationResult.user,
              hasLoadedUser: true,
              isUserLoading: false,
            });
          }
        }
      })();

      set({ initializationPromise });
    }

    try {
      await initializationPromise;
    } finally {
      if (get().initializationPromise === initializationPromise) {
        set({
          initializationPromise: null,
          hasCompletedInitialAuthHydration: true,
        });
      }
    }
  }

  async function logOut(params?: LogOutParams): Promise<void> {
    beginNonValidationAuthRequest();
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

    clearAuthenticatedState();

    const logoutPromise = serverLogout(options);
    pendingLogoutPromise = logoutPromise;
    try {
      await logoutPromise;
      broadcastLogout();
    } finally {
      if (pendingLogoutPromise === logoutPromise) {
        pendingLogoutPromise = null;
      }
    }
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

    const requestGeneration = beginAuthRequest();
    const validationGeneration = ++validationRequestGeneration;
    set({
      isValidating: true,
      lastValidation: now,
    });

    try {
      if (pendingLogoutPromise && !(await waitForPendingLogout())) {
        stopLoadingIfCurrent(requestGeneration);
        return false;
      }
      if (requestGeneration !== authRequestGeneration) return false;
      const result = await validateSessionHelper({
        path: pathname,
        currentUser: get().user,
      });
      if (requestGeneration !== authRequestGeneration) return false;
      if (result.user) clearWebSocketDisconnectIntent();

      if (!result.isValid) {
        clearAuthenticatedState();

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
      if (validationGeneration === validationRequestGeneration) {
        set({ isValidating: false });
      }
    }
  }

  function handleVisibilityChange(): void {
    if (document.visibilityState !== "visible") return;
    void validateSessionInternal();
  }

  function handleStorageEventInternal(event: StorageEvent): void {
    const result = handleStorageEventHelper({
      event,
      api: get().apiRef,
      router: get().routerRef,
      path: get().currentPath,
    });

    if (!result.shouldLogout) return;

    beginNonValidationAuthRequest();
    clearAuthenticatedState();

    const router = get().routerRef;
    if (router) {
      router.refresh();
      if (result.redirectPath) {
        router.push(result.redirectPath);
      }
    }
  }

  async function refreshSessionInternal() {
    const requestGeneration = beginNonValidationAuthRequest();
    set({ isValidating: false });
    if (pendingLogoutPromise && !(await waitForPendingLogout())) {
      stopLoadingIfCurrent(requestGeneration);
      return {};
    }
    if (requestGeneration !== authRequestGeneration) return {};
    const result = await refreshSessionHelper();
    if (requestGeneration !== authRequestGeneration) return {};

    if (result.user) {
      clearWebSocketDisconnectIntent();
      set({
        user: result.user,
        hasLoadedUser: true,
        isUserLoading: false,
      });
    } else if (result.error) {
      clearAuthenticatedState();
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
    hasCompletedInitialAuthHydration: false,
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

if (typeof window !== "undefined") {
  useAuthStore.subscribe((state, previousState) => {
    const previousIdentityKey = previousState.user?.id ?? null;
    const identityKey = state.user?.id ?? null;
    if (previousIdentityKey === identityKey) return;
    if (!previousState.hasCompletedInitialAuthHydration) return;
    void resetQueryClientForIdentityChange(identityKey);
  });
}
