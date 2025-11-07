"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { SupabaseClient, User } from "@supabase/supabase-js";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
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
        if (!get().hasLoadedUser) {
          set({ isUserLoading: true });
          const result = await fetchUser();
          set(result);
        } else {
          set({ isUserLoading: false });
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
    const router = params?.router ?? get().routerRef;
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

    try {
      await serverLogout(options);
    } catch (error) {
      console.error("Error logging out:", error);
    } finally {
      set({
        user: null,
        hasLoadedUser: false,
        isUserLoading: false,
      });

      if (router) {
        router.refresh();
      }
    }
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
