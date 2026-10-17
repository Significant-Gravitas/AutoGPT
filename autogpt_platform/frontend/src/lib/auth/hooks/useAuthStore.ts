"use client";

import type BackendAPI from "@/lib/autogpt-server-api/client";
import { resetAnalyticsIdentity } from "@/services/analytics/reset-identity";
import {
  clearImpersonationEverywhere,
  ImpersonationState,
  isImpersonationChangedEvent,
  isImpersonationClearedEvent,
  isImpersonationUsableBy,
  hasImpersonation,
  setImpersonationViewer,
} from "@/lib/impersonation";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { create } from "zustand";
import { serverLogout, type ServerLogoutOptions } from "../actions";
import {
  broadcastLogout,
  reloadPage,
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

const SIGN_OUT_ATTEMPTS = 3;
const SIGN_OUT_RETRY_DELAY_MS = 1_000;

// The sign-out action itself never throws once delivered (it expires the
// cookies even if revoking the session fails), so a throw means the request
// didn't get through. Retry those before giving up.
async function refetchEverything(): Promise<void> {
  try {
    const { getQueryClient } = await import("@/lib/react-query/queryClient");
    await getQueryClient().invalidateQueries();
  } catch (error) {
    console.debug("Couldn't refetch after impersonation resumed:", error);
  }
}

async function signOutOnServer(options: ServerLogoutOptions): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    try {
      await serverLogout(options);
      return;
    } catch (error) {
      if (attempt >= SIGN_OUT_ATTEMPTS) throw error;
      await new Promise((resolve) =>
        setTimeout(resolve, SIGN_OUT_RETRY_DELAY_MS * attempt),
      );
    }
  }
}

async function afterSignOut(
  step: string,
  run: () => void | Promise<void>,
): Promise<void> {
  try {
    await run();
  } catch (error) {
    console.error(`Signed out, but couldn't ${step}:`, error);
  }
}

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
  // Bumped whenever the signed-in identity ends or changes (sign-out, a
  // rejected session, another account or role). An auth request started in
  // an earlier epoch has its answer dropped rather than applied.
  sessionEpoch: number;
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
        // Listen first: a sign-out in another tab while this tab's first
        // lookup is in flight must still reach it (and void that lookup).
        const existingCleanup = get().listenersCleanup;
        if (existingCleanup) {
          existingCleanup();
        }

        const cleanup = setupSessionEventListeners(
          handleVisibilityChange,
          handleStorageEventInternal,
        );
        set({ listenersCleanup: cleanup.cleanup });

        // Always fetch user if we haven't loaded it yet, or if user is null but hasLoadedUser is true
        // This handles the case where hasLoadedUser might be stale after logout/login
        if (!get().hasLoadedUser || !get().user) {
          set({ isUserLoading: true });
          // A check already running for this session answers for start-up
          // too; only look again if it didn't find anyone.
          while (checkRunning()) await currentCheck?.done;
          if (get().user) {
            set({ isUserLoading: false });
            return;
          }
          const finishCheck = beginCheck();
          try {
            const epoch = get().sessionEpoch;
            const { user, ...result } = await fetchUser();
            if (get().sessionEpoch !== epoch) return;
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
              if (get().sessionEpoch !== epoch) return;

              if (validationResult.user && validationResult.isValid) {
                adoptUser(validationResult.user);
                set({ hasLoadedUser: true, isUserLoading: false });
              } else if (!validationResult.isValid) {
                // No session at startup: a target left from the last one goes,
                // and a protected page sends the user to sign in (the
                // middleware only checks that a session cookie exists, which an
                // expired or revoked one still does).
                if (hasImpersonation()) {
                  endSessionEpoch();
                  clearImpersonationEverywhere();
                }
                // Replace, like the middleware's redirect: the page the
                // dead session landed on isn't somewhere to come back to.
                if (validationResult.redirectPath) {
                  params.router.replace(validationResult.redirectPath);
                }
              }
            }
          } finally {
            finishCheck();
          }
        } else {
          set({ isUserLoading: false });
        }
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

    // Whatever the outcome, answers to auth requests already in flight
    // belong to the session being signed out of, and must not be applied.
    endSessionEpoch();

    // Drop any admin impersonation (sessionStorage + cookie, in every tab) so
    // the next user signing in on this browser doesn't send X-Act-As-User-Id.
    clearImpersonationEverywhere();

    // Only a sign-out the server received ends the session. If the request
    // never got through, the cookies are still live, so this tab stays signed
    // in and no other tab is told otherwise; the caller reports the failure.
    await signOutOnServer(options);
    // Anything answered while the sign-out was in flight described the
    // session that has now ended, including a target set meanwhile.
    endSessionEpoch();
    clearImpersonationEverywhere();

    // The server has ended the session, so the user is signed out whatever
    // happens next: a local cleanup step that fails is logged and skipped.
    await afterSignOut("disconnect the WebSocket", () => {
      setWebSocketDisconnectIntent();
      api?.disconnectWebSocket();
    });
    // The session listeners stay: this tab must keep hearing other tabs'
    // sign-outs and impersonation clears after it signs in again.
    // Clear React Query cache to prevent stale data from old user
    await afterSignOut("clear cached data", async () => {
      if (typeof window === "undefined") return;
      const { getQueryClient } = await import("@/lib/react-query/queryClient");
      getQueryClient().clear();
    });

    set({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });

    // Other tabs act on this at once. Sent only now that the server has
    // expired the session cookies, or a tab could reload into the old session.
    broadcastLogout();
  }

  async function validateSessionInternal(
    params?: ValidateParams,
  ): Promise<boolean> {
    const router = params?.router ?? get().routerRef;
    const pathname = params?.path ?? get().currentPath;

    if (!router || !pathname) return true;
    // Even a forced check waits its turn: one that lands mid-check would
    // race it and act on a stale answer.
    if (checkRunning()) return true;

    const now = Date.now();
    if (!params?.force && now - get().lastValidation < 2000) return true;

    set({ lastValidation: now });
    const finishCheck = beginCheck();

    const epoch = get().sessionEpoch;
    try {
      const result = await validateSessionHelper({
        path: pathname,
        currentUser: get().user,
      });
      if (get().sessionEpoch !== epoch) return true;

      if (!result.isValid) {
        endSessionEpoch();
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
      finishCheck();
    }
  }

  function handleVisibilityChange(): void {
    if (document.visibilityState !== "visible") return;
    void validateSessionInternal();
  }

  function endSessionEpoch(): void {
    set({ sessionEpoch: get().sessionEpoch + 1 });
  }

  // One session check at a time per epoch: start-up's lookups and every
  // validation go through here. A check from an earlier epoch doesn't block
  // a new one, since its answer will be dropped anyway.
  let currentCheck: { epoch: number; done: Promise<void> } | null = null;

  function checkRunning(): boolean {
    return currentCheck !== null && currentCheck.epoch === get().sessionEpoch;
  }

  function beginCheck(): () => void {
    let finished!: () => void;
    const check = {
      epoch: get().sessionEpoch,
      done: new Promise<void>((resolve) => {
        finished = resolve;
      }),
    };
    currentCheck = check;
    set({ isValidating: true });
    return () => {
      if (currentCheck === check) {
        currentCheck = null;
        set({ isValidating: false });
      }
      finished();
    };
  }

  // An impersonation target belongs to the admin session that chose it. When
  // this tab takes on a different account, or one that may not use it, the
  // target goes. That includes a tab starting from an empty store, which can
  // hold a target from a session that ended while it wasn't listening.
  function adoptUser(user: User): void {
    const previous = get().user;
    const switchedAccount = previous !== null && previous.id !== user.id;
    if (switchedAccount || (previous !== null && previous.role !== user.role)) {
      endSessionEpoch();
    }
    if (
      hasImpersonation() &&
      (switchedAccount || !isImpersonationUsableBy(user))
    ) {
      clearImpersonationEverywhere();
    }
    set({ user });
    // Requests sent before this tab knew who was signed in went out without
    // the admin's target; fetch them again as the user they're acting as.
    if (previous === null && isImpersonationUsableBy(user)) {
      void refetchEverything();
    }
  }
  function handleStorageEventInternal(event: StorageEvent): void {
    if (isImpersonationClearedEvent(event)) {
      ImpersonationState.clear();
      return;
    }
    if (isImpersonationChangedEvent(event)) {
      // Another tab started impersonating someone else: this tab's page
      // belongs to the old choice. Only the owning admin's tabs follow it.
      const user = get().user;
      if (user && isImpersonationUsableBy(user)) reloadPage();
      return;
    }

    const result = handleStorageEventHelper({
      event,
      api: get().apiRef,
      router: get().routerRef,
      path: get().currentPath,
    });

    if (!result.shouldLogout) return;

    endSessionEpoch();
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
    // A refresh is asked for explicitly, so it isn't skipped like a
    // background check: it waits its turn, then holds the one-at-a-time slot.
    while (checkRunning()) await currentCheck?.done;
    const finishCheck = beginCheck();
    try {
      return await refreshOnce();
    } finally {
      finishCheck();
    }
  }

  async function refreshOnce() {
    const epoch = get().sessionEpoch;
    const result = await refreshSessionHelper();
    if (get().sessionEpoch !== epoch) return result;

    if (result.user) {
      adoptUser(result.user);
      set({ hasLoadedUser: true, isUserLoading: false });
    } else if (result.error) {
      endSessionEpoch();
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
    sessionEpoch: 0,
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

// Headers are built only for the user this store has signed in, so keep the
// impersonation module told who that is.
useAuthStore.subscribe((state, previousState) => {
  if (state.user !== previousState.user) setImpersonationViewer(state.user);
});
