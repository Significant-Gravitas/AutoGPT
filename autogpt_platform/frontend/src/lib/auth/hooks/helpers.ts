import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import {
  getCurrentUser,
  refreshSession as refreshSessionAction,
  validateSession as validateSessionAction,
} from "../actions";
import {
  clearWebSocketDisconnectIntent,
  getRedirectPath,
  isLogoutEvent,
  setWebSocketDisconnectIntent,
} from "../helpers";
import type { User } from "../types";
import { ImpersonationState } from "@/lib/impersonation";

interface FetchUserResult {
  user: User | null;
  hasLoadedUser: boolean;
  isUserLoading: boolean;
}

export async function fetchUser(): Promise<FetchUserResult> {
  try {
    const { user, error } = await getCurrentUser();

    if (error || !user) {
      // Only mark as loaded if we got an explicit error (not just no user)
      // This allows retrying when cookies aren't ready yet after login
      return {
        user: null,
        hasLoadedUser: !!error, // Only true if there was an error, not just no user
        isUserLoading: false,
      };
    }

    clearWebSocketDisconnectIntent();
    return {
      user,
      hasLoadedUser: true,
      isUserLoading: false,
    };
  } catch (error) {
    console.error("Get user error:", error);
    return {
      user: null,
      hasLoadedUser: true, // Error means we tried and failed, so mark as loaded
      isUserLoading: false,
    };
  }
}

// Next's `unstable_isUnrecognizedActionError` uses `instanceof`; the name check
// also holds when the error crosses a module or bundle boundary.
function isUnrecognizedActionError(error: unknown): boolean {
  return error instanceof Error && error.name === "UnrecognizedActionError";
}

interface ValidateSessionParams {
  path: string;
  currentUser: User | null;
}

// Why a validation couldn't reach a verdict. "deployment-skew" means the
// server no longer knows this bundle's server action IDs (a deploy happened
// while the tab was open); everything else is a network or server hiccup.
export type SessionUnavailableReason = "network" | "deployment-skew";

interface ValidateSessionResult {
  isValid: boolean;
  user?: User | null;
  redirectPath?: string | null;
  shouldUpdateUser: boolean;
  unavailable?: SessionUnavailableReason;
}

export async function validateSession(
  params: ValidateSessionParams,
): Promise<ValidateSessionResult> {
  try {
    const result = await validateSessionAction(params.path);

    if (result.isUnavailable) {
      return { isValid: true, shouldUpdateUser: false, unavailable: "network" };
    }

    if (!result.isValid) {
      return {
        isValid: false,
        redirectPath: result.redirectPath,
        shouldUpdateUser: true,
      };
    }

    if (result.user) {
      // A role change (e.g. an admin demoted) is a new identity too: it
      // decides what this tab may do, impersonation included.
      const shouldUpdateUser =
        params.currentUser?.id !== result.user.id ||
        params.currentUser?.role !== result.user.role;
      clearWebSocketDisconnectIntent();
      return {
        isValid: true,
        user: result.user,
        shouldUpdateUser,
      };
    }

    return {
      isValid: true,
      shouldUpdateUser: false,
    };
  } catch (error) {
    // Only the auth backend can say a session is invalid, and it says so by
    // returning `isValid: false`. A throw means we never got its answer, so
    // the user stays signed in and the caller retries.
    if (isUnrecognizedActionError(error)) {
      return {
        isValid: true,
        shouldUpdateUser: false,
        unavailable: "deployment-skew",
      };
    }
    console.warn("Session validation unavailable:", error);
    return { isValid: true, shouldUpdateUser: false, unavailable: "network" };
  }
}

interface RefreshSessionResult {
  user?: User | null;
  error?: string;
  isUnavailable?: boolean;
}

// Never rejects: callers get a result either way. A call that didn't reach
// the server says nothing about the session, so it comes back unavailable.
export async function refreshSession(): Promise<RefreshSessionResult> {
  let result: RefreshSessionResult;
  try {
    result = await refreshSessionAction();
  } catch (error) {
    console.warn("Session refresh unavailable:", error);
    return {
      user: null,
      error: "Couldn't reach the server",
      isUnavailable: true,
    };
  }

  if (result.user) {
    clearWebSocketDisconnectIntent();
  }

  return result;
}

interface StorageEventHandlerParams {
  event: StorageEvent;
  api: BackendAPI | null;
  router: AppRouterInstance | null;
  path: string;
}

interface StorageEventHandlerResult {
  shouldLogout: boolean;
  redirectPath?: string | null;
}

export function handleStorageEvent(
  params: StorageEventHandlerParams,
): StorageEventHandlerResult {
  if (!isLogoutEvent(params.event)) {
    return { shouldLogout: false };
  }

  setWebSocketDisconnectIntent();

  // Another tab logged out: drop this tab's sessionStorage copy of any admin
  // impersonation too, or it would be re-sent after the next sign-in.
  ImpersonationState.clear();

  if (params.api) {
    params.api.disconnectWebSocket();
  }

  const redirectPath = getRedirectPath(params.path);

  return {
    shouldLogout: true,
    redirectPath,
  };
}
