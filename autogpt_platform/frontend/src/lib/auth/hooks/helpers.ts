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

interface ValidateSessionParams {
  path: string;
  currentUser: User | null;
}

interface ValidateSessionResult {
  isValid: boolean;
  user?: User | null;
  redirectPath?: string | null;
  shouldUpdateUser: boolean;
}

export async function validateSession(
  params: ValidateSessionParams,
): Promise<ValidateSessionResult> {
  try {
    const result = await validateSessionAction(params.path);

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
    console.error("Session validation error:", error);
    const redirectPath = getRedirectPath(params.path);
    return {
      isValid: false,
      redirectPath,
      shouldUpdateUser: true,
    };
  }
}

interface RefreshSessionResult {
  user?: User | null;
  error?: string;
}

// Never rejects: callers always get a result. A call that didn't reach the
// server is reported as an error, which the store treats as no session.
export async function refreshSession(): Promise<RefreshSessionResult> {
  let result: RefreshSessionResult;
  try {
    result = await refreshSessionAction();
  } catch (error) {
    console.warn("Session refresh failed:", error);
    return { user: null, error: "Couldn't reach the server" };
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
