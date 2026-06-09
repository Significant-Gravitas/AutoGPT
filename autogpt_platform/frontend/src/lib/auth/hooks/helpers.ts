import type BackendAPI from "@/lib/autogpt-server-api/client";
import {
  getCurrentUser,
  refreshSession as refreshSessionAction,
  validateSession as validateSessionAction,
} from "@/lib/auth/actions";
import {
  clearWebSocketDisconnectIntent,
  getRedirectPath,
  isLogoutEvent,
  setWebSocketDisconnectIntent,
} from "@/lib/auth/helpers";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";

let authClientSingleton: Record<string, unknown> | null = null;

export const authClient = {
  async getSession() {
    const { user } = await getCurrentUser();
    return {
      data: {
        session: user ? { user } : null,
      },
    };
  },
};

export function ensureSupabaseClient(): Record<string, unknown> {
  if (!authClientSingleton) {
    authClientSingleton = {
      auth: {
        getSession: authClient.getSession,
      },
      getSession: authClient.getSession,
    };
  }

  return authClientSingleton;
}

export async function fetchUser() {
  try {
    const { user, error } = await getCurrentUser();

    if (error || !user) {
      return {
        user: null,
        hasLoadedUser: !!error,
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
      hasLoadedUser: true,
      isUserLoading: false,
    };
  }
}

export async function validateSession(params: {
  path: string;
  currentUser: { id: string } | null;
}) {
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
      const shouldUpdateUser = params.currentUser?.id !== result.user.id;
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
    return {
      isValid: false,
      redirectPath: getRedirectPath(params.path),
      shouldUpdateUser: true,
    };
  }
}

export async function refreshSession() {
  const result = await refreshSessionAction();

  if (result.user) {
    clearWebSocketDisconnectIntent();
  }

  return result;
}

export function handleStorageEvent(params: {
  event: StorageEvent;
  api: BackendAPI | null;
  router: AppRouterInstance | null;
  path: string;
}) {
  if (!isLogoutEvent(params.event)) {
    return { shouldLogout: false };
  }

  setWebSocketDisconnectIntent();

  if (params.api) {
    params.api.disconnectWebSocket();
  }

  return {
    shouldLogout: true,
    redirectPath: getRedirectPath(params.path),
  };
}
