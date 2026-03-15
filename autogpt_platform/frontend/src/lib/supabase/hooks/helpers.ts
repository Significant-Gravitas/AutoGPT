import type BackendAPI from "@/lib/autogpt-server-api/client";
import { environment } from "@/services/environment";
import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient, User } from "@supabase/supabase-js";
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

let supabaseSingleton: SupabaseClient | null = null;

export function ensureSupabaseClient(): SupabaseClient | null {
  if (supabaseSingleton) return supabaseSingleton;

  const supabaseUrl = environment.getSupabaseUrl();
  const supabaseKey = environment.getSupabaseAnonKey();

  if (!supabaseUrl || !supabaseKey) return null;

  try {
    supabaseSingleton = createBrowserClient(supabaseUrl, supabaseKey, {
      isSingleton: true,
      auth: {
        persistSession: false,
      },
    });
  } catch (error) {
    console.error("Error creating Supabase client", error);
    supabaseSingleton = null;
  }

  return supabaseSingleton;
}

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

export async function refreshSession(): Promise<RefreshSessionResult> {
  const result = await refreshSessionAction();

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

  if (params.api) {
    params.api.disconnectWebSocket();
  }

  const redirectPath = getRedirectPath(params.path);

  return {
    shouldLogout: true,
    redirectPath,
  };
}
