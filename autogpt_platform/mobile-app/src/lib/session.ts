import { QueryCache, QueryClient } from "@tanstack/react-query";

import { ApiError } from "@/lib/api/client";
import { authClient } from "@/lib/auth/client";
import { SignedOutError } from "@/lib/auth/errors";
import { clearAccessToken } from "@/lib/auth/token";

const SIGN_OUT_TIMEOUT_MS = 8_000;

let pendingSignOut: Promise<void> | null = null;

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError(error) {
      if (isSessionGone(error)) void signOut();
    },
  }),
  defaultOptions: { queries: { retry: 1 } },
});

/** Several failing queries can ask at once; they share one sign-out. */
export function signOut() {
  if (!pendingSignOut) {
    pendingSignOut = clearSession().finally(() => {
      pendingSignOut = null;
    });
  }
  return pendingSignOut;
}

/** Resolves once no sign-out is in flight, so a new sign-in can't be undone by it. */
export async function whenSignedOut() {
  await pendingSignOut;
}

// apiRequest already retried once with a fresh token, so a 401 here means
// the session itself is dead.
export function isSessionGone(error: unknown) {
  if (error instanceof SignedOutError) return true;
  return error instanceof ApiError && error.status === 401;
}

async function clearSession() {
  // The local session goes first, so a slow network can't leave the old
  // user's data on screen.
  clearAccessToken();
  queryClient.clear();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SIGN_OUT_TIMEOUT_MS);
  try {
    await authClient.signOut({ fetchOptions: { signal: controller.signal } });
  } catch {
    // Offline, timed out or refused: the local session is already gone.
  } finally {
    clearTimeout(timeout);
    // A query in flight during the request can have refilled the cache.
    clearAccessToken();
    queryClient.clear();
  }
}
