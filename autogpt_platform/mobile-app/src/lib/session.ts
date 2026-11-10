import { QueryCache, QueryClient } from "@tanstack/react-query";

import { ApiError } from "@/lib/api/client";
import { authClient } from "@/lib/auth/client";
import { clearAccessToken, SignedOutError } from "@/lib/auth/token";

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

async function clearSession() {
  try {
    await authClient.signOut();
  } catch {
    // Offline or the server refused: the local session must still go.
  } finally {
    clearAccessToken();
    queryClient.clear();
  }
}

// apiRequest already retried once with a fresh token, so a 401 here means
// the session itself is dead.
function isSessionGone(error: unknown) {
  if (error instanceof SignedOutError) return true;
  return error instanceof ApiError && error.status === 401;
}
