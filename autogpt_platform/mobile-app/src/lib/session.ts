import { QueryCache, QueryClient } from "@tanstack/react-query";

import { ApiError } from "@/lib/api/client";
import { authClient } from "@/lib/auth/client";
import { clearAccessToken, SignedOutError } from "@/lib/auth/token";

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError(error) {
      if (isSessionGone(error)) void signOut();
    },
  }),
  defaultOptions: { queries: { retry: 1 } },
});

export async function signOut() {
  await authClient.signOut();
  clearAccessToken();
  queryClient.clear();
}

// apiRequest already retried once with a fresh token, so a 401 here means
// the session itself is dead.
function isSessionGone(error: unknown) {
  if (error instanceof SignedOutError) return true;
  return error instanceof ApiError && error.status === 401;
}
