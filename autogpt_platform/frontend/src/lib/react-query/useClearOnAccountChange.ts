"use client";

import { onAccountChange } from "@/lib/auth/on-account-change";
import type { QueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

// logOut() clears the cache in the tab that signs out, but a session can end
// without it (a sign-out in another tab, a failed session check). Whoever
// signs in next on this tab must not be shown the last account's data (its
// chats, among the rest), so the cache goes the moment another account
// signs in. A dropped session and the same person signing back in keep it.
export function useClearOnAccountChange(queryClient: QueryClient) {
  useEffect(() => onAccountChange(() => queryClient.clear()), [queryClient]);
}
