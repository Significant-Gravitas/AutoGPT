import { QueryClient, useQuery } from "@tanstack/react-query";
import Constants from "expo-constants";
import { useState } from "react";
import { AppState } from "react-native";

import { requestWithin } from "@/lib/deadline";
import { API_URL } from "@/lib/env";
import { useMountEffect } from "@/lib/useMountEffect";

import { updateGate } from "./helpers";

// Short: the answer holds up the launch, and no answer lets the app in.
const CHECK_TIMEOUT_MS = 4_000;
const minVersionKey = ["app", "min-version"] as const;

/** The version this build ships as (`version` in app.json). */
export const APP_VERSION = Constants.expoConfig?.version ?? null;

/**
 * Asks the server for the oldest app version it still lets in, at launch and
 * whenever the app comes back to the front. Signed in or not: the setting is
 * public, so an old app stops before the sign-in screen too.
 */
export function useUpdateGate() {
  const [queryClient] = useState(() => new QueryClient());
  const query = useQuery(
    {
      queryKey: minVersionKey,
      queryFn: () =>
        requestWithin(CHECK_TIMEOUT_MS, fetchMinimumVersion).catch(
          () => queryClient.getQueryData<string | null>(minVersionKey) ?? null,
        ),
      retry: false,
      staleTime: Infinity,
      gcTime: 0,
    },
    queryClient,
  );

  useMountEffect(() => {
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active")
        void queryClient.invalidateQueries({ queryKey: minVersionKey });
    });
    return () => subscription.remove();
  });

  // A failed re-check keeps the last answer rather than letting an old app in.
  const status = query.data !== undefined ? "success" : query.status;
  return updateGate(status, query.data ?? null, APP_VERSION);
}

// Not `apiRequest`: that needs a signed-in session, and this runs before one.
async function fetchMinimumVersion(signal: AbortSignal) {
  const response = await fetch(`${API_URL}/api/platform-config`, { signal });
  if (!response.ok)
    throw new Error(`Platform config failed (${response.status})`);
  const data = (await response.json()) as { mobile_min_version?: unknown };
  return typeof data.mobile_min_version === "string"
    ? data.mobile_min_version
    : null;
}
