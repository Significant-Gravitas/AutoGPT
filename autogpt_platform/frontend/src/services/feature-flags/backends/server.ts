"use client";

import {
  getGetFeatureFlagsQueryKey,
  useGetFeatureFlags,
} from "@/app/api/__generated__/endpoints/feature-flags/feature-flags";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { FlagSourceResult } from "../flag-source";

const FLAGS_STALE_MS = 60_000;

export function useServerFlag(key: string): FlagSourceResult {
  const userID = useAuthStore((state) => state.user?.id ?? null);
  const isUserLoading = useAuthStore((state) => state.isUserLoading);
  // Keyed by user, so signing in or out reads that user's flags rather than
  // serving the last one's from the cache.
  const { data } = useGetFeatureFlags({
    query: {
      queryKey: [...getGetFeatureFlagsQueryKey(), userID],
      enabled: !isUserLoading,
      staleTime: FLAGS_STALE_MS,
      select: (response) => response.data,
    },
  });

  if (!data) return { value: undefined, resolved: false };
  // A flag the server left out had no answer from the vendor; it only counts as
  // resolved when there is no vendor to wait for.
  return {
    value: data.flags[key],
    resolved: key in data.flags || !data.configured,
  };
}
