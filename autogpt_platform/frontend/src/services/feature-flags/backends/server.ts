"use client";

import {
  getGetFeatureFlagsQueryKey,
  useGetFeatureFlags,
  type getFeatureFlagsResponse,
} from "@/app/api/__generated__/endpoints/feature-flags/feature-flags";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useConsent } from "@/services/consent/useConsent";
import { environment } from "@/services/environment";
import posthog from "posthog-js";
import type { FlagSourceResult } from "../flag-source";

const FLAGS_STALE_MS = 60_000;
const UNANSWERED_RETRY_MS = 5_000;

export function useServerFlag(key: string): FlagSourceResult {
  const userID = useAuthStore((state) => state.user?.id ?? null);
  const isUserLoading = useAuthStore((state) => state.isUserLoading);
  const { analytics } = useConsent();
  const visitorID = userID === null ? consentedVisitorID(analytics) : null;
  const params = visitorID ? { visitor_id: visitorID } : undefined;
  // Keyed by user and visitor id, so signing in or out, or answering the
  // consent banner, reads the matching flags rather than the cached ones.
  const { data } = useGetFeatureFlags(params, {
    query: {
      queryKey: [...getGetFeatureFlagsQueryKey(params), userID],
      enabled: !isUserLoading,
      staleTime: FLAGS_STALE_MS,
      // A 200 leaves a flag out while the vendor is down or starting, so ask
      // again, backing off to once a minute, until this one is answered.
      refetchInterval: (query) =>
        isUnanswered(query.state.data, key) &&
        Math.min(
          UNANSWERED_RETRY_MS * 2 ** (query.state.dataUpdateCount - 1),
          FLAGS_STALE_MS,
        ),
      select: (response) => (response.status === 200 ? response.data : null),
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

// Only with analytics consent does a signed-out visitor get their own bucket;
// without it the server reads them as the one shared anonymous visitor.
function consentedVisitorID(analytics: boolean) {
  if (!analytics || !environment.isPostHogEnabled() || !posthog.__loaded) {
    return null;
  }
  return posthog.get_distinct_id() || null;
}

function isUnanswered(
  response: getFeatureFlagsResponse | undefined,
  key: string,
) {
  return (
    response?.status === 200 &&
    response.data.configured &&
    !(key in response.data.flags)
  );
}
