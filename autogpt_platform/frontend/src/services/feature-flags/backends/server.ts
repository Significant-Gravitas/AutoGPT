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
      // A consent answer keeps the last flags until the new ones land; another
      // user never sees them.
      placeholderData: (previous, previousQuery) =>
        previousQuery?.queryKey.at(-1) === userID ? previous : undefined,
      // A failed request, or a 200 leaving this flag out (vendor down or
      // starting), is asked again, backing off to once a minute.
      refetchInterval: ({ state }) =>
        (state.status === "error" || isUnanswered(state.data, key)) &&
        Math.min(
          UNANSWERED_RETRY_MS *
            2 ** (state.dataUpdateCount + state.errorUpdateCount - 1),
          FLAGS_STALE_MS,
        ),
      select: (response) => (response.status === 200 ? response.data : null),
    },
  });

  if (!data) return { value: undefined, resolved: false };
  // A flag left out has no answer yet, so it resolves only with no vendor to wait
  // for; one answered null has no definition there, so its default stands.
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
