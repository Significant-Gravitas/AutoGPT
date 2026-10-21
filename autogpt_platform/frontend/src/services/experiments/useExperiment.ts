"use client";

import {
  getGetExperimentsGetExperimentAssignmentQueryKey,
  useGetExperimentsGetExperimentAssignment,
} from "@/app/api/__generated__/endpoints/experiments/experiments";
import type { Experiment } from "@/app/api/__generated__/models/experiment";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { environment } from "@/services/environment";
import { useEffect } from "react";

/**
 * The A/B/C arm the backend assigned this user, fixed on first read.
 *
 * The backend buckets the user by id and stores the arm, so it is the same on
 * every page load whether or not analytics consent lets the browser identify
 * them. `variant` is the arm key, or `null` when the user is not enrolled or
 * not signed in. `isResolved` is false while the arm loads: render the control
 * experience and hold one-shot side effects until it flips.
 */
export function useExperiment(experiment: Experiment) {
  const userID = useAuthStore((state) => state.user?.id ?? null);
  const isUserLoading = useAuthStore((state) => state.isUserLoading);
  const { data, isPending } = useGetExperimentsGetExperimentAssignment(
    experiment,
    {
      query: {
        // Keyed by user, so a sign-in never reads the previous user's arm.
        queryKey: [
          ...getGetExperimentsGetExperimentAssignmentQueryKey(experiment),
          userID,
        ],
        enabled: userID !== null,
        staleTime: Infinity,
        select: (response) => (response.status === 200 ? response.data : null),
      },
    },
  );
  const variant = data?.variant ?? null;

  useReportExposure(experiment, variant);

  return {
    variant,
    isResolved: !isUserLoading && (userID === null || !isPending),
  };
}

// PostHog's experiment results count this event. Held until PostHog captures, so
// an arm that loads before the consent answer still counts.
function useReportExposure(experiment: Experiment, variant: string | null) {
  useEffect(() => {
    if (!variant || !environment.isPostHogEnabled()) return;
    capturePostHogEvent("$feature_flag_called", {
      $feature_flag: experiment,
      $feature_flag_response: variant,
    });
  }, [experiment, variant]);
}
