"use client";

import { environment } from "@/services/environment";
import { useFeatureFlagVariantKey } from "@posthog/react";
import { useReportAssignment } from "./useReportAssignment";

export { resetReportedAssignmentsForTests } from "./useReportAssignment";

/**
 * Read an A/B/C experiment arm bucketed by PostHog and make it durable.
 *
 * PostHog buckets the user (multivariate feature flag) and records the
 * exposure for its own significance testing. The arm is also reported to
 * the backend once per user and experiment, so the `analytics.*` views can
 * split activation, retention and cost by variant.
 *
 * `variant` is the arm key (`"control"`, `"yearly-pro"`, ...) or `null` when
 * the user is not enrolled. `isResolved` is false while PostHog is still
 * loading flags: render the control experience and hold any one-shot
 * side effects until it flips, so a late variant is never mis-recorded as
 * control. When PostHog is disabled the experiment resolves immediately.
 *
 * For an experiment bucketed by a LaunchDarkly flag instead, use
 * `useLaunchDarklyExperiment`; both report into the same table.
 */
export function useExperiment(experimentKey: string) {
  const rawVariant = useFeatureFlagVariantKey(experimentKey);

  const isResolved =
    rawVariant !== undefined || !environment.isPostHogEnabled();
  const variant = typeof rawVariant === "string" ? rawVariant : null;

  useReportAssignment({
    experimentKey,
    variant,
    isResolved,
    source: "posthog",
  });

  return { variant, isResolved };
}
