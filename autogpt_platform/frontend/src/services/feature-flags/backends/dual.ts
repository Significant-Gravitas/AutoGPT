"use client";

import { environment } from "@/services/environment";
import { usePostHog } from "@posthog/react";
import { useEffect } from "react";
import { isPostHogFlagsEnabled } from "../flag-backend";
import type { FlagSourceResult } from "../flag-source";
import { useLaunchDarklyFlag } from "./launchdarkly";
import { usePostHogFlag } from "./posthog";

export function useDualFlag(key: string): FlagSourceResult {
  const launchDarkly = useLaunchDarklyFlag(key);
  const postHog = usePostHogFlag(key);
  // LaunchDarkly is what dual serves, but only when it is actually configured:
  // with no LDProvider mounted its answer is an empty set that never resolves.
  const serveLaunchDarkly = environment.areFeatureFlagsEnabled();

  useReportMismatch(key, launchDarkly, postHog, serveLaunchDarkly);

  return serveLaunchDarkly ? launchDarkly : postHog;
}

function useReportMismatch(
  key: string,
  launchDarkly: FlagSourceResult,
  postHog: FlagSourceResult,
  bothConfigured: boolean,
) {
  const posthog = usePostHog();
  const record = JSON.stringify({
    flag: key,
    launchdarkly: {
      value: launchDarkly.value,
      resolved: launchDarkly.resolved,
    },
    posthog: { value: postHog.value, resolved: postHog.resolved },
  });
  // The two vendors never answer on the same render — LaunchDarkly fetches
  // over the network, PostHog reads its bootstrapped snapshot — so comparing
  // before both have resolved reports the load order, not a disagreement.
  const comparable =
    bothConfigured &&
    isPostHogFlagsEnabled() &&
    launchDarkly.resolved &&
    postHog.resolved;
  const agree =
    JSON.stringify(launchDarkly.value) === JSON.stringify(postHog.value);

  useEffect(() => {
    if (!comparable || agree) return;
    const mismatch = JSON.parse(record);
    console.warn("feature-flag mismatch", mismatch);
    posthog?.capture("feature_flag_mismatch", mismatch);
  }, [comparable, agree, record, posthog]);
}
