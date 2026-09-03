"use client";

import { usePostHog } from "@posthog/react";
import { useEffect } from "react";
import type { FlagSourceResult } from "../flag-source";
import { useLaunchDarklyFlag } from "./launchdarkly";
import { usePostHogFlag } from "./posthog";

export function useDualFlag(key: string): FlagSourceResult {
  const launchDarkly = useLaunchDarklyFlag(key);
  const postHog = usePostHogFlag(key);

  useReportMismatch(key, launchDarkly, postHog);

  // Serving LaunchDarkly is what keeps the diff week free of user-visible
  // risk: PostHog's answer is observed, never acted on.
  return launchDarkly;
}

function useReportMismatch(
  key: string,
  launchDarkly: FlagSourceResult,
  postHog: FlagSourceResult,
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
  const agree =
    launchDarkly.resolved === postHog.resolved &&
    JSON.stringify(launchDarkly.value) === JSON.stringify(postHog.value);

  useEffect(() => {
    if (agree) return;
    const mismatch = JSON.parse(record);
    console.warn("feature-flag mismatch", mismatch);
    posthog?.capture("feature_flag_mismatch", mismatch);
  }, [agree, record, posthog]);
}
