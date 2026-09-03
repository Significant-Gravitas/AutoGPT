"use client";

import { useFeatureFlagEnabled, useFeatureFlagPayload } from "@posthog/react";
import type { FlagSourceResult } from "../flag-source";

export function usePostHogFlag(key: string): FlagSourceResult {
  const enabled = useFeatureFlagEnabled(key);
  const payload = useFeatureFlagPayload(key);

  // A payload stands in for the JSON-valued LaunchDarkly variations; a plain
  // release toggle carries none and answers with the boolean.
  if (payload !== undefined && payload !== null) {
    return { value: payload, resolved: true };
  }

  return { value: enabled, resolved: enabled !== undefined };
}
