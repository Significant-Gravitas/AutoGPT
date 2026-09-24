"use client";

import {
  useFeatureFlagEnabled,
  useFeatureFlagPayload,
  usePostHog,
} from "@posthog/react";
import { useEffect, useState } from "react";
import type { FlagSourceResult } from "../flag-source";

export function usePostHogFlag(key: string): FlagSourceResult {
  const enabled = useFeatureFlagEnabled(key);
  const payload = useFeatureFlagPayload(key);
  // Not `enabled !== undefined`: posthog-js serves its persisted snapshot
  // before /flags runs, and answers `undefined` for a flag it has never heard of.
  const resolved = useFlagsLoaded();

  // An explicit off wins over a payload, as on the backend.
  if (enabled === false) {
    return { value: false, resolved };
  }

  // A payload stands in for the JSON-valued LaunchDarkly variations; a plain
  // release toggle carries none and answers with the boolean.
  if (payload !== undefined && payload !== null) {
    return { value: payload, resolved };
  }

  return { value: enabled, resolved };
}

function useFlagsLoaded() {
  const posthog = usePostHog();
  const [loaded, setLoaded] = useState(
    () => posthog?.featureFlags?.hasLoadedFlags ?? false,
  );

  useEffect(() => {
    return posthog?.onFeatureFlags((_flags, _variants, context) => {
      if (!context?.errorsLoading) setLoaded(true);
    });
  }, [posthog]);

  return loaded;
}
