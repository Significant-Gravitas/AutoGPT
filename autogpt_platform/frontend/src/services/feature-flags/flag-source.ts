"use client";

import { useDualFlag } from "./backends/dual";
import { useLaunchDarklyFlag } from "./backends/launchdarkly";
import { usePostHogFlag } from "./backends/posthog";
import { FLAG_BACKEND } from "./flag-backend";

export interface FlagSourceResult {
  value: unknown;
  // Whether the vendor actually answered, as opposed to not having answered
  // yet. Callers that gate a route must not treat "no answer" as "off".
  resolved: boolean;
}

// The one seam every flag read passes through, and the only thing a test has
// to mock to control flags regardless of which vendor is configured.
export function useFlagSource(key: string): FlagSourceResult {
  return useSelectedFlagSource(key);
}

// Picked at module load, not per render: the backend is a build-time constant,
// so the hooks a component calls stay stable for the life of the app.
const useSelectedFlagSource =
  FLAG_BACKEND === "posthog"
    ? usePostHogFlag
    : FLAG_BACKEND === "dual"
      ? useDualFlag
      : useLaunchDarklyFlag;
