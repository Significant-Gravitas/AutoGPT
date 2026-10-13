"use client";

import { getFlagsClient } from "@/providers/posthog/posthog-flags-client";
import { useEffect, useState } from "react";
import type { FlagSourceResult } from "../flag-source";

export function usePostHogFlag(key: string): FlagSourceResult {
  const { enabled, payload, resolved } = useFlagAnswer(key);

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

// Read from the flags client rather than through @posthog/react's hooks, which
// only see the analytics instance in context: on dev the two are different
// PostHog projects.
function useFlagAnswer(key: string) {
  const client = getFlagsClient();
  const [enabled, setEnabled] = useState(() => client.isFeatureEnabled(key));
  const [payload, setPayload] = useState(() =>
    client.getFeatureFlagPayload(key),
  );
  // Not `enabled !== undefined`: posthog-js serves its persisted snapshot
  // before /flags runs, and answers `undefined` for a flag it has never heard of.
  const [resolved, setResolved] = useState(
    () => client.featureFlags?.hasLoadedFlags ?? false,
  );

  useEffect(() => {
    return client.onFeatureFlags((_flags, _variants, context) => {
      setEnabled(client.isFeatureEnabled(key));
      setPayload(client.getFeatureFlagPayload(key));
      if (!context?.errorsLoading) setResolved(true);
    });
  }, [client, key]);

  return { enabled, payload, resolved };
}
