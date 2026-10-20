"use client";

import { usePostHog } from "@posthog/react";
import type { FlagSourceResult } from "../flag-source";
import { type Flag, resolveFlagValue } from "../use-get-flag";
import { usePostHogFlagResolution } from "./posthog-resolution";

export function usePostHogFlag(key: string): FlagSourceResult {
  const posthog = usePostHog();
  const resolution = usePostHogFlagResolution(posthog);
  const enabled = posthog?.isFeatureEnabled(key);
  const payload = posthog?.getFeatureFlagPayload(key);
  const mature = resolveFlagValue(key as Flag, undefined) === true;
  const resolved =
    resolution.resolved &&
    (!mature ||
      resolution.omissionsResolved ||
      resolution.evaluatedKeys.has(key));

  // Only a complete response makes an omitted mature flag authoritatively off.
  if (resolution.omissionsResolved && enabled === undefined && mature) {
    return { value: false, resolved: true };
  }

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
