"use client";

import { environment } from "@/services/environment";
import { useFlags, useLDClient } from "launchdarkly-react-client-sdk";
import posthog from "posthog-js";
import { useEffect, useState } from "react";
import { useReportAssignment } from "./useReportAssignment";

const LD_READY_TIMEOUT_MS = 5000;

/**
 * Run an experiment whose arms are a LaunchDarkly string flag.
 *
 * The team gates with LaunchDarkly, but LaunchDarkly only knows a user after
 * login and PostHog is where funnels and significance live. This hook is
 * the bridge: LaunchDarkly picks the arm, and the hook (1) reports the
 * assignment to the backend like `useExperiment` does, so Looker can split
 * by arm, and (2) sends PostHog an `experiment_exposed` event carrying the
 * arm as `$feature/<flag>`, so a PostHog experiment can use it as its
 * exposure and measure the activation events against it.
 *
 * `variant` is the flag's string value, or `null` when the flag is off,
 * boolean, or not yet known. `isResolved` flips once the LaunchDarkly client
 * has initialised (or timed out) so a late arm is never recorded as control.
 */
export function useLaunchDarklyExperiment(flagKey: string) {
  const flags = useFlags<Record<string, unknown>>();
  const client = useLDClient();
  const [isClientReady, setIsClientReady] = useState(false);
  const flagsEnabled = environment.areFeatureFlagsEnabled();

  useEffect(() => {
    if (!client) return;
    let active = true;
    client
      .waitForInitialization(LD_READY_TIMEOUT_MS)
      .catch(() => undefined)
      .finally(() => {
        if (active) setIsClientReady(true);
      });
    return () => {
      active = false;
    };
  }, [client]);

  const isResolved = !flagsEnabled || isClientReady;
  const raw = flags[flagKey];
  const variant = typeof raw === "string" ? raw : null;

  useReportAssignment({
    experimentKey: flagKey,
    variant,
    isResolved,
    source: "launchdarkly",
  });

  useEffect(() => {
    if (!isResolved || !variant || !claimExposure(flagKey, variant)) return;
    if (!environment.isPostHogEnabled()) return;
    try {
      posthog.capture("experiment_exposed", {
        experiment_key: flagKey,
        variant,
        provider: "launchdarkly",
        [`$feature/${flagKey}`]: variant,
      });
    } catch {
      // Analytics must never break the experience it measures.
    }
  }, [isResolved, variant, flagKey]);

  return { variant, isResolved };
}

const exposures = new Set<string>();

function claimExposure(flagKey: string, variant: string) {
  const key = `${flagKey}:${variant}`;
  if (exposures.has(key)) return false;
  exposures.add(key);
  return true;
}

export function resetExposuresForTests() {
  exposures.clear();
}
