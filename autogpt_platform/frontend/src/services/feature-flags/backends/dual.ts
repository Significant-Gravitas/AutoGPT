"use client";

import { FeatureFlagEvent } from "@/services/analytics/posthog-events";
import { environment } from "@/services/environment";
import { usePostHog } from "@posthog/react";
import * as Sentry from "@sentry/nextjs";
import isEqual from "lodash/isEqual";
import { useEffect } from "react";
import { isPostHogFlagsEnabled } from "../flag-backend";
import type { FlagSourceResult } from "../flag-source";
import { type Flag, resolveFlagValue } from "../use-get-flag";
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
    // `null`, not `undefined`: a flag PostHog has never heard of must still
    // appear in the record rather than vanish from the JSON.
    posthog: { value: postHog.value ?? null, resolved: postHog.resolved },
  });
  // The two vendors never answer on the same render — LaunchDarkly fetches
  // over the network, PostHog reads its bootstrapped snapshot — so comparing
  // before both have resolved reports the load order, not a disagreement.
  const comparable =
    bothConfigured &&
    isPostHogFlagsEnabled() &&
    launchDarkly.resolved &&
    postHog.resolved;
  // What each vendor would serve, not its raw answer: LaunchDarkly's `false`
  // and PostHog's empty answer for a flag it has switched off both serve the
  // default (BUILDER-94P). Key-order-insensitive, like the backend's dict
  // comparison: the vendors need not serialise a JSON flag's keys in one order.
  const agree = isEqual(
    resolveFlagValue(key as Flag, launchDarkly.value),
    resolveFlagValue(key as Flag, postHog.value),
  );

  useEffect(() => {
    if (!comparable || agree) return;
    const mismatch = JSON.parse(record);
    reportMismatchToSentry(mismatch);
    posthog?.capture(FeatureFlagEvent.FEATURE_FLAG_MISMATCHED, mismatch);
  }, [comparable, agree, record, posthog]);
}

interface Mismatch {
  flag: string;
  launchdarkly: { value: unknown };
  posthog: { value: unknown };
}

// Sentry tag values are capped at 200 characters.
const MAX_TAG_LENGTH = 200;

// One issue per flag, so the patrol sees which flags disagree; the tags carry
// each vendor's answer, so the issue shows the direction too (BUILDER-94P).
export function reportMismatchToSentry({
  flag,
  launchdarkly,
  posthog,
}: Mismatch) {
  const launchDarklyValue = describeValue(launchdarkly.value);
  const postHogValue = describeValue(posthog.value);
  const message = `Feature flag mismatch: ${flag} (LaunchDarkly=${launchDarklyValue}, PostHog=${postHogValue})`;
  console.info(message);
  Sentry.captureMessage(message, {
    level: "warning",
    fingerprint: ["feature-flag-mismatch", flag],
    tags: {
      "feature_flag.key": flag,
      "feature_flag.launchdarkly": launchDarklyValue.slice(0, MAX_TAG_LENGTH),
      "feature_flag.posthog": postHogValue.slice(0, MAX_TAG_LENGTH),
    },
    contexts: {
      feature_flag_mismatch: {
        flag,
        launchdarkly: launchDarklyValue,
        posthog: postHogValue,
      },
    },
  });
}

function describeValue(value: unknown) {
  return value === undefined ? "undefined" : JSON.stringify(value);
}
