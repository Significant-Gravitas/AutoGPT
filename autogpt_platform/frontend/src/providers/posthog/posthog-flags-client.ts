import { getPostHogBaseProperties } from "@/services/analytics/posthog-base-properties";
import posthog, {
  type PostHog,
  type PostHogConfig,
  type PostHogInterface,
} from "posthog-js";

// Dev reads prod's flag definitions (NEXT_PUBLIC_POSTHOG_FLAGS_KEY) while its
// analytics stay in the dev project. posthog-js asks /flags of the project its
// token names, so those reads need an instance of their own.
const FLAGS_INSTANCE_NAME = "flags";

let flagsOnlyClient: PostHog | null = null;

interface InitArgs {
  flagsKey?: string;
  analyticsKey: string;
  hosts: Partial<PostHogConfig>;
  anonymousID: string | null;
}

/** The instance flag reads go through: the analytics one, unless flags come from another project. */
export function getFlagsClient(): PostHog {
  return flagsOnlyClient ?? posthog;
}

/** Run after the analytics instance is initialised; a no-op unless flags come from another project. */
export function initFlagsOnlyClient({
  flagsKey,
  analyticsKey,
  hosts,
  anonymousID,
}: InitArgs) {
  if (!flagsKey || flagsKey === analyticsKey || flagsOnlyClient) return;
  flagsOnlyClient = posthog.init(
    flagsKey,
    {
      ...hosts,
      ...getFlagsOnlyConfig(),
      ...(anonymousID && {
        bootstrap: { distinctID: anonymousID, isIdentifiedID: false },
      }),
      loaded: applyFlagEnvironment,
    },
    FLAGS_INSTANCE_NAME,
  );
}

/**
 * Everything but /flags off, so the flags project holds no event or person
 * from this deployment: no capture of any kind (before_send drops whatever
 * an extension still tries), nothing stored in the browser, no scripts loaded.
 */
function getFlagsOnlyConfig() {
  return {
    persistence: "memory",
    opt_out_capturing_by_default: true,
    opt_out_persistence_by_default: true,
    before_send: () => null,
    autocapture: false,
    capture_pageview: false,
    capture_pageleave: false,
    capture_dead_clicks: false,
    capture_exceptions: false,
    capture_heatmaps: false,
    capture_performance: false,
    rageclick: false,
    disable_session_recording: true,
    disable_surveys: true,
    disable_product_tours: true,
    disable_conversations: true,
    disable_web_experiments: true,
    disable_external_dependency_loading: true,
    advanced_disable_toolbar_metrics: true,
    save_referrer: false,
    save_campaign_params: false,
  } satisfies Partial<PostHogConfig>;
}

/**
 * Lets one set of flag definitions hold a rollout to dev alone: a release
 * condition on the same `environment` the analytics events carry. Set before
 * the first /flags call (posthog-js runs `loaded` just ahead of it), and
 * again after a reset, which clears it.
 */
export function applyFlagEnvironment(client: PostHogInterface) {
  client.setPersonPropertiesForFlags(
    { environment: getPostHogBaseProperties().environment },
    false,
  );
}

export function setFlagPersonProperties(properties: Record<string, string>) {
  getFlagsClient().setPersonPropertiesForFlags(properties);
}

export function identifyFlagsOnlyClient(userID: string) {
  flagsOnlyClient?.identify(userID);
}

export function resetFlagsOnlyClient() {
  if (!flagsOnlyClient) return;
  flagsOnlyClient.reset(true);
  applyFlagEnvironment(flagsOnlyClient);
  flagsOnlyClient.reloadFeatureFlags();
}
