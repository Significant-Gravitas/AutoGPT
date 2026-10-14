import { getAnonymousID } from "@/services/analytics/anonymous-id";
import { getPostHogBaseProperties } from "@/services/analytics/posthog-base-properties";
import { usesPostHog } from "@/services/feature-flags/flag-backend";
import posthog, {
  type PostHog,
  type PostHogConfig,
  type PostHogInterface,
} from "posthog-js";
import { getPostHogClientHosts } from "./posthog-proxy";

// Dev reads prod's flag definitions (NEXT_PUBLIC_POSTHOG_FLAGS_KEY) while its
// analytics stay in the dev project. posthog-js asks /flags of the project its
// token names, so those reads need an instance of their own.
const FLAGS_INSTANCE_NAME = "flags";

let flagsOnlyClient: PostHog | null = null;

/**
 * The instance flag reads go through: the analytics one, unless flags come
 * from another project. That instance is created on the first read, not in
 * the provider's effect, which React runs only after its children's: a flag
 * read in the first render must already go to the flags project.
 */
export function getFlagsClient(): PostHog {
  return ensureFlagsOnlyClient() ?? posthog;
}

/** The flags-only instance, created on first use; null unless flags come from another project. */
export function ensureFlagsOnlyClient(): PostHog | null {
  if (flagsOnlyClient || typeof window === "undefined") return flagsOnlyClient;
  const project = getFlagsOnlyProject();
  if (!project) return null;
  flagsOnlyClient = posthog.init(
    project.flagsKey,
    {
      ...project.hosts,
      ...getFlagsOnlyConfig(),
      ...bootstrapAnonymousID(getAnonymousID()),
      loaded: applyFlagEnvironment,
    },
    FLAGS_INSTANCE_NAME,
  );
  return flagsOnlyClient;
}

// Read as literals, which Next inlines into the client bundle, rather than
// through `environment`, which the flag hooks' tests stub.
function getFlagsOnlyProject() {
  const analyticsKey = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  const flagsKey = process.env.NEXT_PUBLIC_POSTHOG_FLAGS_KEY;
  if (!usesPostHog() || !analyticsKey || !flagsKey) return null;
  if (flagsKey === analyticsKey) return null;
  const hosts = getPostHogClientHosts({
    key: analyticsKey,
    host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  });
  return { flagsKey, hosts };
}

// The first-party anonymous id the analytics instance and the backend also
// use, so an anonymous rollout buckets the visitor the same everywhere.
function bootstrapAnonymousID(anonymousID: string | null) {
  if (!anonymousID) return {};
  return { bootstrap: { distinctID: anonymousID, isIdentifiedID: false } };
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
  ensureFlagsOnlyClient()?.identify(userID);
}

/**
 * Called with the app's new anonymous id: reset() alone would mint a random
 * one, and the flags project would bucket the signed-out visitor on an id
 * nothing else uses. Re-seeded the way posthog-js applies a bootstrap id.
 */
export function resetFlagsOnlyClient(anonymousID: string | null) {
  if (!flagsOnlyClient) return;
  flagsOnlyClient.reset(true);
  if (anonymousID) {
    flagsOnlyClient.register({
      distinct_id: anonymousID,
      $device_id: anonymousID,
    });
  }
  applyFlagEnvironment(flagsOnlyClient);
  flagsOnlyClient.reloadFeatureFlags();
}
