import type { PostHog } from "posthog-js";
import { useSyncExternalStore } from "react";

interface Snapshot {
  resolved: boolean;
  omissionsResolved: boolean;
  evaluatedKeys: ReadonlySet<string>;
}

interface Resolution {
  snapshot: Snapshot;
  response?: Snapshot;
  listeners: Set<() => void>;
  observers: number;
  unsubscribe?: () => void;
  subscribe: (listener: () => void) => () => void;
}

const resolutions = new WeakMap<PostHog, Resolution>();

function getResolution(client: PostHog) {
  const existing = resolutions.get(client);
  if (existing) return existing;
  const resolution: Resolution = {
    snapshot: unresolved,
    listeners: new Set(),
    observers: 0,
    subscribe(listener) {
      resolution.listeners.add(listener);
      const stopTracking = trackPostHogFlagResolution(client);
      return () => {
        resolution.listeners.delete(listener);
        stopTracking();
      };
    },
  };
  resolutions.set(client, resolution);
  return resolution;
}

export function trackPostHogFlagResolution(client: PostHog) {
  const resolution = getResolution(client);
  if (resolution.observers++ === 0) {
    const restoreReceiver = observeResponses(client, resolution);
    let subscribing = true;
    const unsubscribe = client.onFeatureFlags((_flags, _variants, context) => {
      // hasLoadedFlags includes failures; immediate replay omits context.
      if (subscribing && context?.errorsLoading === undefined) return;
      resolution.snapshot = resolution.response ?? {
        resolved:
          context?.errorsLoading === undefined
            ? resolution.snapshot.resolved
            : !context.errorsLoading,
        omissionsResolved:
          context?.errorsLoading === undefined &&
          resolution.snapshot.omissionsResolved,
        evaluatedKeys:
          context?.errorsLoading === undefined
            ? resolution.snapshot.evaluatedKeys
            : new Set(),
      };
      resolution.listeners.forEach((listener) => listener());
    });
    subscribing = false;
    resolution.unsubscribe = () => {
      unsubscribe();
      restoreReceiver();
    };
  }
  return () => {
    if (--resolution.observers === 0) {
      resolution.unsubscribe?.();
      resolution.unsubscribe = undefined;
      resolution.snapshot = unresolved;
    }
  };
}

// onFeatureFlags drops partial-evaluation metadata; retain it before the SDK
// merges the response into its cache and notifies React.
function observeResponses(client: PostHog, resolution: Resolution) {
  const featureFlags = client.featureFlags;
  const original = featureFlags.receivedFeatureFlags;
  const receive: typeof original = function (response, errorsLoading) {
    const flags = response.flags ?? response.featureFlags;
    const hasFlags =
      flags !== null && typeof flags === "object" && !Array.isArray(flags);
    const previous = resolution.response;
    resolution.response = {
      resolved: errorsLoading === false,
      omissionsResolved:
        errorsLoading === false &&
        response.errorsWhileComputingFlags === false &&
        hasFlags,
      evaluatedKeys: new Set(hasFlags ? Object.keys(flags) : []),
    };
    try {
      return original.call(featureFlags, response, errorsLoading);
    } finally {
      resolution.response = previous;
    }
  };
  featureFlags.receivedFeatureFlags = receive;
  return () => {
    if (featureFlags.receivedFeatureFlags === receive) {
      featureFlags.receivedFeatureFlags = original;
    }
  };
}

const unresolved: Snapshot = {
  resolved: false,
  omissionsResolved: false,
  evaluatedKeys: new Set(),
};

function subscribeWithoutClient() {
  return () => {};
}

export function usePostHogFlagResolution(client: PostHog | undefined) {
  const snapshot = useSyncExternalStore(
    client ? getResolution(client).subscribe : subscribeWithoutClient,
    () => (client ? getResolution(client).snapshot : unresolved),
    () => unresolved,
  );
  return client?.featureFlags?.hasLoadedFlags ? snapshot : unresolved;
}
