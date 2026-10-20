import type { PostHog } from "posthog-js";
import { useSyncExternalStore } from "react";

interface Resolution {
  snapshot: { resolved: boolean };
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
    snapshot: { resolved: false },
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
    let subscribing = true;
    resolution.unsubscribe = client.onFeatureFlags(
      (_flags, _variants, context) => {
        // hasLoadedFlags includes failures; immediate replay omits context.
        if (subscribing && context?.errorsLoading === undefined) return;
        resolution.snapshot = {
          resolved:
            context?.errorsLoading === undefined
              ? resolution.snapshot.resolved
              : !context.errorsLoading,
        };
        resolution.listeners.forEach((listener) => listener());
      },
    );
    subscribing = false;
  }
  return () => {
    if (--resolution.observers === 0) {
      resolution.unsubscribe?.();
      resolution.unsubscribe = undefined;
    }
  };
}

const unresolved = { resolved: false };

function subscribeWithoutClient() {
  return () => {};
}

export function usePostHogFlagResolution(client: PostHog | undefined) {
  const snapshot = useSyncExternalStore(
    client ? getResolution(client).subscribe : subscribeWithoutClient,
    () => (client ? getResolution(client).snapshot : unresolved),
    () => unresolved,
  );
  return Boolean(client?.featureFlags?.hasLoadedFlags && snapshot.resolved);
}
