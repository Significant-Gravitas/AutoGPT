"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import * as Sentry from "@sentry/nextjs";
import { LDProvider } from "launchdarkly-react-client-sdk";
import type { ReactNode } from "react";
import { useMemo } from "react";
import { getAnonymousID } from "../analytics/anonymous-id";
import { environment } from "../environment";
import { buildLDContext } from "./helpers";

// `LDProvider`'s timeout is in SECONDS, and the SDK warns on every page load
// above 5 — the previous 5000 read as 83 minutes, so initialisation was
// effectively unbounded.
const LAUNCHDARKLY_INIT_TIMEOUT_SECONDS = 5;

// Proxied to https://events.launchdarkly.com by src/app/api/ld-events.
const LAUNCHDARKLY_EVENTS_PATH = "/api/ld-events";

// The SDK default is 2s, so every open tab posts 30 times a minute through our
// own edge. Nothing reads flag analytics at that granularity.
const LAUNCHDARKLY_FLUSH_INTERVAL_MS = 30_000;

export function LaunchDarklyProvider({ children }: { children: ReactNode }) {
  const { user, isUserLoading } = useAuth();
  const envEnabled = environment.areFeatureFlagsEnabled();
  const clientId = environment.getLaunchDarklyClientId();

  const context = useMemo(() => {
    if (isUserLoading) return;
    return buildLDContext(user, getAnonymousID());
  }, [user, isUserLoading]);

  if (!envEnabled) {
    return <>{children}</>;
  }

  if (isUserLoading) {
    return <LoadingSpinner size="large" cover />;
  }

  return (
    <LDProvider
      clientSideID={clientId ?? ""}
      context={context}
      timeout={LAUNCHDARKLY_INIT_TIMEOUT_SECONDS}
      reactOptions={{ useCamelCaseFlagKeys: false }}
      options={{
        inspectors: [Sentry.buildLaunchDarklyFlagUsedHandler()],
        // Analytics events go through our own origin: `events.launchdarkly.com`
        // is on every tracker blocklist, and each rejected flush prints two
        // console errors and is retried once by the SDK.
        eventsUrl: LAUNCHDARKLY_EVENTS_PATH,
        flushInterval: LAUNCHDARKLY_FLUSH_INTERVAL_MS,
      }}
    >
      {children}
    </LDProvider>
  );
}
