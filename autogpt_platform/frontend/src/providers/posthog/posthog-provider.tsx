"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import {
  captureFirstLanding,
  followAnalyticsConsentForIdentity,
  getAnonymousID,
} from "@/services/analytics/anonymous-id";
import { useConsent } from "@/services/consent/useConsent";
import { environment } from "@/services/environment";
import { usesPostHog } from "@/services/feature-flags/flag-backend";
import { buildFlagPersonProperties } from "@/services/feature-flags/helpers";
import { PostHogProvider as PHProvider } from "@posthog/react";
import { usePathname, useSearchParams } from "next/navigation";
import posthog from "posthog-js";
import { ReactNode, useEffect, useRef } from "react";
import {
  followAnalyticsConsent,
  forgetPostHogStorageWithoutConsent,
  getConsentGatedConfig,
} from "./posthog-consent";

export function PostHogProvider({ children }: { children: ReactNode }) {
  const isPostHogEnabled = environment.isPostHogEnabled();
  const postHogCredentials = environment.getPostHogCredentials();

  useEffect(() => {
    captureFirstLanding();
    const unfollowIdentity = followAnalyticsConsentForIdentity();
    let unfollowConsent = () => {};
    if (postHogCredentials.key) {
      // Seed PostHog's anonymous identity with the first-party anonymous id
      // LaunchDarkly and the backend also use, so pre-signup activity from
      // every tool lands on the same person once identify() runs. Without
      // analytics consent that id only lives for this page load.
      const anonymousID = getAnonymousID();
      forgetPostHogStorageWithoutConsent(postHogCredentials.key);
      posthog.init(postHogCredentials.key, {
        api_host: postHogCredentials.host,
        defaults: "2025-11-30",
        capture_pageview: false,
        capture_pageleave: true,
        autocapture: true,
        ...(anonymousID && {
          bootstrap: { distinctID: anonymousID, isIdentifiedID: false },
        }),
        ...getConsentGatedConfig(),
      });
      unfollowConsent = followAnalyticsConsent();
    }
    return () => {
      unfollowConsent();
      unfollowIdentity();
    };
  }, []);

  if (!isPostHogEnabled) return <>{children}</>;

  return <PHProvider client={posthog}>{children}</PHProvider>;
}

export function PostHogUserTracker() {
  const { user, isUserLoading } = useAuth();
  const { analytics } = useConsent();
  const previousUserIdRef = useRef<string | null>(null);
  const isPostHogEnabled = environment.isPostHogEnabled();

  useEffect(() => {
    if (isUserLoading) return;
    // Identifying hands PostHog the user id, email and name (flag evaluation
    // sends them even while capture is opted out), so it waits for consent.
    if (!isPostHogEnabled || !analytics) {
      previousUserIdRef.current = null;
      return;
    }

    if (user) {
      if (previousUserIdRef.current !== user.id) {
        // Flag-only properties ride on /flags and are never stored on the
        // person profile, so identify stays exactly what analytics sends.
        if (usesPostHog()) {
          posthog.setPersonPropertiesForFlags(buildFlagPersonProperties(user));
        }
        posthog.identify(user.id, {
          email: user.email,
          ...(user.user_metadata?.name && { name: user.user_metadata.name }),
        });
        previousUserIdRef.current = user.id;
      }
    } else if (previousUserIdRef.current !== null) {
      previousUserIdRef.current = null;
    }
  }, [user, isUserLoading, isPostHogEnabled, analytics]);

  return null;
}

export function PostHogPageViewTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { analytics } = useConsent();
  const isPostHogEnabled = environment.isPostHogEnabled();

  useEffect(() => {
    if (pathname && isPostHogEnabled && analytics) {
      let url = window.origin + pathname;
      if (searchParams && searchParams.toString()) {
        url = url + `?${searchParams.toString()}`;
      }
      posthog.capture("$pageview", { $current_url: url });
    }
  }, [pathname, searchParams, isPostHogEnabled, analytics]);

  return null;
}
