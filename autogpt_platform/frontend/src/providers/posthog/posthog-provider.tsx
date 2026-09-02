"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import {
  captureFirstLanding,
  getAnonymousID,
} from "@/services/analytics/anonymous-id";
import { environment } from "@/services/environment";
import { PostHogProvider as PHProvider } from "@posthog/react";
import { usePathname, useSearchParams } from "next/navigation";
import posthog from "posthog-js";
import { ReactNode, useEffect, useRef } from "react";

export function PostHogProvider({ children }: { children: ReactNode }) {
  const isPostHogEnabled = environment.isPostHogEnabled();
  const postHogCredentials = environment.getPostHogCredentials();

  useEffect(() => {
    if (postHogCredentials.key) {
      // Seed PostHog's anonymous identity with the first-party anonymous id
      // LaunchDarkly and the backend also use, so pre-signup activity from
      // every tool lands on the same person once identify() runs.
      const anonymousID = getAnonymousID();
      posthog.init(postHogCredentials.key, {
        api_host: postHogCredentials.host,
        defaults: "2025-11-30",
        capture_pageview: false,
        capture_pageleave: true,
        autocapture: true,
        ...(anonymousID && {
          bootstrap: { distinctID: anonymousID, isIdentifiedID: false },
        }),
      });
    }
    captureFirstLanding();
  }, []);

  if (!isPostHogEnabled) return <>{children}</>;

  return <PHProvider client={posthog}>{children}</PHProvider>;
}

export function PostHogUserTracker() {
  const { user, isUserLoading } = useAuth();
  const previousUserIdRef = useRef<string | null>(null);
  const isPostHogEnabled = environment.isPostHogEnabled();

  useEffect(() => {
    if (isUserLoading || !isPostHogEnabled) return;

    if (user) {
      if (previousUserIdRef.current !== user.id) {
        posthog.identify(user.id, {
          email: user.email,
          ...(user.user_metadata?.name && { name: user.user_metadata.name }),
        });
        previousUserIdRef.current = user.id;
      }
    } else if (previousUserIdRef.current !== null) {
      posthog.reset();
      previousUserIdRef.current = null;
    }
  }, [user, isUserLoading, isPostHogEnabled]);

  return null;
}

export function PostHogPageViewTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const isPostHogEnabled = environment.isPostHogEnabled();

  useEffect(() => {
    if (pathname && isPostHogEnabled) {
      let url = window.origin + pathname;
      if (searchParams && searchParams.toString()) {
        url = url + `?${searchParams.toString()}`;
      }
      posthog.capture("$pageview", { $current_url: url });
    }
  }, [pathname, searchParams, isPostHogEnabled]);

  return null;
}
