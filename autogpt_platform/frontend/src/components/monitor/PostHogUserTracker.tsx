"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { usePathname, useSearchParams } from "next/navigation";
import posthog from "posthog-js";
import { useEffect, useRef } from "react";

/**
 * PostHogUserTracker component identifies users in PostHog for analytics.
 * This component should be placed high in the component tree to ensure user
 * identification happens as soon as the user logs in.
 *
 * It automatically:
 * - Identifies the user when they log in (linking anonymous to authenticated)
 * - Resets PostHog when a user logs out
 * - Updates identification when user data changes
 */
export function PostHogUserTracker() {
  const { user, isUserLoading } = useSupabase();
  const previousUserIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (isUserLoading) return;

    if (user) {
      // Only identify if we haven't already identified this user
      if (previousUserIdRef.current !== user.id) {
        posthog.identify(user.id, {
          email: user.email,
          ...(user.user_metadata?.name && { name: user.user_metadata.name }),
        });
        previousUserIdRef.current = user.id;
      }
    } else if (previousUserIdRef.current !== null) {
      // User logged out - reset PostHog to generate new anonymous ID
      posthog.reset();
      previousUserIdRef.current = null;
    }
  }, [user, isUserLoading]);

  return null;
}

/**
 * PostHogPageViewTracker captures page views on route changes in Next.js App Router.
 * The default PostHog capture_pageview only works for initial page loads.
 * This component ensures soft navigations (client-side route changes) are also tracked.
 */
export function PostHogPageViewTracker() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (pathname) {
      let url = window.origin + pathname;
      if (searchParams && searchParams.toString()) {
        url = url + `?${searchParams.toString()}`;
      }
      posthog.capture("$pageview", { $current_url: url });
    }
  }, [pathname, searchParams]);

  return null;
}
