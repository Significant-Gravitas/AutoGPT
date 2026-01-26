"use client";

import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { PostHogProvider as PHProvider } from "@posthog/react";
import { usePathname, useSearchParams } from "next/navigation";
import posthog from "posthog-js";
import { ReactNode, useEffect, useRef } from "react";

export function PostHogProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_POSTHOG_KEY) {
      posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY, {
        api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
        defaults: "2025-11-30",
        capture_pageview: false,
        capture_pageleave: true,
        autocapture: true,
      });
    }
  }, []);

  return <PHProvider client={posthog}>{children}</PHProvider>;
}

export function PostHogUserTracker() {
  const { user, isUserLoading } = useSupabase();
  const previousUserIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (isUserLoading) return;

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
  }, [user, isUserLoading]);

  return null;
}

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
