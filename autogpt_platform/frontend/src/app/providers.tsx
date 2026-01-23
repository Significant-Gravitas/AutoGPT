"use client";

import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import {
  PostHogPageViewTracker,
  PostHogUserTracker,
} from "@/components/monitor/PostHogUserTracker";
import { SentryUserTracker } from "@/components/monitor/SentryUserTracker";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { LaunchDarklyProvider } from "@/services/feature-flags/feature-flag-provider";
import { PostHogProvider as PHProvider } from "@posthog/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider, ThemeProviderProps } from "next-themes";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import posthog from "posthog-js";
import { Suspense, useEffect } from "react";

function PostHogProvider({ children }: { children: React.ReactNode }) {
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

export function Providers({ children, ...props }: ThemeProviderProps) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsAdapter>
        <PostHogProvider>
          <BackendAPIProvider>
            <SentryUserTracker />
            <PostHogUserTracker />
            <Suspense fallback={null}>
              <PostHogPageViewTracker />
            </Suspense>
            <CredentialsProvider>
              <LaunchDarklyProvider>
                <OnboardingProvider>
                  <ThemeProvider forcedTheme="light" {...props}>
                    <TooltipProvider>{children}</TooltipProvider>
                  </ThemeProvider>
                </OnboardingProvider>
              </LaunchDarklyProvider>
            </CredentialsProvider>
          </BackendAPIProvider>
        </PostHogProvider>
      </NuqsAdapter>
    </QueryClientProvider>
  );
}
