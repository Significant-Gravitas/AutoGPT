"use client";

import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { SentryUserTracker } from "@/components/monitor/SentryUserTracker";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import OrgTeamProvider from "@/providers/org-team/OrgTeamProvider";
import {
  PostHogPageViewTracker,
  PostHogProvider,
  PostHogUserTracker,
} from "@/providers/posthog/posthog-provider";
import { AdsConversionTracker } from "@/services/analytics/AdsConversionTracker";
import { AttributionReporter } from "@/services/analytics/AttributionReporter";
import { LaunchDarklyProvider } from "@/services/feature-flags/feature-flag-provider";
import { QueryClientProvider } from "@tanstack/react-query";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ReactNode, Suspense } from "react";

interface Props {
  children: ReactNode;
}

// Light is the only theme: nothing adds the `.dark` class that switches the
// semantic variables in globals.css. Dark mode needs a class toggle here
// (next-themes with attribute="class" fits) once its values are signed off.
export function Providers({ children }: Props) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsAdapter>
        <PostHogProvider>
          <BackendAPIProvider>
            {/* All four read useSearchParams (directly or via useAuth), which
                bails out of static rendering unless it sits under Suspense. */}
            <Suspense fallback={null}>
              <SentryUserTracker />
              <PostHogUserTracker />
              <AttributionReporter />
              <AdsConversionTracker />
              <PostHogPageViewTracker />
            </Suspense>
            <CredentialsProvider>
              <OrgTeamProvider>
                <LaunchDarklyProvider>
                  <OnboardingProvider>
                    <TooltipProvider>{children}</TooltipProvider>
                  </OnboardingProvider>
                </LaunchDarklyProvider>
              </OrgTeamProvider>
            </CredentialsProvider>
          </BackendAPIProvider>
        </PostHogProvider>
      </NuqsAdapter>
    </QueryClientProvider>
  );
}
