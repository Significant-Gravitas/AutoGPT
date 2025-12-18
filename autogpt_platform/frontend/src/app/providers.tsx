"use client";

import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { SentryUserTracker } from "@/components/monitor/SentryUserTracker";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { LaunchDarklyProvider } from "@/services/feature-flags/feature-flag-provider";
import { QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider, ThemeProviderProps } from "next-themes";
import { NuqsAdapter } from "nuqs/adapters/next/app";

export function Providers({ children, ...props }: ThemeProviderProps) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsAdapter>
        <BackendAPIProvider>
          <SentryUserTracker />
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
      </NuqsAdapter>
    </QueryClientProvider>
  );
}
