"use client";

import { LaunchDarklyProvider } from "@/services/feature-flags/feature-flag-provider";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import {
  ThemeProvider as NextThemesProvider,
  ThemeProviderProps,
} from "next-themes";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import { SentryUserTracker } from "@/components/monitor/SentryUserTracker";

export function Providers({ children, ...props }: ThemeProviderProps) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsAdapter>
        <NextThemesProvider {...props}>
          <BackendAPIProvider>
            <SentryUserTracker />
            <CredentialsProvider>
              <LaunchDarklyProvider>
                <OnboardingProvider>
                  <TooltipProvider>{children}</TooltipProvider>
                </OnboardingProvider>
              </LaunchDarklyProvider>
            </CredentialsProvider>
          </BackendAPIProvider>
        </NextThemesProvider>
      </NuqsAdapter>
    </QueryClientProvider>
  );
}
