"use client";

import { LaunchDarklyProvider } from "@/services/feature-flags/feature-flag-provider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import OnboardingProvider from "@/components/onboarding/onboarding-provider";
import { TooltipProvider } from "@/components/ui/tooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import {
  ThemeProvider as NextThemesProvider,
  ThemeProviderProps,
} from "next-themes";
import { NuqsAdapter } from "nuqs/adapters/next/app";

export function Providers({ children, ...props }: ThemeProviderProps) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsAdapter>
        <NextThemesProvider {...props}>
          <BackendAPIProvider>
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
