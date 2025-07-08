"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { TooltipProvider } from "@/components/ui/tooltip";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { LaunchDarklyProvider } from "@/components/feature-flag/feature-flag-provider";
import OnboardingProvider from "@/components/onboarding/onboarding-provider";
import { QueryClientProvider } from "@tanstack/react-query";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function Providers({ children, ...props }: ThemeProviderProps) {
  const queryClient = getQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
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
    </QueryClientProvider>
  );
}
