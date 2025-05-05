"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { TooltipProvider } from "@/components/ui/tooltip";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { LaunchDarklyProvider } from "@/components/feature-flag/feature-flag-provider";
import OnboardingProvider from "@/components/onboarding/onboarding-provider";
import { MockClientProps } from "@/lib/autogpt-server-api/mock_client";
import PageStructureContainer from "@/components/page-structure-container-provider";

export interface ProvidersProps extends ThemeProviderProps {
  children: React.ReactNode;
  useMockBackend?: boolean;
  mockClientProps?: MockClientProps;
}

export function Providers({
  children,
  useMockBackend,
  mockClientProps,
  ...props
}: ProvidersProps) {
  return (
    <NextThemesProvider {...props}>
      <BackendAPIProvider mockClientProps={mockClientProps}>
        <CredentialsProvider>
          <LaunchDarklyProvider>
            <OnboardingProvider>
              <TooltipProvider>
                <PageStructureContainer>{children}</PageStructureContainer>
              </TooltipProvider>
            </OnboardingProvider>
          </LaunchDarklyProvider>
        </CredentialsProvider>
      </BackendAPIProvider>
    </NextThemesProvider>
  );
}
