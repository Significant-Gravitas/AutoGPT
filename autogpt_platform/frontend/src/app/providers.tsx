"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { TooltipProvider } from "@/components/ui/tooltip";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { LaunchDarklyProvider } from "@/components/feature-flag/feature-flag-provider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <BackendAPIProvider>
        <CredentialsProvider>
          <LaunchDarklyProvider>
            <TooltipProvider>{children}</TooltipProvider>
          </LaunchDarklyProvider>
        </CredentialsProvider>
      </BackendAPIProvider>
    </NextThemesProvider>
  );
}
