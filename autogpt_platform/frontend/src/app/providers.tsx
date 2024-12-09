"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { BackendAPIProvider } from "@/lib/autogpt-server-api";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { LaunchDarklyProvider } from "@/components/feature-flag/feature-flag-provider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <BackendAPIProvider>
          <CredentialsProvider>
            <LaunchDarklyProvider>
              <TooltipProvider>{children}</TooltipProvider>
            </LaunchDarklyProvider>
          </CredentialsProvider>
        </BackendAPIProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
