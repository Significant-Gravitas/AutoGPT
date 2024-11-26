"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { LDProvider } from "launchdarkly-react-client-sdk";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <CredentialsProvider>
          <LDProvider clientSideID="client-side-id">
            <TooltipProvider>{children}</TooltipProvider>
          </LDProvider>
        </CredentialsProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
