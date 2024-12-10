"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/providers/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { User } from "@supabase/supabase-js";
import { LaunchDarklyProvider } from "@/components/feature-flag/feature-flag-provider";

export function Providers({
  children,
  initialUser,
  ...props
}: ThemeProviderProps & { initialUser: User | null }) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider initialUser={initialUser}>
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
