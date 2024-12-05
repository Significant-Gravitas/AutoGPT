"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { BackendAPIProvider } from "@/lib/autogpt-server-api";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/providers/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { User } from "@supabase/supabase-js";

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
            <TooltipProvider>{children}</TooltipProvider>
          </CredentialsProvider>
        </BackendAPIProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
