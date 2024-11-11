"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/providers/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";
import { Session } from "@supabase/supabase-js";

export function Providers({
  children,
  initialSession,
  ...props
}: ThemeProviderProps & { initialSession: Session | null }) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider initialSession={initialSession}>
        <CredentialsProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </CredentialsProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
