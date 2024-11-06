"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/components/providers/AuthContext";
import SupabaseProvider from "@/components/providers/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <AuthProvider>
          <CredentialsProvider>
            <TooltipProvider>{children}</TooltipProvider>
          </CredentialsProvider>
        </AuthProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
