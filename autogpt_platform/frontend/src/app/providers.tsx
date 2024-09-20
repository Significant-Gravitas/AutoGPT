"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/SupabaseProvider";
import CredentialsProvider from "@/components/integrations/credentials-provider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <CredentialsProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </CredentialsProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
