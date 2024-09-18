"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/SupabaseProvider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <TooltipProvider>{children}</TooltipProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
