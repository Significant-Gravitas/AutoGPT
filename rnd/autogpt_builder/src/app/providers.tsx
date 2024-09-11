"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { ThemeProviderProps } from "next-themes/dist/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import SupabaseProvider from "@/components/SupabaseProvider";
import { PageViewProvider } from "@/components/providers/PageViewProvider";

export function Providers({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider {...props}>
      <SupabaseProvider>
        <PageViewProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </PageViewProvider>
      </SupabaseProvider>
    </NextThemesProvider>
  );
}
