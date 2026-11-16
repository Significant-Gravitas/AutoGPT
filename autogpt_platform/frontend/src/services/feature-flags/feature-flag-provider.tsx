"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import type { ReactNode } from "react";
import { FlagResolutionContext } from "./flag-resolution";

export function FeatureFlagProvider({ children }: { children: ReactNode }) {
  const { isUserLoading } = useAuth();

  // The server reads flags for the signed-in user, so the wait starts once the
  // session is known. The page renders regardless so crawlers get its content.
  return (
    <FlagResolutionContext.Provider value={!isUserLoading}>
      {children}
    </FlagResolutionContext.Provider>
  );
}
