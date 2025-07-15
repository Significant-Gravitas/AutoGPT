"use client";

import BackendAPI from "./client";
import React, { createContext, useMemo } from "react";

// Add window.api type declaration for global access
declare global {
  interface Window {
    api?: BackendAPI;
  }
}

const BackendAPIProviderContext = createContext<BackendAPI | null>(null);

export function BackendAPIProvider({
  children,
}: {
  children?: React.ReactNode;
}): React.ReactNode {
  const api = useMemo(() => new BackendAPI(), []);

  if (
    process.env.NEXT_PUBLIC_BEHAVE_AS == "LOCAL" &&
    typeof window !== "undefined"
  ) {
    window.api = api; // Expose the API globally for debugging purposes
  }

  return (
    <BackendAPIProviderContext.Provider value={api}>
      {children}
    </BackendAPIProviderContext.Provider>
  );
}

export function useBackendAPI(): BackendAPI {
  const context = React.useContext(BackendAPIProviderContext);
  if (!context) {
    throw new Error(
      "useBackendAPI must be used within a BackendAPIProviderContext",
    );
  }
  return context;
}
