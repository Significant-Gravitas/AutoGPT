"use client";

import { environment } from "@/services/environment";
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

  if (environment.isLocal() && !environment.isServerSide()) {
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
