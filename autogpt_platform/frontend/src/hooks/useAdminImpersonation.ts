"use client";

import { useState, useCallback } from "react";
import { environment } from "@/services/environment";
import { IMPERSONATION_STORAGE_KEY } from "@/lib/constants";

interface AdminImpersonationState {
  isImpersonating: boolean;
  impersonatedUserId: string | null;
}

interface AdminImpersonationActions {
  startImpersonating: (userId: string) => void;
  stopImpersonating: () => void;
}

type AdminImpersonationHook = AdminImpersonationState &
  AdminImpersonationActions;

/**
 * Get the initial impersonation state from sessionStorage synchronously to prevent UI flicker.
 */
function getInitialImpersonationState(): string | null {
  if (!environment.isClientSide()) {
    return null;
  }

  try {
    return sessionStorage.getItem(IMPERSONATION_STORAGE_KEY);
  } catch (error) {
    console.error("Failed to read initial impersonation state:", error);
    return null;
  }
}

/**
 * Hook for managing admin user impersonation state.
 *
 * Provides functions to start/stop impersonating users and automatically
 * sets the X-Act-As-User-Id header for API requests via sessionStorage.
 * Triggers a full page refresh to ensure all data is updated correctly.
 */
export function useAdminImpersonation(): AdminImpersonationHook {
  const [impersonatedUserId] = useState<string | null>(
    getInitialImpersonationState,
  );

  const isImpersonating = Boolean(impersonatedUserId);

  const startImpersonating = useCallback((userId: string) => {
    if (!userId.trim()) {
      console.error("Failed to start impersonation: User ID is required");
      return;
    }

    if (environment.isClientSide()) {
      try {
        sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, userId);
        window.location.reload();
      } catch (error) {
        console.error("Failed to start impersonation:", error);
      }
    }
  }, []);

  const stopImpersonating = useCallback(() => {
    if (environment.isClientSide()) {
      try {
        sessionStorage.removeItem(IMPERSONATION_STORAGE_KEY);
        window.location.reload();
      } catch (error) {
        console.error("Failed to stop impersonation:", error);
      }
    }
  }, []);

  return {
    isImpersonating,
    impersonatedUserId,
    startImpersonating,
    stopImpersonating,
  };
}
