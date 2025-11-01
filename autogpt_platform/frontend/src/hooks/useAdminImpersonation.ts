"use client";

import { useState, useCallback, useEffect } from "react";
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
 * Hook for managing admin user impersonation state.
 *
 * Provides functions to start/stop impersonating users and automatically
 * sets the X-Act-As-User-Id header for API requests via sessionStorage.
 * Triggers a full page refresh to ensure all data is updated correctly.
 */
export function useAdminImpersonation(): AdminImpersonationHook {
  const [impersonatedUserId, setImpersonatedUserId] = useState<string | null>(
    null,
  );

  const isImpersonating = Boolean(impersonatedUserId);

  const startImpersonating = useCallback((userId: string) => {
    if (!userId.trim()) {
      throw new Error("User ID is required for impersonation");
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

  // Restore impersonation state from sessionStorage on mount
  useEffect(() => {
    if (environment.isClientSide()) {
      try {
        const storedUserId = sessionStorage.getItem(IMPERSONATION_STORAGE_KEY);
        if (storedUserId) {
          setImpersonatedUserId(storedUserId);
        }
      } catch (error) {
        console.error("Failed to restore impersonation state:", error);
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
