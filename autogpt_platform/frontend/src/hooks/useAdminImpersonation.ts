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
