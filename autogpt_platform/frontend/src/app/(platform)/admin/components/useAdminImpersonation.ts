"use client";

import { useState, useCallback } from "react";
import { environment } from "@/services/environment";
import { IMPERSONATION_STORAGE_KEY } from "@/lib/constants";
import { useToast } from "@/components/molecules/Toast/use-toast";

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
  const [impersonatedUserId, setImpersonatedUserId] = useState<string | null>(
    getInitialImpersonationState,
  );
  const { toast } = useToast();

  const isImpersonating = Boolean(impersonatedUserId);

  const startImpersonating = useCallback(
    (userId: string) => {
      if (!userId.trim()) {
        toast({
          title: "User ID is required for impersonation",
          variant: "destructive",
        });
        return;
      }

      if (environment.isClientSide()) {
        try {
          sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, userId);
          setImpersonatedUserId(userId);
          window.location.reload();
        } catch (error) {
          console.error("Failed to start impersonation:", error);
          toast({
            title: "Failed to start impersonation",
            description:
              error instanceof Error ? error.message : "Unknown error",
            variant: "destructive",
          });
        }
      }
    },
    [toast],
  );

  const stopImpersonating = useCallback(() => {
    if (environment.isClientSide()) {
      try {
        sessionStorage.removeItem(IMPERSONATION_STORAGE_KEY);
        setImpersonatedUserId(null);
        window.location.reload();
      } catch (error) {
        console.error("Failed to stop impersonation:", error);
        toast({
          title: "Failed to stop impersonation",
          description: error instanceof Error ? error.message : "Unknown error",
          variant: "destructive",
        });
      }
    }
  }, [toast]);

  return {
    isImpersonating,
    impersonatedUserId,
    startImpersonating,
    stopImpersonating,
  };
}
