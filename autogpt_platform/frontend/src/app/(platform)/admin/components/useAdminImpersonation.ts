"use client";

import { useState, useCallback } from "react";
import { ImpersonationState } from "@/lib/impersonation";
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

export function useAdminImpersonation(): AdminImpersonationHook {
  const [impersonatedUserId, setImpersonatedUserId] = useState<string | null>(
    ImpersonationState.get,
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

      try {
        ImpersonationState.set(userId);
        setImpersonatedUserId(userId);
        window.location.reload();
      } catch (error) {
        console.error("Failed to start impersonation:", error);
        toast({
          title: "Failed to start impersonation",
          description: error instanceof Error ? error.message : "Unknown error",
          variant: "destructive",
        });
      }
    },
    [toast],
  );

  const stopImpersonating = useCallback(() => {
    try {
      ImpersonationState.clear();
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
  }, [toast]);

  return {
    isImpersonating,
    impersonatedUserId,
    startImpersonating,
    stopImpersonating,
  };
}
