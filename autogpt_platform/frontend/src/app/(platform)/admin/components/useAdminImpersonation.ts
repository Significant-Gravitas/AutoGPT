"use client";

import { useState, useCallback } from "react";
import { ImpersonationState } from "@/lib/impersonation";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2NotifyImpersonationStart } from "@/app/api/__generated__/endpoints/admin/admin";

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
  const { mutateAsync: notifyImpersonationStart } =
    usePostV2NotifyImpersonationStart();

  const isImpersonating = Boolean(impersonatedUserId);

  const startImpersonating = useCallback(
    async (userId: string) => {
      if (!userId.trim()) {
        toast({
          title: "User ID is required for impersonation",
          variant: "destructive",
        });
        return;
      }

      // The audit alert GATES impersonation. Await it BEFORE setting state so
      // the request authenticates as the admin (the X-Act-As-User-Id header is
      // derived from sessionStorage, which isn't populated yet). The Orval
      // mutator throws on non-2xx, so a failed/blocked alert lands here and we
      // abort the swap rather than impersonate without an audit trail.
      try {
        await notifyImpersonationStart({ data: { target_user_id: userId } });
      } catch {
        toast({
          title: "Couldn't start impersonation",
          description: "Audit alert failed — impersonation blocked.",
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
    [toast, notifyImpersonationStart],
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
