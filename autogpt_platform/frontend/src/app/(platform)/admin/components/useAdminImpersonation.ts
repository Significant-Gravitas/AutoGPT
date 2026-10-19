"use client";

import { useCallback, useRef } from "react";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { ImpersonationState } from "@/lib/impersonation";
import {
  beginImpersonation,
  type BeginImpersonationOutcome,
} from "./beginImpersonation";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2NotifyImpersonationStart } from "@/app/api/__generated__/endpoints/admin/admin";

interface AdminImpersonationState {
  isImpersonating: boolean;
  impersonatedUserId: string | null;
  isStarting: boolean;
}

interface AdminImpersonationActions {
  startImpersonating: (userId: string) => Promise<void>;
  stopImpersonating: () => void;
}

type AdminImpersonationHook = AdminImpersonationState &
  AdminImpersonationActions;

const OUTCOME_MESSAGES: Record<
  Exclude<BeginImpersonationOutcome, "started">,
  string
> = {
  "no-admin": "Sign in again to impersonate a user.",
  "audit-failed": "Audit alert failed — impersonation blocked.",
  "admin-changed":
    "You were signed out or your account changed while it was starting.",
};

export function useAdminImpersonation(): AdminImpersonationHook {
  // Re-read whenever the signed-in user changes: after the reload that starts
  // impersonation, the target only counts once the owning admin is known.
  const viewer = useAuthStore((state) => state.user);
  const impersonatedUserId = viewer ? ImpersonationState.get() : null;
  const { toast } = useToast();
  const { mutateAsync: notifyImpersonationStart, isPending: isStarting } =
    usePostV2NotifyImpersonationStart();
  // Synchronous re-entry guard: `isStarting` only flips on re-render, so rapid
  // double-clicks could otherwise fire two audit alerts for one impersonation.
  const startingRef = useRef(false);

  const isImpersonating = Boolean(impersonatedUserId);

  const startImpersonating = useCallback(
    async (userId: string) => {
      if (startingRef.current) return;

      if (!userId.trim()) {
        toast({
          title: "User ID is required for impersonation",
          variant: "destructive",
        });
        return;
      }

      startingRef.current = true;
      try {
        // The audit alert GATES impersonation and is sent before any target
        // is stored, so it authenticates as the admin. The Orval mutator
        // throws on non-2xx, so a failed/blocked alert aborts the swap.
        const outcome = await beginImpersonation(userId, () =>
          notifyImpersonationStart({ data: { target_user_id: userId } }),
        );
        if (outcome !== "started") {
          toast({
            title: "Couldn't start impersonation",
            description: OUTCOME_MESSAGES[outcome],
            variant: "destructive",
          });
          return;
        }

        window.location.reload();
      } catch (error) {
        console.error("Failed to start impersonation:", error);
        toast({
          title: "Failed to start impersonation",
          description: error instanceof Error ? error.message : "Unknown error",
          variant: "destructive",
        });
      } finally {
        startingRef.current = false;
      }
    },
    [toast, notifyImpersonationStart],
  );

  const stopImpersonating = useCallback(() => {
    try {
      ImpersonationState.clear();
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
    isStarting,
    startImpersonating,
    stopImpersonating,
  };
}
