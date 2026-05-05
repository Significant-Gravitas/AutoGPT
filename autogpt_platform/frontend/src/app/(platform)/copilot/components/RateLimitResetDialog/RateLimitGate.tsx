"use client";

import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import useCredits from "@/hooks/useCredits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect } from "react";
import { RateLimitResetDialog } from "./RateLimitResetDialog";

interface Props {
  rateLimitMessage: string | null;
  onDismiss: () => void;
}

/**
 * Renders the rate-limit dialog when the user has an actionable reset path
 * (positive reset_cost). Otherwise falls back to a toast so the user still
 * gets feedback. Encapsulates all the usage/credits/flag state that's only
 * needed once a rate limit is hit.
 */
export function RateLimitGate({ rateLimitMessage, onDismiss }: Props) {
  const {
    data: usage,
    isSuccess: hasUsage,
    isError: usageError,
  } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      // Only fetch once a rate limit has been hit — avoids a 30s background
      // poll for the 99% of sessions that never hit their quota.
      enabled: !!rateLimitMessage,
      refetchInterval: 30_000,
      staleTime: 10_000,
    },
  });
  const resetCost = usage?.reset_cost;
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { credits, fetchCredits } = useCredits({ fetchInitialCredits: true });
  const hasInsufficientCredits =
    credits !== null && resetCost != null && credits < resetCost;

  // When the credit-based reset is unavailable (feature disabled or the query
  // failed), fall back to a toast so the user still gets feedback.
  useEffect(() => {
    if (!rateLimitMessage) return;
    if (!usageError && !(hasUsage && (resetCost ?? 0) <= 0)) return;
    toast({
      title: "Usage limit reached",
      description: rateLimitMessage,
      variant: "destructive",
    });
    onDismiss();
  }, [rateLimitMessage, resetCost, hasUsage, usageError, onDismiss]);

  const canShowDialog = !!rateLimitMessage && hasUsage && (resetCost ?? 0) > 0;

  return (
    <RateLimitResetDialog
      isOpen={canShowDialog}
      onClose={onDismiss}
      resetCost={resetCost ?? 0}
      resetMessage={rateLimitMessage ?? ""}
      isWeeklyExhausted={
        hasUsage && !!usage.weekly && usage.weekly.percent_used >= 100
      }
      hasInsufficientCredits={hasInsufficientCredits}
      isBillingEnabled={isBillingEnabled}
      onCreditChange={fetchCredits}
    />
  );
}
