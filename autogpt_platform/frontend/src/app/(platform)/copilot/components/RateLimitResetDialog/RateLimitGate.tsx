"use client";

import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect } from "react";
import { RateLimitResetDialog } from "./RateLimitResetDialog";

interface Props {
  rateLimitMessage: string | null;
  onDismiss: () => void;
}

/**
 * Renders the rate-limit dialog when the user hits their daily limit.
 * Falls back to a toast when the usage query fails.
 */
export function RateLimitGate({ rateLimitMessage, onDismiss }: Props) {
  const {
    data: usage,
    isSuccess: hasUsage,
    isError: usageError,
  } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      enabled: !!rateLimitMessage,
      refetchInterval: 30_000,
      staleTime: 10_000,
    },
  });

  // Pulls the user's current tier so the dialog can branch the CTA between
  // "Upgrade plan" (route to /settings/billing) and "Contact us" (top-tier
  // users have no higher self-serve plan to upgrade to).
  const { data: tier } = useGetSubscriptionStatus({
    query: {
      enabled: !!rateLimitMessage,
      select: (res) =>
        res.status === 200
          ? ((res.data as SubscriptionStatusResponse).tier as SubscriptionTier)
          : null,
    },
  });

  useEffect(() => {
    if (!rateLimitMessage) return;
    if (!usageError) return;
    toast({
      title: "Usage limit reached",
      description: rateLimitMessage,
      variant: "destructive",
    });
    onDismiss();
  }, [rateLimitMessage, usageError, onDismiss]);

  const isOpen = !!rateLimitMessage && hasUsage;

  return (
    <RateLimitResetDialog
      isOpen={isOpen}
      onClose={onDismiss}
      resetsAt={usage?.daily?.resets_at ?? usage?.weekly?.resets_at ?? null}
      tier={tier ?? null}
    />
  );
}
