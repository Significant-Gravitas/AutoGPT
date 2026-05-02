"use client";

import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
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
    />
  );
}
