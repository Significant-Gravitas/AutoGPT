"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";

import {
  getGetV1GetCreditHistoryQueryKey,
  getGetV1GetUserCreditsQueryKey,
  usePatchV1FulfillCheckoutSession,
} from "@/app/api/__generated__/endpoints/credits/credits";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { trackCheckoutAbandoned } from "@/services/analytics/monetization-analytics";

// A top-up starts on this tab, and the credits it buys show there.
const AUTOMATION_CREDITS_URL = "/settings/billing?tab=automation-credits";
// Stripe's webhook may still credit the purchase after the page's own fulfil
// call has answered, so the credits are read once more this long after it.
const LATE_WEBHOOK_MS = 10_000;

export function useTopupCheckoutReturn(topupStatus: string | null) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const identityKey = useAuthStore((state) => state.user?.id ?? null);
  const handledTopupRef = useRef<string | null>(null);
  const fulfilment = usePatchV1FulfillCheckoutSession({
    mutation: { onSettled: () => refreshCredits(queryClient) },
  });
  const { mutateAsync: fulfillCheckout } = fulfilment;
  const isFulfilmentSettled = fulfilment.isSuccess || fulfilment.isError;

  useEffect(
    function handleTopupRedirect() {
      if (!topupStatus) return;
      if (topupStatus === "success" && identityKey === null) return;
      // Stripe re-renders the page after redirect — guard so we don't
      // fire fulfillCheckout twice or stack duplicate toasts.
      if (handledTopupRef.current === topupStatus) return;
      handledTopupRef.current = topupStatus;

      if (topupStatus === "success") {
        toast({
          title: "Payment successful",
          description:
            "Your credits will appear shortly. Refresh if you don't see them.",
        });
        fulfillCheckout().catch(() => {
          // Background fulfillment call — webhook is the source of truth,
          // so a failure here is non-blocking.
        });
      } else if (topupStatus === "cancel") {
        trackCheckoutAbandoned({ checkout_kind: "top_up", surface: "billing" });
        toast({
          title: "Payment cancelled",
          description: "Your payment method was not charged.",
          variant: "destructive",
        });
      }

      router.replace(AUTOMATION_CREDITS_URL);
    },
    [topupStatus, fulfillCheckout, identityKey, router],
  );

  useEffect(
    function refreshCreditsForLateWebhook() {
      if (!isFulfilmentSettled) return;
      const timer = setTimeout(
        () => refreshCredits(queryClient),
        LATE_WEBHOOK_MS,
      );
      return () => clearTimeout(timer);
    },
    [isFulfilmentSettled, queryClient],
  );
}

function refreshCredits(queryClient: QueryClient) {
  queryClient.invalidateQueries({
    queryKey: getGetV1GetUserCreditsQueryKey(),
  });
  queryClient.invalidateQueries({
    queryKey: getGetV1GetCreditHistoryQueryKey(),
  });
}
