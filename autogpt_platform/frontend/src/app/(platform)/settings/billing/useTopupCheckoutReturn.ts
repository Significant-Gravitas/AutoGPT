"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

import { usePatchV1FulfillCheckoutSession } from "@/app/api/__generated__/endpoints/credits/credits";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { trackCheckoutAbandoned } from "@/services/analytics/monetization-analytics";

export function useTopupCheckoutReturn(topupStatus: string | null) {
  const router = useRouter();
  const identityKey = useAuthStore((state) => state.user?.id ?? null);
  const { mutateAsync: fulfillCheckout } = usePatchV1FulfillCheckoutSession();
  const handledTopupRef = useRef<string | null>(null);

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

      router.replace("/settings/billing");
    },
    [topupStatus, fulfillCheckout, identityKey, router],
  );
}
