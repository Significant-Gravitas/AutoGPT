"use client";

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetCopilotUsageQueryKey,
  usePostV2BuyUsageReset,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { UsageResetResponse } from "@/app/api/__generated__/models/usageResetResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { formatExactReset } from "@/services/usageExperience/resetTime";
import {
  describePurchaseFailure,
  grantedMessage,
  isPriceChanged,
  newPurchaseKey,
} from "./helpers";

interface Args {
  /** The price the customer is shown, which is the price they agree to. */
  priceCredits: number;
  onBought?: () => void;
  /** Leaves for Stripe's checkout; replaced in tests. */
  goToCheckout?: (url: string) => void;
}

type Outcome = "bought" | "checkout" | "price_changed" | "failed";

function leaveForCheckout(url: string) {
  window.location.assign(url);
}

/** One attempt at the purchase, told to the customer as it ends. */
async function attemptPurchase(
  buy: () => Promise<{ status: number; data: unknown }>,
  goToCheckout: (url: string) => void,
) {
  try {
    const response = await buy();
    if (response.status !== 200) throw new Error();
    const answer = response.data as UsageResetResponse;
    if (answer.status === "payment_required") {
      if (!answer.checkout_url) throw new Error();
      // Nothing is reset until the card payment goes through.
      goToCheckout(answer.checkout_url);
      return "checkout" satisfies Outcome;
    }
    toast({
      title: "Your limits are reset",
      description: grantedMessage(formatExactReset(answer.resets_at)),
    });
    return "bought" satisfies Outcome;
  } catch (error) {
    toast({
      title: "Couldn’t reset your limits",
      description: describePurchaseFailure(error),
      variant: "destructive",
    });
    return (
      isPriceChanged(error) ? "price_changed" : "failed"
    ) satisfies Outcome;
  }
}

export function usePaidResetPurchase({
  priceCredits,
  onBought,
  goToCheckout = leaveForCheckout,
}: Args) {
  const queryClient = useQueryClient();
  const buy = usePostV2BuyUsageReset();
  const [isOpen, setIsOpen] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  // Kept until the purchase goes through, so a retry after a lost answer
  // repeats the same purchase instead of making a second one.
  const [purchaseKey, setPurchaseKey] = useState(newPurchaseKey);

  useEffect(function reenableAfterComingBackFromCheckout() {
    // "Back" from Stripe can restore this page as it was left, busy button
    // included.
    function onPageShow(event: PageTransitionEvent) {
      if (event.persisted) setIsLeaving(false);
    }
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, []);

  async function confirm() {
    const outcome = await attemptPurchase(
      () =>
        buy.mutateAsync({
          data: { purchase_key: purchaseKey, price_credits: priceCredits },
        }),
      goToCheckout,
    );
    if (outcome === "failed") return;
    if (outcome === "checkout") {
      // The page is about to be replaced by Stripe's: keep the button busy.
      setIsLeaving(true);
      return;
    }
    setIsOpen(false);
    // Bought, or the price moved: either way the offer on screen is stale.
    void queryClient.invalidateQueries({
      queryKey: getGetV2GetCopilotUsageQueryKey(),
    });
    if (outcome === "price_changed") return;
    setPurchaseKey(newPurchaseKey());
    if (onBought) onBought();
  }

  return {
    isOpen,
    open: () => setIsOpen(true),
    setIsOpen,
    confirm,
    isBuying: buy.isPending || isLeaving,
  };
}
