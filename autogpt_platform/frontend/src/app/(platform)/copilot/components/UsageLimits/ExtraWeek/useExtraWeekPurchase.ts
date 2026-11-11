"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetCopilotUsageQueryKey,
  usePostV2BuyExtraWeekOfUsage,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useTopUpPrompt } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  describePurchaseFailure,
  isOutOfCredits,
  newPurchaseKey,
} from "./helpers";

interface Args {
  onBought?: () => void;
}

type Outcome = "bought" | "out_of_credits" | "failed";

/** One attempt at the purchase, told to the customer as it ends. */
async function attemptPurchase(buy: () => Promise<{ status: number }>) {
  try {
    const response = await buy();
    if (response.status !== 200) throw new Error();
  } catch (error) {
    if (isOutOfCredits(error)) {
      toast({
        title: "Not enough credits",
        description: "Add credits, then add your extra week.",
      });
      return "out_of_credits" satisfies Outcome;
    }
    toast({
      title: "Couldn’t add an extra week",
      description: describePurchaseFailure(error),
      variant: "destructive",
    });
    return "failed" satisfies Outcome;
  }
  toast({
    title: "Extra week added",
    description: "It’s used once your weekly limit runs out.",
  });
  return "bought" satisfies Outcome;
}

export function useExtraWeekPurchase({ onBought }: Args = {}) {
  const queryClient = useQueryClient();
  const { openTopUp } = useTopUpPrompt();
  const buy = usePostV2BuyExtraWeekOfUsage();
  const [isOpen, setIsOpen] = useState(false);
  // Kept until the purchase goes through, so a retry after a lost answer or
  // a top-up repeats the same purchase instead of making a second one.
  const [purchaseKey, setPurchaseKey] = useState(newPurchaseKey);

  async function confirm() {
    const outcome = await attemptPurchase(() =>
      buy.mutateAsync({ data: { purchase_key: purchaseKey } }),
    );
    if (outcome === "failed") return;
    setIsOpen(false);
    if (outcome === "out_of_credits") {
      openTopUp();
      return;
    }
    setPurchaseKey(newPurchaseKey());
    void queryClient.invalidateQueries({
      queryKey: getGetV2GetCopilotUsageQueryKey(),
    });
    if (onBought) onBought();
  }

  return {
    isOpen,
    open: () => setIsOpen(true),
    setIsOpen,
    confirm,
    isBuying: buy.isPending,
  };
}
