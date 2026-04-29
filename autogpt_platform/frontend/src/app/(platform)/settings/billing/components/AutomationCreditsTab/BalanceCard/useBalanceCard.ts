"use client";

import { useState } from "react";

import {
  useGetV1GetUserCredits,
  usePostV1RequestCreditTopUp,
} from "@/app/api/__generated__/endpoints/credits/credits";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useBalanceCard() {
  const {
    data: balanceCents,
    isLoading,
    isError,
    refetch,
  } = useGetV1GetUserCredits({
    query: {
      select: (res) => {
        const raw = res.data as { credits?: number } | undefined;
        return typeof raw?.credits === "number" ? raw.credits : null;
      },
    },
  });

  const { mutateAsync: requestTopUp, isPending: isAdding } =
    usePostV1RequestCreditTopUp();

  const [open, setOpen] = useState(false);
  const [amount, setAmount] = useState("");
  const numericAmount = Number.parseFloat(amount);
  const isValid =
    Number.isFinite(numericAmount) &&
    Number.isInteger(numericAmount) &&
    numericAmount >= 5;

  async function handleSubmit() {
    if (!isValid) return;
    try {
      const result = await requestTopUp({
        data: { credit_amount: Math.round(numericAmount * 100) },
      });
      const url = (result?.data as { checkout_url?: string } | undefined)
        ?.checkout_url;
      if (url) {
        // Navigating away — don't touch React state on the unmounting tree.
        window.location.href = url;
        return;
      }
      setOpen(false);
    } catch (error) {
      toast({
        title: "Couldn't start checkout",
        description:
          error instanceof Error
            ? error.message
            : "Something went wrong contacting Stripe. Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    balanceCents: balanceCents ?? null,
    isLoading,
    isError,
    refetch,
    open,
    setOpen,
    amount,
    setAmount,
    isValid,
    isAdding,
    handleSubmit: () => void handleSubmit(),
  };
}
