"use client";

import { useState } from "react";

import {
  useGetV1GetUserCredits,
  usePostV1RequestCreditTopUp,
} from "@/app/api/__generated__/endpoints/credits/credits";

export function useBalanceCard() {
  const { data: balanceCents, isLoading } = useGetV1GetUserCredits({
    query: {
      select: (res) => {
        const raw = res.data as { credits?: number } | undefined;
        return typeof raw?.credits === "number" ? raw.credits : 0;
      },
    },
  });

  const { mutateAsync: requestTopUp, isPending: isAdding } =
    usePostV1RequestCreditTopUp();

  const [open, setOpen] = useState(false);
  const [amount, setAmount] = useState("");
  const numericAmount = Number.parseFloat(amount);
  const isValid = Number.isFinite(numericAmount) && numericAmount >= 5;

  async function handleSubmit() {
    if (!isValid) return;
    const result = await requestTopUp({
      data: { credit_amount: Math.round(numericAmount * 100) },
    });
    const url = (result?.data as { checkout_url?: string } | undefined)
      ?.checkout_url;
    if (url) window.location.href = url;
    setOpen(false);
  }

  return {
    balanceCents: balanceCents ?? 0,
    isLoading,
    open,
    setOpen,
    amount,
    setAmount,
    isValid,
    isAdding,
    handleSubmit: () => void handleSubmit(),
  };
}
