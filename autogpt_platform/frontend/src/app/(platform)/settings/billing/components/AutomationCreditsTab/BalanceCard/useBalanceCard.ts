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
  // Backend `top_up_intent` rejects non-whole-dollar amounts
  // (`amount % 100 != 0`), so gate decimals client-side to surface the
  // error before checkout instead of after a failed Stripe redirect.
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
      const status = (result as { status?: number } | undefined)?.status;
      const body = result?.data as
        | { checkout_url?: string; detail?: string | { msg?: string } }
        | undefined;

      if (status && status >= 400) {
        // Surface backend error detail (FastAPI conventionally returns
        // ``{"detail": "..."}``) instead of the generic "no URL" message —
        // the actual failure is almost always a Stripe-side error
        // (stale customer ID, missing product, invalid API key) that the
        // user / their team needs to see to fix.
        const detail =
          typeof body?.detail === "string"
            ? body.detail
            : (body?.detail?.msg ?? `Server returned ${status}.`);
        throw new Error(detail);
      }

      const url = body?.checkout_url;
      if (url) {
        // Navigating away — don't touch React state on the unmounting tree.
        window.location.href = url;
        return;
      }

      throw new Error(
        "Stripe didn't return a checkout URL. Check backend logs for the underlying Stripe error.",
      );
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
