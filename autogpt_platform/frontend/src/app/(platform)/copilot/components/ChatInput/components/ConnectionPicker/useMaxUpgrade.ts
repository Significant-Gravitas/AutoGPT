"use client";

import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { getGetV2ListChatConnectionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetSubscriptionStatusQueryKey,
  useGetSubscriptionStatus,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";

import {
  getMaxUpgradePricing,
  getMaxUpgradeUnavailableReason,
} from "./maxUpgrade";

export function useMaxUpgrade(enabled: boolean) {
  const client = useQueryClient();
  const query = useGetSubscriptionStatus({ query: { enabled, staleTime: 0 } });
  const mutation = useUpdateSubscriptionTier();
  const inFlight = useRef(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const subscription = query.data?.status === 200 ? query.data.data : undefined;
  const pricing = getMaxUpgradePricing(subscription);
  const unavailableReason = getMaxUpgradeUnavailableReason(subscription);
  const isError =
    query.isError || (query.data !== undefined && query.data.status !== 200);
  const canUpgrade = enabled && !isError && unavailableReason === null;

  function resetError() {
    setError(null);
    mutation.reset();
  }

  async function upgrade() {
    if (!canUpgrade || inFlight.current || !pricing) return false;
    inFlight.current = true;
    setIsPending(true);
    setError(null);
    try {
      const refreshed = await query.refetch();
      const latest =
        refreshed.data?.status === 200 ? refreshed.data.data : undefined;
      if (refreshed.isError || !latest) {
        throw new Error("Couldn't check your current plan. Please try again.");
      }
      const reason = getMaxUpgradeUnavailableReason(latest);
      if (reason) throw new Error(reason);
      const latestPricing = getMaxUpgradePricing(latest);
      if (
        latestPricing?.cycle !== pricing.cycle ||
        latestPricing.maxCents !== pricing.maxCents ||
        latestPricing.currentCents !== pricing.currentCents
      ) {
        throw new Error(
          "Your plan details changed. Review the updated price before confirming.",
        );
      }
      const result = await mutation.mutateAsync({
        data: { tier: "MAX", billing_cycle: pricing.cycle },
      });
      if (result.status !== 200 || result.data.url) {
        throw new Error(
          "This upgrade needs to be completed in billing. Your chat is still here.",
        );
      }
      client.setQueryData(getGetSubscriptionStatusQueryKey(), result);
      await Promise.all([
        client.invalidateQueries({
          queryKey: getGetSubscriptionStatusQueryKey(),
        }),
        client.invalidateQueries({
          queryKey: getGetV2ListChatConnectionsQueryKey(),
        }),
      ]);
      return true;
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Couldn't update your plan. Please try again.",
      );
      return false;
    } finally {
      inFlight.current = false;
      setIsPending(false);
    }
  }

  function retry() {
    resetError();
    void query.refetch();
  }

  return {
    subscription,
    pricing,
    canUpgrade,
    unavailableReason,
    isLoading: enabled && query.isLoading,
    isError,
    isPending,
    error,
    resetError,
    retry,
    upgrade,
  };
}
