"use client";

import { useEffect, useState } from "react";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

import { readTier } from "./helpers";

const POLL_MS = 2_000;
const GIVE_UP_MS = 30_000;

// Stripe returns the person before the webhook ends the trial and sets the new
// plan, so a plan bought during a cancel-pending trial is checked for until it
// lands, and no second purchase is offered meanwhile.
export function usePlanCheckoutReturn(isCheckoutReturn: boolean) {
  const [hasGivenUp, setHasGivenUp] = useState(false);
  const isWatching = isCheckoutReturn && !hasGivenUp;
  const tier = useGetSubscriptionStatus({
    query: {
      select: readTier,
      refetchInterval: (query) =>
        isWatching && readTier(query.state.data) === "TRIAL" ? POLL_MS : false,
    },
  });
  useTrialStatus({
    refetchInterval: (trial) =>
      isWatching && isCancelPending(trial) ? POLL_MS : false,
  });

  useEffect(() => {
    if (!isCheckoutReturn) return;
    const timer = setTimeout(() => setHasGivenUp(true), GIVE_UP_MS);
    return () => clearTimeout(timer);
  }, [isCheckoutReturn]);

  return {
    isConfirmingPlan: isWatching && (tier.isPending || tier.data === "TRIAL"),
  };
}

function isCancelPending(trial: TrialStatusResponse | undefined) {
  return Boolean(trial?.active && trial.cancel_at_period_end);
}
