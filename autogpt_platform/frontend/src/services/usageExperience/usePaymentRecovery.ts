"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { getGetV2GetCopilotUsageQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetSubscriptionStatusQueryKey,
  useGetSubscriptionStatus,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { getUsageExperience } from "./helpers";
import { formatPaymentDue, formatRenewalPaymentDue } from "./payment";

// The open invoice is paid in another tab, so keep checking until the
// payment lands; both queries pause while this tab is hidden.
const PAYMENT_CHECK_MS = 15_000;

export function usePaymentRecovery(tier: string | null | undefined) {
  const queryClient = useQueryClient();
  const trial = useTrialStatus({
    refetchInterval: (current) =>
      isAwaitingPayment(current) ? PAYMENT_CHECK_MS : false,
  });
  const { data: subscription } = useGetSubscriptionStatus({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      enabled: tier === "NO_TIER",
      refetchInterval: (query) => {
        const response = query.state.data;
        return response?.status === 200 && response.data.payment_due
          ? PAYMENT_CHECK_MS
          : false;
      },
    },
  });
  const experience = getUsageExperience(null, trial.data, tier);
  const trialPaymentFailed =
    experience.inactiveTrialStatus === "payment_failed";
  // A running trial grants access, so a bill read beside it is stale.
  const renewalBill =
    !trialPaymentFailed && tier === "NO_TIER" && trial.data?.active !== true
      ? (subscription?.payment_due ?? null)
      : null;
  const paymentFailed = trialPaymentFailed || renewalBill !== null;
  const wasFailed = useRef(paymentFailed);
  useEffect(() => {
    if (wasFailed.current && !paymentFailed) {
      void queryClient.invalidateQueries({
        queryKey: getGetSubscriptionStatusQueryKey(),
      });
      void queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
    }
    wasFailed.current = paymentFailed;
  }, [paymentFailed, queryClient]);
  return {
    paymentFailed,
    payment: trialPaymentFailed
      ? formatPaymentDue(experience.payment)
      : formatRenewalPaymentDue(renewalBill),
  };
}

function isAwaitingPayment(trial: TrialStatusResponse | undefined) {
  return Boolean(
    trial &&
      !trial.active &&
      !trial.converted &&
      ["past_due", "unpaid"].includes(trial.status ?? ""),
  );
}
