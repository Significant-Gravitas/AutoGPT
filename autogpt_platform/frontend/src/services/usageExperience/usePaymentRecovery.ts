"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { getGetV2GetCopilotUsageQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { getUsageExperience } from "./helpers";
import { formatPaymentDue } from "./payment";

// The open invoice is paid in another tab, so keep checking until the
// conversion lands; the trial query already pauses while this tab is hidden.
const PAYMENT_CHECK_MS = 15_000;

export function usePaymentRecovery(tier: string | null | undefined) {
  const queryClient = useQueryClient();
  const trial = useTrialStatus({
    refetchInterval: (current) =>
      isAwaitingPayment(current) ? PAYMENT_CHECK_MS : false,
  });
  const experience = getUsageExperience(null, trial.data, tier);
  const paymentFailed = experience.inactiveTrialStatus === "payment_failed";
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
    payment: paymentFailed ? formatPaymentDue(experience.payment) : null,
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
