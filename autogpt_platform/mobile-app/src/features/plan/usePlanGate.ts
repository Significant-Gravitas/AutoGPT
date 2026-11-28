import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppState } from "react-native";

import {
  billingKeys,
  getPlatformConfig,
  getSubscription,
} from "@/lib/api/billing";
import { requestWithin } from "@/lib/deadline";
import { useMountEffect } from "@/lib/useMountEffect";

import {
  PLAN_CHECK_TIMEOUT_MS,
  planGate,
  readPlanCheck,
  shouldRetryPlanCheck,
} from "./gate";

// Matches the 503's Retry-After.
const ACTIVATION_RETRY_DELAY_MS = 3_000;

/** Asks the server whether this account has a plan, after sign-in and at every launch. */
export function usePlanGate(userID: string | undefined) {
  const queryClient = useQueryClient();
  const query = usePlanCheck(userID);

  useMountEffect(() => {
    // Back from the browser, or after a plan ended in the background: ask
    // again. The last answer holds meanwhile, so nothing flashes.
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active") {
        void queryClient.invalidateQueries({ queryKey: billingKeys.checks });
      }
    });
    return () => subscription.remove();
  });

  return planGate({ check: query.data, status: query.status });
}

/** The one answer the gate, the no-plan screen and Settings all read. */
export function usePlanCheck(userID: string | undefined) {
  return useQuery({
    queryKey: billingKeys.check(userID ?? ""),
    queryFn: userID ? checkPlan : skipToken,
    retry: shouldRetryPlanCheck,
    retryDelay: ACTIVATION_RETRY_DELAY_MS,
    staleTime: Infinity,
  });
}

async function checkPlan() {
  const config = await requestWithin(PLAN_CHECK_TIMEOUT_MS, (signal) =>
    getPlatformConfig({ signal }),
  );
  if (!config.payment_enabled) return readPlanCheck(false, null);
  const subscription = await requestWithin(PLAN_CHECK_TIMEOUT_MS, (signal) =>
    getSubscription({ signal }),
  );
  return readPlanCheck(true, subscription.tier);
}
