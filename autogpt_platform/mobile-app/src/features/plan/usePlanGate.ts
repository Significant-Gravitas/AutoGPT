import { skipToken, useQuery } from "@tanstack/react-query";

import {
  billingKeys,
  getPlatformConfig,
  getSubscription,
} from "@/lib/api/billing";
import { requestWithin } from "@/lib/deadline";

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
  const query = usePlanCheck(userID);
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
