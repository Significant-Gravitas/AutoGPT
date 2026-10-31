import {
  getGetSubscriptionStatusQueryKey,
  useGetSubscriptionStatus,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export function usePaidCheckoutStatus(
  userID: string | null,
  paymentEnabled: boolean,
) {
  const params = useSearchParams();
  const returning = useRef(params.get("subscription") === "success");
  const previousUserID = useRef(userID);
  if (previousUserID.current && previousUserID.current !== userID)
    returning.current = false;
  previousUserID.current = userID;
  const [timedOut, setTimedOut] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const waitingForPayment = returning.current && paymentEnabled;
  const query = useGetSubscriptionStatus({
    query: {
      enabled: !!userID,
      queryKey: [...getGetSubscriptionStatusQueryKey(), userID],
      select: (res) =>
        res.status === 200
          ? (res.data as SubscriptionStatusResponse).tier
          : null,
      refetchInterval: (state) => {
        const response = state.state.data;
        const confirmed =
          response?.status === 200 && response.data.tier !== "NO_TIER";
        return waitingForPayment && !timedOut && !confirmed ? 1000 : false;
      },
    },
  });
  const active = !!query.data && query.data !== "NO_TIER";
  const unconfirmed = waitingForPayment && !active;
  useEffect(() => {
    if (!unconfirmed) return;
    const timer = setTimeout(() => setTimedOut(true), 15_000);
    return () => clearTimeout(timer);
  }, [unconfirmed, attempt]);

  function retry() {
    setTimedOut(false);
    setAttempt((value) => value + 1);
    void query.refetch();
  }
  return {
    active,
    isLoading: query.isLoading,
    ready: !unconfirmed,
    pending: unconfirmed && !timedOut,
    error:
      unconfirmed && timedOut
        ? "Your payment is still being confirmed. Please retry to check its status."
        : null,
    retry,
  };
}
