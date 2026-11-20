import { useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  billingKeys,
  getChatUsage,
  getCredits,
  getSubscription,
  getTrial,
} from "@/lib/api/billing";

import { formatMoney, planSummary, usageMeters } from "./helpers";

export function usePlanScreen() {
  const subscription = useQuery({
    queryKey: billingKeys.subscription,
    queryFn: () => getSubscription(),
  });
  const isTrial = subscription.data?.tier === "TRIAL";
  const trial = useQuery({
    queryKey: billingKeys.trial,
    queryFn: getTrial,
    enabled: isTrial,
  });
  const usage = useQuery({ queryKey: billingKeys.usage, queryFn: getChatUsage });
  const credits = useQuery({ queryKey: billingKeys.credits, queryFn: getCredits });

  // A trial's dates come from the trial; wait for it rather than show none.
  const isLoading = subscription.isPending || (isTrial && trial.isPending);

  async function refresh() {
    await Promise.all([
      subscription.refetch(),
      usage.refetch(),
      credits.refetch(),
      ...(isTrial ? [trial.refetch()] : []),
    ]);
  }

  function goBack() {
    router.navigate("/settings");
  }

  return {
    summary: subscription.data ? planSummary(subscription.data, trial.data) : null,
    meters: usageMeters(usage.data, trial.data),
    balance: credits.data === undefined ? null : formatMoney(credits.data),
    isLoading,
    isError: subscription.isError,
    isRefreshing: subscription.isRefetching,
    refresh,
    goBack,
  };
}
