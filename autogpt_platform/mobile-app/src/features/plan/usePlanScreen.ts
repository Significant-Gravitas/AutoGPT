import { useQuery, useQueryClient } from "@tanstack/react-query";
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
  const queryClient = useQueryClient();
  const subscription = useQuery({
    queryKey: billingKeys.subscription,
    queryFn: async () => {
      const answer = await getSubscription();
      // The plan ended since launch: the gate asks again and shows the no-plan screen.
      if (answer.tier === "NO_TIER") {
        void queryClient.invalidateQueries({ queryKey: billingKeys.checks });
      }
      return answer;
    },
  });
  const isTrial = subscription.data?.tier === "TRIAL";
  const trial = useQuery({
    queryKey: billingKeys.trial,
    queryFn: getTrial,
    enabled: isTrial,
  });
  const usage = useQuery({ queryKey: billingKeys.usage, queryFn: getChatUsage });
  const credits = useQuery({ queryKey: billingKeys.credits, queryFn: getCredits });

  // Kept from before an upgrade, a trial's answer must not reach a paid plan.
  const trialData = isTrial ? trial.data : undefined;
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
    summary: subscription.data ? planSummary(subscription.data, trialData) : null,
    meters: usageMeters(usage.data, trialData),
    balance: credits.data === undefined ? null : formatMoney(credits.data),
    isLoading,
    isError: subscription.isError,
    isRefreshing: subscription.isRefetching,
    refresh,
    goBack,
  };
}
