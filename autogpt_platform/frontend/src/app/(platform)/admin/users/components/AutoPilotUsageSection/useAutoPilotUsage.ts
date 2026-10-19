"use client";

import type { SubscriptionTier } from "@/app/api/__generated__/models/subscriptionTier";
import {
  getGetV2GetUserRateLimitQueryKey,
  useGetV2GetUserRateLimit,
  usePostV2ResetUserRateLimitUsage,
  usePostV2SetUserRateLimitTier,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { getErrorMessage, getPlanName } from "../../helpers";
import type { ResetScope } from "./helpers";

export function useAutoPilotUsage(userId: string) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const rateLimit = useGetV2GetUserRateLimit(
    { user_id: userId },
    { query: { select: okData } },
  );

  const setTier = usePostV2SetUserRateLimitTier({
    mutation: {
      onSuccess: (response, { data }) => {
        const planName = getPlanName(data.tier);
        const warning = okData(response)?.warning;
        toast(
          warning
            ? {
                title: `Plan set to ${planName}, but Stripe will revert it`,
                description: warning,
                variant: "destructive",
              }
            : { title: `Plan set to ${planName}`, variant: "success" },
        );
        void queryClient.invalidateQueries({
          queryKey: getUsageQueryKey(data.user_id),
        });
      },
      onError: (error) => {
        toast({
          title: "Couldn't change the plan",
          description: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  const reset = usePostV2ResetUserRateLimitUsage({
    mutation: {
      onSuccess: (response, { data }) => {
        queryClient.setQueryData(getUsageQueryKey(data.user_id), response);
        toast({
          title: data.reset_weekly
            ? "Daily and weekly usage reset"
            : "Daily usage reset",
          variant: "success",
        });
      },
      onError: (error) => {
        toast({
          title: "Couldn't reset usage",
          description: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  function changePlan(tier: string, onSuccess: () => void) {
    setTier.mutate(
      {
        data: { user_id: userId, tier: tier as SubscriptionTier },
      },
      { onSuccess },
    );
  }

  function resetUsage(scope: ResetScope, onSuccess: () => void) {
    reset.mutate(
      { data: { user_id: userId, reset_weekly: scope === "both" } },
      { onSuccess },
    );
  }

  return {
    rateLimit: rateLimit.data,
    isLoading: rateLimit.isLoading,
    error: rateLimit.error,
    refetch: rateLimit.refetch,
    changePlan,
    isChangingPlan: setTier.isPending,
    resetUsage,
    isResetting: reset.isPending,
  };
}

// Keyed by the user a request was sent for, never the one now on screen: the
// section stays mounted across lookups, and React Query hands a pending
// mutation the latest render's callbacks.
function getUsageQueryKey(userID: string) {
  return getGetV2GetUserRateLimitQueryKey({ user_id: userID });
}
