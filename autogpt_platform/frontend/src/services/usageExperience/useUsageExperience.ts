"use client";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import { useCopilotUsage } from "@/app/(platform)/copilot/components/UsageLimits/useCopilotUsage";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { getUsageExperience } from "./helpers";

export function useUsageExperience() {
  const usage = useCopilotUsage();
  const trial = useTrialStatus();
  const subscription = useGetSubscriptionStatus({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const experience = getUsageExperience(
    usage.data,
    trial.data,
    subscription.data?.tier,
  );
  const trialUnavailable =
    experience.tier === "TRIAL" && (trial.isError || !trial.data);
  return {
    experience,
    usage: usage.data,
    trial: trial.data,
    subscription: subscription.data,
    isBillingEnabled,
    isLoading:
      usage.isLoading || (experience.tier === "TRIAL" && trial.isLoading),
    isError:
      usage.isError || (!usage.isLoading && !usage.data) || trialUnavailable,
    retry: (throwOnError = false) =>
      Promise.all([
        usage.refetch({ throwOnError }),
        trial.refetch({ throwOnError }),
        subscription.refetch({ throwOnError }),
      ]),
  };
}
