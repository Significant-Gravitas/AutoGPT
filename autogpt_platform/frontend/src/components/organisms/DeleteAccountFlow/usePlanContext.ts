import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { getPlanContext } from "./context-helpers";

export function usePlanContext() {
  const subscription = useGetSubscriptionStatus({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
  const trial = useTrialStatus();

  return getPlanContext({
    subscription: subscription.data,
    trial: trial.data,
  });
}
