import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useCopilotUsage } from "../useCopilotUsage";

export function useUsageLimitReachedCard() {
  const { data: usage, isSuccess } = useCopilotUsage();
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  return { usage, isSuccess, isBillingEnabled };
}
